"""
Aggregates v2 results into tables (CSV + LaTeX) and figures. CPU only, runs in seconds-
minutes; fine on a login node.

Uncertainty: intervals are 95% percentile bootstraps that resample test items within each
seed and then average the seeds. They capture item-sampling uncertainty only; variability
due to the DAS initialization / data split is NOT inside them and is reported separately as
the across-seed SD (`*_seed_sd` columns; with 3 seeds, resampling seeds would be too crude).
DAS-vs-patch differences are paired on the same items.

Usage:
    python -m v2.analyze --model llama --results-root $SYCO_RESULTS            # everything
    python -m v2.analyze --model llama --results-root $SYCO_RESULTS --only behavior main
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from v2.config import CONDITIONS, MAIN_SOURCE, MENTION_TEMPLATES
from v2.data import read_jsonl

B = 2000
RNG = np.random.default_rng(0)

# Item subgroup for `--group flip|noflip`: boolean over neutral-correct rows (cache row order),
# True for the items kept. Every result file stores which rows it covers ("rows" or
# "test_rows"), so all per-item analyses can be restricted to a subgroup after the fact.
GROUP = None


def _mask(r):
    if GROUP is None:
        return None
    return GROUP[np.asarray(r["test_rows"] if "test_rows" in r else r["rows"])]


def _sel(r, arrays):
    m = _mask(r)
    return arrays if m is None else [np.asarray(x)[m] for x in arrays]


# ---- vectorized metrics over bootstrap rows [B, n] ----------------------------------------
def v_iia(mi, ms, mn):
    return np.mean((mi < 0) == (ms < 0), axis=-1)


def v_bal(mi, ms, mn):
    f, p = ms < 0, mi < 0
    with np.errstate(invalid="ignore", divide="ignore"):
        tpr = (p & f).sum(-1) / f.sum(-1)
        tnr = (~p & ~f).sum(-1) / (~f).sum(-1)
    return 0.5 * (tpr + tnr)


def v_r(mi, ms, mn):
    a = mi - mi.mean(-1, keepdims=True)
    b = ms - ms.mean(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(-1) / np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1))


def v_rec(mi, ms, mn):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (mi - mn).mean(-1) / (ms - mn).mean(-1)


def v_rshift(mi, ms, mn):
    """Correlation of SHIFTS (m_int - m_neu vs. m_src - m_neu). Unlike v_r it is not
    inflated by the neutral margin both terms share (full patching at block 0 changes
    nothing yet has v_r ~ 0.58)."""
    return v_r(mi - mn, ms - mn, mn)


METRICS = {"iia": v_iia, "balanced_acc": v_bal, "pearson_r": v_r, "shift_r": v_rshift,
           "shift_recovered": v_rec}


def boot(per_seed, fn, paired_with=None):
    """per_seed: list of (m_int, m_src, m_neu) arrays (one tuple per seed). Returns
    (point, lo, hi, seed_sd); if paired_with is given, statistics are of fn(a) - fn(b)."""
    pts, bs = [], []
    for s, tup in enumerate(per_seed):
        n = len(tup[0])
        if n == 0:  # e.g. a subgroup (--group) with no items in this seed's test split
            continue
        idx = RNG.integers(0, n, (B, n))
        val = fn(*[np.asarray(x)[None] for x in tup])[0]
        bv = fn(*[np.asarray(x)[idx] for x in tup])
        if paired_with is not None:
            ot = paired_with[s]
            val = val - fn(*[np.asarray(x)[None] for x in ot])[0]
            bv = bv - fn(*[np.asarray(x)[idx] for x in ot])
        pts.append(val)
        bs.append(bv)
    if not pts:
        return (float("nan"),) * 4
    bs = np.nanmean(np.stack(bs), axis=0)
    sd = float(np.std(pts, ddof=1)) if len(pts) > 1 else float("nan")
    return float(np.mean(pts)), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), sd


def boot_mean(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"),) * 3
    bm = x[RNG.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def fmt(t, pct=False):
    s = 100 if pct else 1
    return f"{t[0] * s:.1f} [{t[1] * s:.1f}, {t[2] * s:.1f}]" if pct else f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))  # union: configurations differ in columns
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"  wrote {path}")


# ---- behavior & entrainment ablation -------------------------------------------------------
def analyze_behavior(root: Path, out: Path):
    from scipy.stats import chi2_contingency
    beh = root / "behavior"
    meta = json.loads((beh / "meta.json").read_text())
    items = read_jsonl(beh / "items.jsonl")
    nc = [items[i] for i in meta["nc_ids"]]
    lines = [f"# Behavior ({meta['hf_id']})",
             f"Neutral-correct: {len(nc)}/{len(items)} ({len(nc) / len(items):.1%}); "
             f"by dataset: " + ", ".join(
                 f"{d} {sum(it['dataset'] == d for it in nc)}/{sum(it['dataset'] == d for it in items)}"
                 for d in sorted({it['dataset'] for it in items})), ""]
    rows = []
    conds = [c for c in CONDITIONS if c in nc[0]["margin"]]  # older runs lack the content-free ones
    D = {c: np.array([it["margin"][c] - it["margin"]["neutral"] for it in nc]) for c in conds}
    for c in conds:
        if c == "neutral":
            continue
        k = sum(it["margin"][c] < 0 for it in nc)
        dcp = [np.mean(it["lp"][c]["c_plus"]) - np.mean(it["lp"]["neutral"]["c_plus"]) for it in nc]
        dcm = [np.mean(it["lp"][c]["c_minus"]) - np.mean(it["lp"]["neutral"]["c_minus"]) for it in nc]
        rows.append({"condition": c, "flip_rate": fmt(wilson(k, len(nc)), True),
                     "margin_shift": fmt(boot_mean(D[c])), "d_lp_correct": fmt(boot_mean(dcp)),
                     "d_lp_incorrect": fmt(boot_mean(dcm))})
    write_csv(out / "behavior_conditions.csv", rows)
    lines += ["## Conditions (neutral-correct items; 95% CI)", ""] + [
        f"- {r['condition']}: flip {r['flip_rate']}%, shift {r['margin_shift']}, "
        f"dlp(c+) {r['d_lp_correct']}, dlp(c-) {r['d_lp_incorrect']}" for r in rows]

    dec = []
    for t in MENTION_TEMPLATES:
        mp, mi = D[f"mention_plausible_{t}"], D[f"mention_irrelevant_{t}"]
        parts = {"entrainment_of_c_minus": mp - mi,
                 "generic_assertion_pressure": D["assert_irrelevant"] - mi,
                 "content_specific_agreement": D["assert_plausible"] - D["assert_irrelevant"] - mp + mi,
                 "total_assert_plausible": D["assert_plausible"]}
        for name, v in parts.items():
            dec.append({"mention_template": t, "component": name, "estimate": fmt(boot_mean(v))})
    write_csv(out / "behavior_decomposition.csv", dec)
    lines += ["", "## 2x2 decomposition of the margin shift (negative = toward c_minus)", ""] + [
        f"- template {d['mention_template']} {d['component']}: {d['estimate']}" for d in dec]

    # paired contrasts between conditions (same items, so the item-level difference is
    # bootstrapped): certainty markers before vs. after the answer, against the unmarked one
    contrasts = [(a, b) for a, b in [("hedge_post", "hedge_none"), ("sure_post", "hedge_none"),
                                     ("hedge_pre", "hedge_none"), ("sure_pre", "hedge_none"),
                                     ("hedge_post", "hedge_pre"), ("sure_post", "sure_pre"),
                                     ("sure_post", "hedge_post"), ("sure_pre", "hedge_pre"),
                                     # same words, only the order of the two sentences differs
                                     ("hedge_post_sep", "hedge_none"), ("sure_post_sep", "hedge_none"),
                                     ("hedge_pre_sep", "hedge_none"), ("sure_pre_sep", "hedge_none"),
                                     ("hedge_post_sep", "hedge_pre_sep"), ("sure_post_sep", "sure_pre_sep"),
                                     ("sure_post_sep", "hedge_post_sep"), ("sure_pre_sep", "hedge_pre_sep")]
                 if a in conds and b in conds]
    con = []
    for a, b in contrasts:
        row = {"contrast": f"{a} - {b}"}
        for key, name in [("margin", "shift_diff"), ("margin_first", "first_token_shift_diff")]:
            d = np.array([it[key][a] - it[key][b] for it in nc])
            row[name] = fmt(boot_mean(d))
        fl = np.array([float(it["margin"][a] < 0) - float(it["margin"][b] < 0) for it in nc])
        row["flip_rate_diff_pct"] = fmt(tuple(100 * x for x in boot_mean(fl)))
        con.append(row)
    if con:
        write_csv(out / "behavior_contrasts.csv", con)
        lines += ["", "## Paired contrasts (same items; negative shift diff = more toward c_minus)", ""] + [
            f"- {r['contrast']}: shift {r['shift_diff']}, first token {r['first_token_shift_diff']}, "
            f"flip rate {r['flip_rate_diff_pct']} pp" for r in con]

    # v1-style control: correct vs. the asserted irrelevant answer itself
    both = [it for it in nc if it["margin"]["neutral__vs_r"] > 0]
    f_syc = np.array([it["margin"]["assert_plausible"] < 0 for it in both])
    f_ent = np.array([it["margin"]["assert_irrelevant__vs_r"] < 0 for it in both])
    tab = np.array([[np.sum(f_syc & f_ent), np.sum(f_syc & ~f_ent)],
                    [np.sum(~f_syc & f_ent), np.sum(~f_syc & ~f_ent)]])
    chi2, p, _, _ = chi2_contingency(tab, correction=False)
    phi = np.sqrt(chi2 / tab.sum())
    jac = tab[0, 0] / max(1, tab[0, 0] + tab[0, 1] + tab[1, 0])
    lines += ["", "## v1-style control (c_plus vs. asserted irrelevant r)", "",
              f"- n = {len(both)} items correct under neutral for both pairs",
              f"- flip rate plausible {f_syc.mean():.1%}, irrelevant (vs r) {f_ent.mean():.1%}",
              f"- overlap Jaccard {jac:.1%}, chi2 p = {p:.3g}, phi = {phi:.3f}"]

    # scoring-rule sensitivity
    lines += ["", "## Scoring-rule sensitivity (assert_plausible)", ""]
    base = np.array([it["margin"][MAIN_SOURCE] < 0 for it in nc])
    for key, name in [("margin", "mean log-prob"), ("margin_sum", "sum log-prob"),
                      ("margin_first", "first token")]:
        nc_k = [it for it in items if it[key]["neutral"] > 0]
        fr = np.mean([it[key][MAIN_SOURCE] < 0 for it in nc_k])
        agree = np.mean([(it[key][MAIN_SOURCE] < 0) == b for it, b in zip(nc, base)])
        lines.append(f"- {name}: neutral-correct {len(nc_k)}, flip rate {fr:.1%}, "
                     f"label agreement with mean-log-prob on its items {agree:.1%}")
    (out / "behavior.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- interventions -------------------------------------------------------------------------
def load_runs(root: Path, tag: str):
    runs = defaultdict(dict)  # (method, k, source) -> {(seed, block): result}
    for f in sorted((root / tag).glob("seed*/block*.json")):
        r = json.loads(f.read_text())
        key = ("patch", None, None) if r["method"] == "patch" else ("das", r["k"], r["source"])
        runs[key][(r["seed"], r["block"])] = r
    return runs


def tuples(res_by_seed, src, part=""):
    """part: "" (mean-token margin), "_first" or "_rest". For "_rest", items whose answers
    are single tokens (NaN) are dropped; the mask depends only on the items, so DAS and
    patch tuples of the same seed stay paired."""
    out = []
    for r in res_by_seed:
        e = r["eval"][src]
        t = _sel(r, [np.array(e[f"m_int{part}"], dtype=float), np.array(e[f"m_src{part}"], dtype=float),
                     np.array(r[f"m_neutral{part}"], dtype=float)])
        ok = np.all([np.isfinite(x) for x in t], axis=0)
        out.append(tuple(x[ok] for x in t))
    return out


def analyze_main(root: Path, out: Path, k=64, src=MAIN_SOURCE):
    runs = load_runs(root, "main")
    das, patch = runs.get(("das", k, src), {}), runs.get(("patch", None, None), {})
    blocks = sorted({b for _, b in das} & {b for _, b in patch})
    if not blocks:
        print("  main: no completed blocks yet")
        return
    rows = []
    for b in blocks:
        seeds = sorted({s for s, bb in das if bb == b} & {s for s, bb in patch if bb == b})
        td = tuples([das[(s, b)] for s in seeds], src)
        tp = tuples([patch[(s, b)] for s in seeds], src)
        row = {"block": b, "n_seeds": len(seeds),
               "n_test": len(td[0][0]),
               "no_intervention_iia": float(np.mean([np.mean(t[1] >= 0) for t in td])),
               "untrained_iia": float(np.mean([das[(s, b)]["untrained"]["iia"] for s in seeds]))}
        for m, fn in METRICS.items():
            for name, tt in [("das", td), ("patch", tp)]:
                (row[f"{name}_{m}"], row[f"{name}_{m}_lo"], row[f"{name}_{m}_hi"],
                 row[f"{name}_{m}_seed_sd"]) = boot(tt, fn)
            (row[f"diff_{m}"], row[f"diff_{m}_lo"], row[f"diff_{m}_hi"],
             row[f"diff_{m}_seed_sd"]) = boot(td, fn, paired_with=tp)
        # Limitation 1: is the late-layer patching advantage carried by the first answer
        # token (read directly off the patched position) or also by the later tokens?
        for part in ["_first", "_rest"]:
            tdp = tuples([das[(s, b)] for s in seeds], src, part)
            tpp = tuples([patch[(s, b)] for s in seeds], src, part)
            for name, tt in [("das", tdp), ("patch", tpp)]:
                (row[f"{name}_shift_recovered{part}"], row[f"{name}_shift_recovered{part}_lo"],
                 row[f"{name}_shift_recovered{part}_hi"], _) = boot(tt, v_rec)
            row[f"n_test{part}"] = len(tdp[0][0])
        rows.append(row)
    write_csv(out / f"main_k{k}.csv", rows)
    _plot_main(rows, out / f"fig_main_k{k}.png")
    _plot_first_rest(rows, out / f"fig_first_vs_rest_k{k}.png")
    _latex_main(rows, out / f"table_main_k{k}.tex")


def _plot_main(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["block"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, m, lab in zip(axes, ["iia", "balanced_acc", "shift_r"], ["IIA", "Balanced accuracy", "Shift correlation"]):
        for name, col, mk in [("patch", "tab:blue", "o"), ("das", "tab:red", "s")]:
            y = [r[f"{name}_{m}"] for r in rows]
            ax.plot(x, y, color=col, marker=mk, ms=3, label={"patch": "Full patching", "das": "DAS (k=64)"}[name])
            ax.fill_between(x, [r[f"{name}_{m}_lo"] for r in rows], [r[f"{name}_{m}_hi"] for r in rows],
                            color=col, alpha=0.15, lw=0)
        if m == "iia":
            ax.plot(x, [r["no_intervention_iia"] for r in rows], "k--", lw=1, label="No intervention")
            ax.plot(x, [1 - r["no_intervention_iia"] for r in rows], ":", color="gray", lw=1, label="Always flip")
        if m == "balanced_acc":
            ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_xlabel("Decoder block")
        ax.set_title(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {path}")


def _plot_first_rest(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["block"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for ax, part, title in zip(axes, ["", "_first", "_rest"],
                               ["All answer tokens", "First answer token", "Answer tokens 2..k"]):
        for name, col in [("patch", "tab:blue"), ("das", "tab:red")]:
            key = f"{name}_shift_recovered{part}"
            ax.plot(x, [r[key] for r in rows], color=col, marker="o", ms=3,
                    label={"patch": "Full patching", "das": "DAS (k=64)"}[name])
            ax.fill_between(x, [r[key + "_lo"] for r in rows], [r[key + "_hi"] for r in rows],
                            color=col, alpha=0.15, lw=0)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(1, color="k", lw=0.8, ls=":")
        ax.set_title(title + (f" (n={rows[0]['n_test_rest']}/seed)" if part == "_rest" else ""))
        ax.set_xlabel("Decoder block")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Fraction of margin shift recovered")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {path}")


def _latex_main(rows, path):
    lines = [r"\begin{tabular}{rcccc}", r"\toprule",
             r"Block & DAS IIA & Patch IIA & $\Delta$ IIA (P$-$D) & DAS / Patch $r$ \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['block']} & {100 * r['das_iia']:.1f} & {100 * r['patch_iia']:.1f} & "
                     f"${-100 * r['diff_iia']:+.1f}$ [{-100 * r['diff_iia_hi']:+.1f}, {-100 * r['diff_iia_lo']:+.1f}] & "
                     f"{r['das_pearson_r']:.2f} / {r['patch_pearson_r']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def analyze_rank(root: Path, out: Path, src=MAIN_SOURCE):
    runs = load_runs(root, "rank")
    patch = load_runs(root, "main").get(("patch", None, None), {})
    rows = []
    for (method, k, s), res in sorted(runs.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        if method != "das" or s != src:
            continue
        for b in sorted({bb for _, bb in res}):
            seeds = sorted(ss for ss, bb in res if bb == b)
            t = tuples([res[(ss, b)] for ss in seeds], src)
            row = {"block": b, "k": k, "n_seeds": len(seeds)}
            for m in ["iia", "pearson_r", "shift_recovered"]:
                row[m], row[m + "_lo"], row[m + "_hi"], row[m + "_seed_sd"] = boot(t, METRICS[m])
            ps = [ss for ss in seeds if (ss, b) in patch]
            if ps:
                row["patch_pearson_r"] = boot(tuples([patch[(ss, b)] for ss in ps], src), v_r)[0]
            rows.append(row)
    if not rows:
        print("  rank: no results yet")
        return
    write_csv(out / "rank.csv", rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for b in sorted({r["block"] for r in rows}):
        rr = sorted((r for r in rows if r["block"] == b), key=lambda r: r["k"])
        line, = ax.plot([r["k"] for r in rr], [r["pearson_r"] for r in rr], marker="o", ms=3, label=f"block {b}")
        if "patch_pearson_r" in rr[0]:
            ax.axhline(rr[0]["patch_pearson_r"], color=line.get_color(), ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("DAS rank k (dotted: full patching)")
    ax.set_ylabel("Pearson r")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_rank.png", dpi=200)
    plt.close(fig)


def analyze_transfer(root: Path, out: Path, k=64):
    """Train-source x eval-source matrix per block (shift recovered, r), and subspace overlap
    between DAS solutions trained on different sources: mean cos^2 of principal angles
    = ||W1^T W2||_F^2 / k (random subspaces: k/d)."""
    import torch
    runs = {}
    for tag in ["main", "ablation"]:
        for key, res in load_runs(root, tag).items():
            if key[0] == "das" and key[1] == k:
                runs.setdefault(key[2], {}).update(res)
    if len(runs) < 2:
        print("  transfer: need main + ablation results")
        return
    rows, ov = [], []
    srcs = sorted(runs)
    blocks = sorted({b for res in runs.values() for _, b in res})
    for b in blocks:
        for tr in srcs:
            seeds = sorted(s for s, bb in runs[tr] if bb == b)
            if not seeds:
                continue
            for ev in runs[tr][(seeds[0], b)]["eval"]:
                t = tuples([runs[tr][(s, b)] for s in seeds], ev)
                rows.append({"block": b, "train_source": tr, "eval_source": ev, "n_seeds": len(seeds),
                             "shift_recovered": boot(t, v_rec)[0], "pearson_r": boot(t, v_r)[0]})
        # Sources trained with the same seed share their initialization and stay close to it
        # (see init_check), so only pairs from DIFFERENT seeds measure learned overlap; the
        # same-seed value is kept for reference only.
        def w(src, s):
            f = root / ("main" if src == MAIN_SOURCE else "ablation") / f"seed{s}" / f"block{b:02d}__das__k{k}__src-{src}.W.pt"
            return torch.load(f).float() if f.exists() else None
        for i, a in enumerate(srcs):
            for c in srcs[i + 1:]:
                sa = [s for s, bb in runs[a] if bb == b]
                sc = [s for s, bb in runs[c] if bb == b]
                Wa_, Wc_ = {s: w(a, s) for s in sa}, {s: w(c, s) for s in sc}
                same, diff = [], []
                for s1, Wa in Wa_.items():
                    for s2, Wc in Wc_.items():
                        if Wa is not None and Wc is not None:
                            (same if s1 == s2 else diff).append(float((Wa.T @ Wc).pow(2).sum() / k))
                if diff:
                    ov.append({"block": b, "source_a": a, "source_b": c, "overlap": float(np.mean(diff)),
                               "same_seed_overlap_confounded": float(np.mean(same)) if same else float("nan"),
                               "random_baseline": k / next(x for x in Wa_.values() if x is not None).shape[0],
                               "n_pairs": len(diff)})
    write_csv(out / f"transfer_k{k}.csv", rows)
    write_csv(out / f"subspace_overlap_k{k}.csv", ov)
    write_csv(out / f"subspace_init_check_k{k}.csv", init_check(root, runs, blocks, k))


def init_check(root: Path, runs, blocks, k):
    """Controls for subspace_overlap: all sources at the same (seed, block, k) start from the
    SAME initialization (see run_intervention.init_seed), so a high cross-source overlap
    could be inherited from the init. Reports, per block and source, the overlap of each
    trained W with its own init (`to_init`) and between seeds of the same source, which
    start from different inits (`cross_seed`)."""
    import torch
    from v2.das import Subspace
    rows = []
    for b in blocks:
        for src in sorted(runs):
            tag = "main" if src == MAIN_SOURCE else "ablation"
            Ws, to_init = {}, []
            for s in sorted({s for s, bb in runs[src] if bb == b}):
                f = root / tag / f"seed{s}" / f"block{b:02d}__das__k{k}__src-{src}.W.pt"
                if not f.exists():
                    continue
                W = torch.load(f).float()
                W0 = Subspace(W.shape[0], k, 100_000 * s + 100 * b + k)().detach()
                to_init.append(float((W.T @ W0).pow(2).sum() / k))
                Ws[s] = W
            ss = sorted(Ws)
            cross = [float((Ws[a].T @ Ws[c]).pow(2).sum() / k) for i, a in enumerate(ss) for c in ss[i + 1:]]
            if to_init:
                rows.append({"block": b, "source": src, "to_init": float(np.mean(to_init)),
                             "cross_seed": float(np.mean(cross)) if cross else float("nan"),
                             "random_baseline": k / next(iter(Ws.values())).shape[0], "n_seeds": len(ss)})
    return rows


def analyze_illusion(root: Path, out: Path):
    """Trained W fed sources that should not reproduce the item's shift (forward), and the
    necessity test (reverse: biased base, subspace set to a non-biased source). See
    v2/run_illusion_control.py. Both store m_int / m_src (aim) / m_neutral (base run)."""
    found = False
    folders = sorted(d.name for d in root.glob("illusion*") if d.is_dir())
    for folder in folders:
        name = folder
        ylabel = ("First-token shift removed" if folder.startswith("illusion_reverse") else
                  "First-token shift recovered") + "\n(relative to item's own shift)"
        files = sorted((root / folder).glob("seed*/block*.json"))
        if not files:
            continue
        found = True
        _illusion_dir(files, out, name, ylabel)
    if not found:
        print("  illusion: no results yet")


def _illusion_dir(files, out, name, ylabel):
    runs = defaultdict(list)
    for f in files:
        r = json.loads(f.read_text())
        runs[(r["block"], r["method"])].append(r)
    rows = []
    for (b, method), rs in sorted(runs.items()):
        row = {"block": b, "method": method, "n_seeds": len(rs)}
        for c in rs[0]["controls"]:
            for part, sfx in [("", ""), ("_first", "_first")]:
                t = [tuple(_sel(r, [np.array(r["controls"][c][f"m_int{part}"]), np.array(r[f"m_src{part}"]),
                                    np.array(r[f"m_neutral{part}"])])) for r in rs]
                (row[f"{c}{sfx}"], row[f"{c}{sfx}_lo"], row[f"{c}{sfx}_hi"], _) = boot(t, v_rec)
                # log-prob units: needed when the condition's own shift is small (content-free)
                (row[f"{c}{sfx}_abs"], row[f"{c}{sfx}_abs_lo"], row[f"{c}{sfx}_abs_hi"], _) = boot(t, v_abs)
                if c == "matched":  # the condition's own shift, for reference
                    row[f"shift{sfx}_abs"] = float(np.mean([np.mean(x[1] - x[2]) for x in t]))
        rows.append(row)
    write_csv(out / f"{name}.csv", rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    styles = {"matched": "-", "other_biased": "--", "other_neutral": ":"}
    for ax, method in zip(axes, ["das", "patch"]):
        rr = [r for r in rows if r["method"] == method]
        x = [r["block"] for r in rr]
        for c, ls in styles.items():
            if not rr or f"{c}_first" not in rr[0]:
                continue
            ax.plot(x, [r[f"{c}_first"] for r in rr], ls, marker="o", ms=2.5, label=c.replace("_", " "))
            ax.fill_between(x, [r[f"{c}_first_lo"] for r in rr], [r[f"{c}_first_hi"] for r in rr], alpha=0.12, lw=0)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title({"das": "DAS (k=64)", "patch": "Full patching"}[method] + ": source")
        ax.set_xlabel("Decoder block")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"fig_{name}.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {out / f'fig_{name}.png'}")


def v_abs(mi, ms, mn):
    """Absolute counterpart of v_rec: mean margin change in log-prob units."""
    return (mi - mn).mean(-1)


def _cand_parts(runs):
    """Per-candidate first-token parts, if the runs stored them (newer run_knockout/run_tracing)."""
    r = next(iter(runs.values()))[0]
    ok = "m_neutral_first_plus" in r and "first_plus" in next(iter(r["margins"].values()))
    return ["first_plus", "first_minus"] if ok else []


def _position_rows(runs, conditions, ref, parts=("mean", "first", "rest")):
    """runs: block -> list of result dicts (one per seed / noise draw). One row per block:
    for each column, v_rec(m_cond, m_target, m_base) and its absolute counterpart with
    item-bootstrap CIs (seeds averaged, see boot), per margin part. `conditions` maps a
    column to (condition key, target, base), where target/base name a run in `margins` or
    a stored reference array (`ref` maps names to JSON fields)."""
    rows = []
    for b in sorted(runs):
        rs = runs[b]

        def arr(r, name, part):
            sfx = "" if part == "mean" else f"_{part}"
            return np.array(r[ref[name] + sfx] if name in ref else r["margins"][name][part], dtype=float)
        row = {"block": b, "n_items": len(rs[0]["rows"]), "n_seeds": len(rs)}
        for col, (cond, target, base) in conditions.items():
            for part in parts:
                ts = []
                for r in rs:
                    t = _sel(r, [arr(r, cond, part), arr(r, target, part), arr(r, base, part)])
                    ok = np.all([np.isfinite(x) for x in t], axis=0)
                    ts.append(tuple(x[ok] for x in t))
                sfx = "" if part == "mean" else f"_{part}"
                row[f"{col}{sfx}"], row[f"{col}{sfx}_lo"], row[f"{col}{sfx}_hi"], _ = boot(ts, v_rec)
                row[f"{col}{sfx}_abs"], row[f"{col}{sfx}_abs_lo"], row[f"{col}{sfx}_abs_hi"], _ = boot(ts, v_abs)
        rows.append(row)
    return rows


def _plot_positions(rows, panels, path, ylabel):
    """panels: list of (title, [(column, label), ...]); first-token and rest columns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["block"] for r in rows]
    fig, axes = plt.subplots(len(panels), 2, figsize=(11, 3.2 * len(panels)), sharex=True, squeeze=False)
    for i, (title, cols) in enumerate(panels):
        for j, (part, ptitle) in enumerate([("_first", "first answer token"), ("_rest", "answer tokens 2..k")]):
            ax = axes[i, j]
            for col, label in cols:
                ax.plot(x, [r[col + part] for r in rows], marker="o", ms=2.5, label=label)
                ax.fill_between(x, [r[col + part + "_lo"] for r in rows], [r[col + part + "_hi"] for r in rows],
                                alpha=0.12, lw=0)
            ax.axhline(0, color="k", lw=0.8)
            ax.axhline(1, color="k", lw=0.8, ls=":")
            ax.set_title(f"{title}: {ptitle}")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(ylabel)
        axes[i, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Decoder block")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {path}")


def _load_position_runs(root, kind, legacy_tag, config_of):
    """config -> block -> [result dicts]; `config_of(tag)` drops the seed from a run tag so
    repeated draws of one configuration are pooled."""
    from v2.positions import run_dirs
    out = defaultdict(lambda: defaultdict(list))
    for tag, d in run_dirs(root, kind, legacy_tag).items():
        for f in sorted(d.glob("block*.json")):
            r = json.loads(f.read_text())
            out[config_of(tag)][r["block"]].append(r)
    return out


def _with_config(config, rows):
    return [{"config": config, **r} for r in rows]


def analyze_tracing(root: Path, out: Path):
    """Causal tracing (v2/run_tracing.py): fraction of the clean-vs-corrupted margin
    difference restored / removed by setting one (block, position group), per
    configuration <condition>__<corruption>, pooling seeds."""
    from v2.run_tracing import LEGACY_TAG
    configs = _load_position_runs(root, "tracing", LEGACY_TAG, lambda t: t.rsplit("__s", 1)[0])
    if not configs:
        print("  tracing: no results yet")
        return
    all_rows = {}
    for cfg, runs in sorted(configs.items()):
        groups = next(iter(runs.values()))[0]["groups"]
        conds = {"corruption_strength": ("corr", "neutral", "clean")}
        conds |= {f"restore_{g}": (f"restore/{g}", "clean", "corr") for g in groups}
        conds |= {f"remove_{g}": (f"remove/{g}", "corr", "clean") for g in groups}
        rows = _position_rows(runs, conds, {"neutral": "m_neutral"}, ("mean", "first", "rest", *_cand_parts(runs)))
        all_rows[cfg] = rows
        sfx = "" if cfg == LEGACY_TAG.rsplit("__s", 1)[0] else f"__{cfg}"
        _plot_positions(rows, [(f"Restore clean state (sufficiency)", [(f"restore_{g}", g) for g in groups]),
                               (f"Remove: insert corrupted state (necessity)", [(f"remove_{g}", g) for g in groups])],
                        out / f"fig_tracing{sfx}.png", f"{cfg}\nfraction of clean-corrupted diff.")
    write_csv(out / "tracing.csv", [x for cfg, rows in sorted(all_rows.items()) for x in _with_config(cfg, rows)])
    return all_rows


def analyze_knockout(root: Path, out: Path):
    """Attention knockout (v2/run_knockout.py): fraction (and absolute log-prob amount) of
    the shift (condition - neutral margin) eliminated when later positions cannot attend to
    the inserted text in a window of blocks, per condition."""
    from v2.run_knockout import KEYS, LEGACY_TAG, QUERIES, WINDOWS
    configs = _load_position_runs(root, "knockout", LEGACY_TAG, lambda t: t)
    if not configs:
        print("  knockout: no results yet")
        return
    all_rows = {}
    for cfg, runs in sorted(configs.items()):
        first = next(iter(runs.values()))[0]
        keys = first.get("keys", KEYS)
        variants = [tuple(v) for v in first.get("variants", [])] or \
            [(w, k, q) for w in WINDOWS for k in keys for q in QUERIES]
        conds = {f"{w}_{k}_{q}": (f"{w}/{k}/{q}", "neutral", "biased") for w, k, q in variants}
        conds["shift"] = ("biased", "neutral", "neutral")  # absolute shift: mean(m_cond - m_neutral)
        cand = _cand_parts(runs)
        rows = _position_rows(runs, conds, {"neutral": "m_neutral"}, ("mean", "first", "rest", *cand))
        if cand:
            _plot_candidates(rows, [v for v in variants if v[0] == "from"], out / f"fig_knockout_candidates__{cfg}.png", cfg)
        for r in rows:  # `shift` is only meaningful in log-prob units (its fraction is x/0)
            for k in [k for k in r if k.startswith("shift") and "_abs" not in k]:
                del r[k]
        all_rows[cfg] = rows
        w = next(iter(runs.values()))[0]["window"]
        sfx = "" if cfg == LEGACY_TAG else f"__{cfg}"
        if cfg.endswith("__routes"):  # every variant, labelled key <- query
            panel = lambda win: [(f"{a}_{k}_{q}", f"{q} -/-> {k}") for a, k, q in variants if a == win]
        else:
            panel = lambda win: [(f"{win}_{k}_after", k) for k in keys if (win, k, "after") in variants]
        _plot_positions(rows, [(f"Blocked from block b on", panel("from")),
                               (f"Blocked in blocks b..b+{w - 1}", panel("win"))],
                        out / f"fig_knockout{sfx}.png", f"{cfg}\nfraction of shift eliminated")
    write_csv(out / "knockout.csv", [x for cfg, rows in sorted(all_rows.items()) for x in _with_config(cfg, rows)])
    comparable = {c: r for c, r in all_rows.items() if r and "from_span_after_first" in r[0]}
    if len(comparable) > 1:
        _plot_condition_compare(comparable, analyze_tracing_rows(root), out / "fig_condition_compare.png")


def _plot_candidates(rows, variants, path, cfg):
    """Knockout from block b: fraction of the drop of c_plus and of the rise of c_minus (first
    token) that is eliminated. Separates when the assertion's effect on the model's own answer
    and on the asserted answer is read."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["block"] for r in rows]
    fig, axes = plt.subplots(1, len(variants), figsize=(4 * len(variants), 3.2), squeeze=False)
    for ax, (w, k, q) in zip(axes[0], variants):
        col = f"{w}_{k}_{q}"
        for part, lab in [("first_plus", "drop of c+"), ("first_minus", "rise of c-"), ("first", "margin")]:
            ax.plot(x, [r[f"{col}_{part}"] for r in rows], marker="o", ms=2.5, label=lab)
            ax.fill_between(x, [r[f"{col}_{part}_lo"] for r in rows], [r[f"{col}_{part}_hi"] for r in rows],
                            alpha=0.12, lw=0)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(1, color="k", lw=0.8, ls=":")
        ax.set_title(f"{cfg}: {q} -/-> {k} from b", fontsize=9)
        ax.set_xlabel("Decoder block")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("fraction eliminated")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {path}")


def analyze_tracing_rows(root):
    """Noise-tracing rows per condition (seeds pooled), without writing files."""
    from v2.run_tracing import LEGACY_TAG
    configs = _load_position_runs(root, "tracing", LEGACY_TAG, lambda t: t.rsplit("__s", 1)[0])
    out = {}
    for cfg, runs in configs.items():
        cond, corruption = cfg.split("__", 1)
        if corruption == "noise":
            out[cond] = _position_rows(runs, {"restore_last": ("restore/last", "clean", "corr")},
                                       {"neutral": "m_neutral"})
    return out


def _plot_condition_compare(ko, tr, path):
    """Is the late read of the inserted answer specific to assertions? One line per
    condition: knockout from block b (first token; fraction and log-prob units) and, if
    available, noise tracing of the last token."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("Knockout from block b: fraction of shift eliminated", ko, "from_span_after_first", False),
              ("Knockout from block b: log-prob eliminated", ko, "from_span_after_first_abs", False),
              ("Tracing: restore last token (fraction)", tr, "restore_last_first", True)]
    panels = [p for p in panels if p[1]]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.4), squeeze=False)
    for ax, (title, data, col, _) in zip(axes[0], panels):
        for cfg, rows in sorted(data.items()):
            x = [r["block"] for r in rows]
            ax.plot(x, [r[col] for r in rows], marker="o", ms=2.5, label=cfg)
            ax.fill_between(x, [r[col + "_lo"] for r in rows], [r[col + "_hi"] for r in rows], alpha=0.12, lw=0)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Decoder block")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {path}")


def analyze_answer_direction(root: Path, out: Path, k=64):
    """See v2/run_answer_direction.py. Per block: how much of Delta points at the item's own
    answer direction (vs. another item's), how much of the answer directions of test vs.
    training items lies in span(W), and DAS / patching first-token recovery for test items
    whose c_minus first token was or was not seen in training."""
    files = sorted((root / "answer_direction").glob("seed*.json"))
    if not files:
        print("  answer_direction: no results yet")
        return
    runs = [json.loads(f.read_text()) for f in files]
    main = load_runs(root, "main")
    das, patch = main.get(("das", k, MAIN_SOURCE), {}), main.get(("patch", None, None), {})
    rows = []
    for b in sorted(int(x) for x in runs[0]["blocks"]):
        row = {"block": b, "n_seeds": len(runs)}
        for key in ["delta_on_own", "delta_on_other", "w_on_test", "w_on_train", "delta_in_w"]:
            vals = [np.mean(r["blocks"][str(b)][key]) for r in runs if key in r["blocks"][str(b)]]
            row[key] = float(np.mean(vals)) if vals else float("nan")
        row["w_chance"] = runs[0]["blocks"][str(b)].get("w_chance", float("nan"))
        for name, res in [("das", das), ("patch", patch)]:
            for grp, want in [("seen", True), ("unseen", False)]:
                ts = []
                for r in runs:
                    jr = res.get((r["seed"], b))
                    if jr is None or jr["test_rows"] != r["test_rows"]:
                        continue
                    mask = np.array(r["seen"]) == want
                    e = jr["eval"][MAIN_SOURCE]
                    ts.append(tuple(np.array(x, dtype=float)[mask] for x in
                                    (e["m_int_first"], e["m_src_first"], jr["m_neutral_first"])))
                if ts:
                    row[f"{name}_rec_first_{grp}"], row[f"{name}_rec_first_{grp}_lo"], \
                        row[f"{name}_rec_first_{grp}_hi"], _ = boot(ts, v_rec)
        row["n_seen"] = int(np.mean([sum(r["seen"]) for r in runs]))
        row["n_unseen"] = int(np.mean([len(r["seen"]) - sum(r["seen"]) for r in runs]))
        rows.append(row)
    write_csv(out / "answer_direction.csv", rows)


def analyze_answer_patch(root: Path, out: Path, k=64):
    """Item-specific answer-direction interventions (v2/run_answer_patch.py) next to DAS (k)
    and full patching from the illusion control (same test items): first-token shift
    recovered. patch_specific = full patching with the matched source minus with another
    item's neutral source (the part of full patching that depends on the item)."""
    files = sorted((root / "answer_patch").glob("seed*/block*.json"))
    if not files:
        print("  answer_patch: no results yet")
        return
    by_block = defaultdict(list)
    for f in files:
        r = json.loads(f.read_text())
        by_block[r["block"]].append(r)
    ill = defaultdict(dict)
    for f in sorted((root / "illusion").glob("seed*/block*.json")):
        r = json.loads(f.read_text())
        ill[(r["block"], r["method"])][r["seed"]] = r
    rows = []
    for b in sorted(by_block):
        rs = by_block[b]
        row = {"block": b, "n_seeds": len(rs)}
        for v in rs[0]["variants"]:
            t = [tuple(_sel(r, [np.array(x, dtype=float) for x in
                                (r["variants"][v]["m_int_first"], r["m_src_first"], r["m_neutral_first"])]))
                 for r in rs]
            row[v], row[f"{v}_lo"], row[f"{v}_hi"], _ = boot(t, v_rec)
        for method, ctrl, name in [("das", "matched", "das"), ("patch", "matched", "patch"),
                                   ("patch", "other_neutral", "patch_on")]:
            ts = [tuple(_sel(ir, [np.array(x, dtype=float) for x in
                                  (ir["controls"][ctrl]["m_int_first"], ir["m_src_first"], ir["m_neutral_first"])]))
                  for s, ir in ill.get((b, method), {}).items() if ctrl in ir["controls"]]
            if ts:
                row[name], row[f"{name}_lo"], row[f"{name}_hi"], _ = boot(ts, v_rec)
        if "patch" in row and "patch_on" in row:
            row["patch_specific"] = row["patch"] - row["patch_on"]
        rows.append(row)
    write_csv(out / "answer_patch.csv", rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = [r["block"] for r in rows]
    for col, label, ls in [("dir1", "own answer direction (1-d)", "-"), ("dir2", "own answer subspace (2-d)", "-"),
                           ("dir2_other", "another item's answer subspace", ":"),
                           ("dir2_on", "own subspace, other item's neutral source", ":"),
                           ("das", f"DAS (k={k}, shared)", "--"), ("patch", "full patching", "--"),
                           ("patch_specific", "full patching, item-specific part", "-.")]:
        if col in rows[0]:
            ax.plot(x, [r[col] for r in rows], ls, marker="o", ms=2.5, label=label)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Decoder block")
    ax.set_ylabel("First-token shift recovered")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "fig_answer_patch.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {out / 'fig_answer_patch.png'}")


def analyze_probe(root: Path, out: Path):
    f = root / "probe.json"
    if not f.exists():
        print("  probe: no probe.json yet")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.loads(f.read_text())
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for cond, rows in data.items():
        x = [r["block"] for r in rows]
        line, = ax.plot(x, [r["r2"] for r in rows], marker="o", ms=2.5, label=cond)
        ax.plot(x, [r["null_95"] for r in rows], ls=":", color=line.get_color(), lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Decoder block")
    ax.set_ylabel("Out-of-fold $R^2$ (dotted: null 95th pct.)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_probe.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {out / 'fig_probe.png'}")


def _flip_labels(root: Path):
    """Per neutral-correct row: does the item flip under the main assertion?"""
    meta = json.loads((root / "behavior" / "meta.json").read_text())
    items = read_jsonl(root / "behavior" / "items.jsonl")
    nc = [items[i] for i in meta["nc_ids"]]
    return nc, np.array([it["margin"][MAIN_SOURCE] < 0 for it in nc])


def analyze_groups(root: Path, out: Path):
    """Why do some items not follow the assertion? Behavior of flipping vs. non-flipping
    items (flip = margin < 0 under the main assertion): prior strength (neutral margin) vs.
    size of the shift, per condition, and flip rates by quartile of each."""
    nc, flip = _flip_labels(root)
    conds = [c for c in ["assert_plausible", "mention_plausible_1", "assert_irrelevant", "hedge_none"]
             if c in nc[0]["margin"]]
    neu = np.array([it["margin"]["neutral"] for it in nc])
    neu_f = np.array([it["margin_first"]["neutral"] for it in nc])
    lines = [f"# Flipping vs. non-flipping items ({MAIN_SOURCE})", "",
             f"- flip {flip.sum()}, no flip {(~flip).sum()} of {len(nc)} neutral-correct items", ""]
    rows = []
    for name, m in [("flip", flip), ("noflip", ~flip)]:
        row = {"group": name, "n": int(m.sum()),
               "neutral_margin": fmt(boot_mean(neu[m])), "neutral_margin_first": fmt(boot_mean(neu_f[m])),
               "trivia_qa_share_pct": f"{100 * np.mean([nc[i]['dataset'] == 'trivia_qa' for i in np.where(m)[0]]):.1f}"}
        for c in conds:
            d = np.array([nc[i]["margin"][c] - nc[i]["margin"]["neutral"] for i in np.where(m)[0]])
            row[f"shift_{c}"] = fmt(boot_mean(d))
        for key, lab in [("c_plus", "dlp_correct"), ("c_minus", "dlp_incorrect")]:
            v = [np.mean(nc[i]["lp"][MAIN_SOURCE][key]) - np.mean(nc[i]["lp"]["neutral"][key]) for i in np.where(m)[0]]
            row[lab] = fmt(boot_mean(v))
        rows.append(row)
    write_csv(out / "groups.csv", rows)
    for r in rows:
        lines.append(f"- {r['group']} (n={r['n']}): neutral margin {r['neutral_margin']}, "
                     + ", ".join(f"{k} {v}" for k, v in r.items() if k.startswith(("shift_", "dlp_"))))
    shift = np.array([it["margin"][MAIN_SOURCE] for it in nc]) - neu
    lines += ["", f"- correlation(neutral margin, shift) = {np.corrcoef(neu, shift)[0, 1]:+.3f}", ""]
    for name, v in [("neutral margin", neu), (f"shift under {MAIN_SOURCE}", shift)]:
        q = np.quantile(v, [0.25, 0.5, 0.75])
        bins = np.digitize(v, q)
        lines.append(f"- flip rate by quartile of {name}: " + ", ".join(
            f"Q{k + 1} {100 * flip[bins == k].mean():.1f}%" for k in range(4)))
    (out / "groups.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- scoring rules ------------------------------------------------------------------------
def decision_scores(lp):
    """(margin, s(c_plus), s(c_minus)) at the first token where the candidates differ. Up to
    there they share their tokens, so their teacher-forced log-probs are identical; equal
    stored log-probs are taken as the same token. NaN if one candidate is a prefix of the other."""
    a, b = lp["c_plus"], lp["c_minus"]
    for x, y in zip(a, b):
        if x != y:
            return x - y, x, y
    return (np.nan,) * 3


def _scores(lp, rule):
    if rule == "mean":
        a, b = np.mean(lp["c_plus"]), np.mean(lp["c_minus"])
        return a - b, a, b
    if rule == "first":
        a, b = lp["c_plus"][0], lp["c_minus"][0]
        return a - b, a, b
    return decision_scores(lp)


SCORING_CONDS = ["assert_plausible", "assert_irrelevant", "mention_plausible_1", "assert_end_plausible",
                 "assert_end_irrelevant", "assert_alt_plausible", "assert_alt_irrelevant"]


def analyze_scoring(root: Path, out: Path):
    """Does a behavioral conclusion depend on how a multi-token answer is scored? Mean
    log-prob over the answer tokens, first token, and the first token where c_plus and c_minus
    differ (the decision token; candidates sharing a first token, e.g. "The ...", have a
    first-token margin of 0). Each rule uses the neutral-correct items whose neutral margin
    under that rule is positive. Per condition: flip rate, shift, and its split into the
    change of s(c_plus) and of s(c_minus); for the main assertion also the quartiles of the
    neutral margin and the fit m_assert = alpha * m_neutral + beta."""
    nc, _ = _flip_labels(root)
    conds = [c for c in SCORING_CONDS if c in nc[0]["lp"]]
    lines, rows = ["# Scoring rules", ""], []
    for rule in ["mean", "first", "decision"]:
        N = np.array([_scores(it["lp"]["neutral"], rule) for it in nc])
        keep = np.isfinite(N[:, 0]) & (N[:, 0] > 0)
        lines += [f"## {rule} (n = {keep.sum()} of {len(nc)})", ""]
        for c in conds:
            X = np.array([_scores(it["lp"][c], rule) for it in nc])
            ok = keep & np.isfinite(X[:, 0])
            d = X[ok] - N[ok]
            row = {"rule": rule, "condition": c, "n": int(ok.sum()),
                   "flip_rate": fmt(wilson(int((X[ok, 0] < 0).sum()), int(ok.sum())), True),
                   "shift": fmt(boot_mean(d[:, 0])), "d_lp_correct": fmt(boot_mean(d[:, 1])),
                   "d_lp_incorrect": fmt(boot_mean(d[:, 2])),
                   "shifted_toward_c_minus_pct": f"{100 * np.mean(d[:, 0] < 0):.1f}"}
            rows.append(row)
            lines.append(f"- {c}: flip {row['flip_rate']}%, shift {row['shift']}, dlp(c+) {row['d_lp_correct']}, "
                         f"dlp(c-) {row['d_lp_incorrect']}, shifted toward c- {row['shifted_toward_c_minus_pct']}%")
        for c in [c for c in conds if c.endswith("_irrelevant")]:
            pc = c.replace("_irrelevant", "_plausible")
            X, Y = (np.array([_scores(it["lp"][k], rule) for it in nc]) for k in (c, pc))
            ok = keep & np.isfinite(X[:, 0]) & np.isfinite(Y[:, 0])
            lines.append(f"- shift({c}) / shift({pc}) = {np.mean(X[ok, 0] - N[ok, 0]) / np.mean(Y[ok, 0] - N[ok, 0]):.2f}")
        X = np.array([_scores(it["lp"][MAIN_SOURCE], rule) for it in nc])
        ok = keep & np.isfinite(X[:, 0])
        n0, m1 = N[ok], X[ok]
        alpha, beta = np.polyfit(n0[:, 0], m1[:, 0], 1)
        r2 = np.corrcoef(n0[:, 0], m1[:, 0])[0, 1] ** 2
        lines += ["", f"- {MAIN_SOURCE}: m_assert = {alpha:.2f} * m_neutral {beta:+.2f}, R^2 = {r2:.2f}",
                  "- by quartile of the neutral margin: m_neutral | s(c+) | s(c-) | shift | ds(c+) | ds(c-) | flip"]
        q = np.quantile(n0[:, 0], [0.25, 0.5, 0.75])
        bins = np.digitize(n0[:, 0], q)
        for k in range(4):
            m = bins == k
            dd = m1[m] - n0[m]
            lines.append(f"  - Q{k + 1}: {n0[m, 0].mean():.2f} | {n0[m, 1].mean():.2f} | {n0[m, 2].mean():.2f} | "
                         f"{dd[:, 0].mean():.2f} | {dd[:, 1].mean():.2f} | {dd[:, 2].mean():.2f} | "
                         f"{100 * np.mean(m1[m, 0] < 0):.0f}%")
        lines.append("")
    write_csv(out / "scoring.csv", rows)
    (out / "scoring.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- E13: attention heads -------------------------------------------------------------------
def _heads_load(root):
    d = root / "heads"
    out = {}
    for f in sorted(d.glob("*.npz")) if d.exists() else []:
        z = np.load(f)
        stats, rows = z["stats"], z["rows"]
        if GROUP is not None:
            m = GROUP[rows]
            stats, rows = stats[m], rows[m]
        out[f.stem] = {"stats": stats, "rows": rows, "names": list(z["stats_names"])}
    return out


def analyze_heads(root: Path, out: Path, k=10):
    """E13 (v2/run_heads.py). (1) Per block, the direct contribution of all heads' reads of
    the answer / pre / post tokens to the first-token logit difference. (2) Are the heads that
    read an asserted answer the ones that read a mentioned one (per-head means, second half
    of the network)? (3) On the top-k heads (selected on even rows, evaluated on odd rows):
    is the assertion's larger read due to more attention to the answer, or to more read per
    unit of attention? (4) Head knockout: shift eliminated per head set."""
    from v2.run_heads import select_heads
    H = _heads_load(root)
    if not H:
        print("  heads: no results yet")
        return
    names = next(iter(H.values()))["names"]
    L = next(iter(H.values()))["stats"].shape[1]
    j = {n: i for i, n in enumerate(names)}
    lines = ["# Attention heads at the last prompt token (E13)", ""]
    # (1) per block
    rows = []
    for c, h in H.items():
        for b in range(L):
            row = {"condition": c, "block": b}
            for st in ["dla_answer", "dla_pre", "dla_post", "dla_all", "attn_answer"]:
                x = h["stats"][:, b, :, j[st]].sum(-1)
                row[st], row[f"{st}_lo"], row[f"{st}_hi"] = boot_mean(x[np.isfinite(x)])
            rows.append(row)
    write_csv(out / "heads_by_block.csv", rows)
    # (2) head overlap
    late = slice(L // 2, L)
    mean_head = {c: np.nanmean(h["stats"][:, late, :, j["dla_answer"]], axis=0).ravel() for c, h in H.items()}
    ref = "assert_plausible"
    if ref in mean_head:
        lines += [f"## Heads reading the answer (blocks {L // 2}-{L - 1}): correlation with {ref}", ""]
        top = lambda v: set(np.argsort(v)[::-1][:k])
        for c, v in mean_head.items():
            if c in (ref, "neutral"):
                continue
            r = np.corrcoef(mean_head[ref], v)[0, 1]
            lines.append(f"- {c}: r = {r:.3f}, top-{k} overlap {len(top(mean_head[ref]) & top(v))}/{k}")
        lines.append("")
    # (3) amplification on the top-k heads
    if ref in H:
        heads = select_heads(H[ref]["stats"], H[ref]["rows"], k)
        bi, hi = np.array([b for b, _ in heads]), np.array([h for _, h in heads])
        lines += [f"## Top-{k} heads by {ref} (even rows): {heads}", "",
                  "Odd rows; per item summed over these heads: attention to the answer, direct read "
                  "of the answer (logit-difference units), read per unit of attention (pooled).", ""]
        amp = []
        for c, h in H.items():
            if c == "neutral":
                continue
            odd = h["rows"] % 2 == 1
            S = h["stats"][odd][:, bi, hi]  # [n, k, stats]
            att, dla = S[..., j["attn_answer"]].sum(-1), S[..., j["dla_answer"]].sum(-1)
            ok = np.isfinite(dla)
            att, dla = att[ok], dla[ok]
            idx = RNG.integers(0, len(att), (B, len(att)))
            ratio = dla.sum() / att.sum()
            rb = dla[idx].sum(1) / att[idx].sum(1)
            row = {"condition": c, "n": int(ok.sum()), "attn_answer": fmt(boot_mean(att)),
                   "dla_answer": fmt(boot_mean(dla)),
                   "dla_per_attn": f"{ratio:+.3f} [{np.percentile(rb, 2.5):+.3f}, {np.percentile(rb, 97.5):+.3f}]"}
            amp.append(row)
            lines.append(f"- {c}: attention {row['attn_answer']}, read {row['dla_answer']}, per attention {row['dla_per_attn']}")
        write_csv(out / "heads_amplification.csv", amp)
        lines.append("")
    # (4) head knockout
    f = root / "heads" / "knockout.json"
    if f.exists():
        ko = json.loads(f.read_text())
        lines += ["## Head knockout (last token -/-> answer for the set's heads; odd rows)", ""]
        kor = []
        for c, res in ko["conditions"].items():
            r_ = {"rows": res["rows"]}
            for name in ko["heads"]:
                row = {"condition": c, "heads": name}
                t = _sel(r_, [np.array(res[name]["first"]), np.array(res["m_neutral_first"]),
                              np.array(res["biased"]["first"])])
                row["eliminated"], row["lo"], row["hi"], _ = boot([tuple(t)], v_rec)
                kor.append(row)
                lines.append(f"- {c} / {name}: {row['eliminated']:+.2f} [{row['lo']:+.2f}, {row['hi']:+.2f}]")
        write_csv(out / "heads_knockout.csv", kor)
    _plot_heads(rows, mean_head, out / "fig_heads.png")
    (out / "heads.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _plot_heads(rows, mean_head, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    conds = sorted({r["condition"] for r in rows})
    for ax, st in zip(axes[:2], ["dla_answer", "dla_post"]):
        for c in conds:
            rr = [r for r in rows if r["condition"] == c]
            ax.plot([r["block"] for r in rr], [r[st] for r in rr], marker="o", ms=2, label=c)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"sum over heads: {st}", fontsize=9)
        ax.set_xlabel("Decoder block")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    if "assert_plausible" in mean_head and "mention_plausible_1" in mean_head:
        axes[2].scatter(mean_head["mention_plausible_1"], mean_head["assert_plausible"], s=4)
        axes[2].set_xlabel("mention: mean dla_answer per head")
        axes[2].set_ylabel("assertion")
        axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    global GROUP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--only", nargs="+", default=["behavior", "groups", "probe", "main", "rank", "transfer", "illusion",
                             "tracing", "knockout", "answer_direction", "answer_patch", "scoring", "heads"])
    ap.add_argument("--group", choices=["all", "flip", "noflip", "conf_low", "conf_high", "firsttok"], default="all",
                    help="restrict per-item analyses to items that do / do not flip under the main assertion, "
                         "or to the lower / upper half of the neutral margin (prior confidence; unlike the "
                         "flip split, not selected on the outcome), or to the items whose first-token "
                         "neutral margin is positive (firsttok: excludes candidates sharing their first "
                         "token, whose first-token shift is 0). Outputs go to analysis__<group>/")
    args = ap.parse_args()
    root = Path(args.results_root) / args.model
    out = root / "analysis"
    if args.group != "all":
        nc, flip = _flip_labels(root)
        neu = np.array([it["margin"]["neutral"] for it in nc])
        neu_first = np.array([it["margin_first"]["neutral"] for it in nc])
        GROUP = {"flip": flip, "noflip": ~flip, "conf_low": neu <= np.median(neu),
                 "conf_high": neu > np.median(neu), "firsttok": neu_first > 0}[args.group]
        out = root / f"analysis__{args.group}"
        # item-level parts only; behavior-level summaries are not subgroup-specific
        args.only = [p for p in args.only if p not in ("behavior", "groups", "probe", "transfer", "answer_direction",
                                                       "scoring")]
    out.mkdir(exist_ok=True)
    for part in args.only:
        print(f"== {part} ==")
        {"behavior": analyze_behavior, "groups": analyze_groups, "probe": analyze_probe, "main": analyze_main,
         "rank": analyze_rank, "transfer": analyze_transfer, "illusion": analyze_illusion,
         "tracing": analyze_tracing, "knockout": analyze_knockout,
         "answer_direction": analyze_answer_direction, "answer_patch": analyze_answer_patch,
         "scoring": analyze_scoring, "heads": analyze_heads}[part](root, out)


if __name__ == "__main__":
    main()
