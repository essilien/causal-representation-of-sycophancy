"""
Figures for the SRW paper, built from the `analysis/` CSVs of both models (run
`python -m v2.analyze` first). Static PDFs sized for the ACL two-column format.

    python -m v2.paper_figures --llama results_llama/analysis --qwen results_qwen/analysis \
        --out <paper>/latex/figures

fig_read_depth.pdf : where the inserted answer is read (attention knockout from block b on,
                     and restoring the last prompt token in noise tracing), for an asserted
                     wrong answer, a mere mention of it, and an asserted irrelevant answer
fig_das_necessity.pdf : DAS (k=64) and full patching at the last prompt token, inserted into
                     the neutral run (sufficiency) vs. removed from the biased run (necessity)
fig_last_token.pdf : every last-token curve in one place: DAS inserted / removed, full
                     patching and its item-specific part, and the item-specific part of the
                     per-item answer-direction intervention (needs answer_patch.csv)
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-3 of the dataviz reference palette (validated: CVD dE >= 9.2; the
# aqua slot is below 3:1 contrast, so every series also has its own line style + legend).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
CONDITIONS = [("assert_plausible", "assert $c_-$", BLUE, "-"),
              ("mention_plausible_1", "mention $c_-$", ORANGE, "--"),
              ("assert_irrelevant", "assert irrelevant $r$", AQUA, ":")]
MODELS = [("llama", "Llama"), ("qwen", "Qwen")]

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "font.family": "serif", "pdf.fonttype": 42, "axes.linewidth": 0.6})


def rows(path, **match):
    with open(path) as f:
        out = [r for r in csv.DictReader(f) if all(r.get(k) == v for k, v in match.items())]
    return sorted(out, key=lambda r: int(r["block"]))


def series(ax, rs, col, color, ls, label):
    x = [int(r["block"]) for r in rs]
    ax.fill_between(x, [float(r[col + "_lo"]) for r in rs], [float(r[col + "_hi"]) for r in rs],
                    color=color, alpha=0.15, lw=0)
    ax.plot(x, [float(r[col]) for r in rs], color=color, ls=ls, lw=1.3, label=label)


def style(ax, title, ylabel=None):
    ax.set_title(title, color=INK)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axhline(1, color=MUTED, lw=0.5, ls=(0, (1, 2)))
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Decoder block")
    if ylabel:
        ax.set_ylabel(ylabel)


def fig_read_depth(dirs, out):
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.0), sharey=True)
    for j, (m, name) in enumerate(MODELS):
        ax_ko, ax_tr = axes[j], axes[2 + j]
        for cond, label, color, ls in CONDITIONS:
            series(ax_ko, rows(dirs[m] / "knockout.csv", config=cond), "from_span_after_first", color, ls, label)
            tr = rows(dirs[m] / "tracing.csv", config=f"{cond}__noise")
            # noise removes only ~35% of this condition's shift in Llama, so its tracing curve
            # is not interpretable there (stated in the caption)
            if tr and not (m == "llama" and cond == "mention_plausible_1"):
                series(ax_tr, tr, "restore_last_first", color, ls, label)
        style(ax_ko, f"{name}: knockout from $b$ on", "Fraction of first-token shift" if j == 0 else None)
        style(ax_tr, f"{name}: restore last token")
    for ax in axes:
        ax.set_ylim(-0.1, 1.12)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, handlelength=2.4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.9))
    fig.savefig(out / "fig_read_depth.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "fig_read_depth.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def line(ax, x, y, color, ls, label):
    ax.plot(x, y, color=color, ls=ls, lw=1.3, label=label)


def fig_last_token(dirs, out):
    """Item-specific parts = matched source minus another item's neutral source (the part
    of an intervention that depends on the item; full patching also disrupts generically)."""
    fig, axes = plt.subplots(1, 2, figsize=(3.1, 2.25), sharey=True)
    for ax, (m, name) in zip(axes, MODELS):
        fw, rv, ap = dirs[m] / "illusion.csv", dirs[m] / "illusion_reverse.csv", dirs[m] / "answer_patch.csv"
        series(ax, rows(fw, method="das"), "matched_first", BLUE, "-", "DAS insert")
        series(ax, rows(rv, method="das"), "matched_first", BLUE, "--", "DAS remove")
        pr = rows(fw, method="patch")
        x = [int(r["block"]) for r in pr]
        line(ax, x, [float(r["matched_first"]) for r in pr], ORANGE, "-", "full patching")
        line(ax, x, [float(r["matched_first"]) - float(r["other_neutral_first"]) for r in pr],
             ORANGE, ":", "full patching, item-specific")
        a = rows(ap)
        line(ax, [int(r["block"]) for r in a], [float(r["dir2"]) - float(r["dir2_on"]) for r in a],
             AQUA, "-", "own answer direction, item-specific")
        style(ax, name, "Frac. of first-token shift" if ax is axes[0] else None)
        ax.set_ylim(-0.15, 1.08)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, handlelength=2.2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, w_pad=0.5, rect=(0, 0, 1, 0.72))
    fig.savefig(out / "fig_last_token.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "fig_last_token.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_das_necessity(dirs, out):
    fig, axes = plt.subplots(1, 2, figsize=(3.1, 2.0), sharey=True)
    for ax, (m, name) in zip(axes, MODELS):
        fw, rv = dirs[m] / "illusion.csv", dirs[m] / "illusion_reverse.csv"
        series(ax, rows(fw, method="das"), "matched_first", BLUE, "-", "DAS insert")
        series(ax, rows(rv, method="das"), "matched_first", BLUE, "--", "DAS remove")
        series(ax, rows(fw, method="patch"), "matched_first", ORANGE, "-", "patch insert")
        series(ax, rows(rv, method="patch"), "matched_first", ORANGE, "--", "patch remove")
        style(ax, name, "Frac. of first-token shift" if ax is axes[0] else None)
        ax.set_ylim(-0.1, 1.08)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, handlelength=2.2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, w_pad=0.5, rect=(0, 0, 1, 0.8))
    fig.savefig(out / "fig_das_necessity.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "fig_das_necessity.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama", required=True, type=Path)
    ap.add_argument("--qwen", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dirs = {"llama": a.llama, "qwen": a.qwen}
    fig_read_depth(dirs, a.out)
    fig_das_necessity(dirs, a.out)
    if (a.llama / "answer_patch.csv").exists() and (a.qwen / "answer_patch.csv").exists():
        fig_last_token(dirs, a.out)
    print(f"wrote figures to {a.out}")


if __name__ == "__main__":
    main()
