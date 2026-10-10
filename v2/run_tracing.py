"""
Position-resolved causal tracing (Meng et al., 2022) of the assertion inside the biased
prompt. All v1/v2 interventions sat at the last prompt token; this asks WHERE in the prompt,
and from which block on, the information that drives the sycophantic shift is carried.

Runs are on the prompt of --condition (default assert_plausible), so every position exists
in both runs:
  clean     : the prompt as is (shows the shift)
  corrupted : --corruption noise: Gaussian noise added to the input embeddings of the whole
              inserted span (scale = --noise-mult x std of the span-token embeddings)
              --corruption resample: the inserted answer x replaced by another item's answer
              x' with the same token count, all other tokens unchanged (symmetric token
              replacement; removes only the answer CONTENT, keeps the framing)
For each block b and position group G (see v2/positions.py):
  restore : corrupted run, but the output of block b at G is set to its clean value.
            Fraction restored = (m - m_corr) / (m_clean - m_corr): is the state at (b, G)
            SUFFICIENT to bring the shift back?
  remove  : clean run, but the output of block b at G is set to its corrupted value.
            Fraction removed = (m - m_clean) / (m_corr - m_clean): is the state at (b, G)
            NECESSARY for the shift?
Prediction if the assertion is read late from the context: `last`/`suffix` matter only at
late blocks, while `span` (or `answer`) carries the effect through the middle blocks.

Conditions other than assert_plausible (e.g. mention_plausible_1) test whether the same
pathway carries a mere mention of the answer, i.e. contextual entrainment.

--groups may add pre / post / tmpl / marker (positions.ALL_GROUPS); --variant names such a
run so it does not collide with the default groups of the same condition:
tracing/<condition>__<corruption>-<variant>__s<seed>/. Items lacking a group (e.g. no `post`
token) get NaN for it.

Outputs <results>/<model>/tracing/<condition>__<corruption>__s<seed>/block*.json with
per-item margins (mean / first / rest) and first-token log-probs of c_plus / c_minus;
see positions.run_dir for the first run's location.
"""
import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from v2.config import MODELS
from v2.das import PARTS, pair_scores
from v2.data import read_jsonl, write_json
from v2.lm import LM, Seq
from v2.positions import ALL_GROUPS, GROUPS, assert_items, run_dir
from v2.run_intervention import chunk, parse_blocks, per_token_margins
from v2.run_knockout import first_token_lps

DIRECTIONS = ["restore", "remove"]
LEGACY_TAG = "assert_plausible__noise__s0"


@contextmanager
def embed_noise(lm, rows, cols, noise):
    """Adds noise[j] to the input embedding at (rows[j], cols[j])."""
    if noise is None:
        yield
        return

    def hook(mod, inp, out):
        out = out.clone()
        out[rows, cols] += noise.to(out.dtype)
        return out
    h = lm.model.get_input_embeddings().register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def set_fn(rows, cols, vals):
    """Block-output rewrite: h[rows[j], cols[j]] <- vals[j]."""
    def fn(h):
        h = h.clone()
        h[rows, cols] = vals.to(h.dtype)
        return h
    return fn


@torch.no_grad()
def record(lm, prefixes, b, rows, cols, noise_args):
    """Output of block b at (rows, cols) for prompt-only runs -> [n, d]."""
    out = {}

    def grab(h):
        out["v"] = h[rows, cols].detach().clone()
        return h
    with embed_noise(lm, *noise_args):
        lm.token_logprobs([Seq(p, []) for p in prefixes], b, grab)
    return out["v"]


def noise_for(items, idx, d, scale, seed, device):
    """Per-item fixed noise over its span positions (same draw for every block)."""
    rows, cols, chunks = [], [], []
    for r, i in enumerate(idx):
        span = items[i].pos["span"]
        g = torch.Generator().manual_seed(seed * 1_000_003 + int(i))
        chunks.append(torch.randn(len(span), d, generator=g) * scale)
        rows += [r] * len(span)
        cols += span
    return (torch.tensor(rows, device=device), torch.tensor(cols, device=device),
            torch.cat(chunks).to(device))


def pair_index(items, idx, group, where, device):
    """Flat (row, col) indices of `group` for the interleaved (c_plus, c_minus) batch, and
    for each entry its row in the prompt-level recording (`where`: (item row, pos) -> row)."""
    rows, cols, sel = [], [], []
    for r, i in enumerate(idx):
        for p in items[i].pos[group]:
            rows += [2 * r, 2 * r + 1]
            cols += [p, p]
            sel += [where[(r, p)]] * 2
    return (torch.tensor(rows, device=device), torch.tensor(cols, device=device),
            torch.tensor(sel, device=device))


def run_block(lm, items, b, groups, scale, seed, bs, corruption="noise"):
    """Margins for every item: clean, corrupted, and each (direction, group)."""
    dev = lm.device
    res = {"clean": [], "corr": [], **{f"{d}/{g}": [] for d in DIRECTIONS for g in groups}}
    for s in range(0, len(items), bs):
        idx = list(range(s, min(s + bs, len(items))))
        prefixes = [items[i].prefix for i in idx]
        seqs = [Seq(items[i].prefix, c) for i in idx for c in (items[i].c_plus, items[i].c_minus)]
        if corruption == "noise":
            nr, nc_, nz = noise_for(items, idx, lm.d_model, scale, seed, dev)
            # the same noise for the paired (c_plus, c_minus) sequences
            pr, pc = 2 * nr, nc_
            pair_noise = (torch.cat([pr, pr + 1]), torch.cat([pc, pc]), torch.cat([nz, nz]))
            prompt_noise = (nr, nc_, nz)
            corr_prefixes, corr_seqs = prefixes, seqs
        else:  # resample: a different, token-aligned prompt; no noise
            pair_noise = prompt_noise = (None, None, None)
            corr_prefixes = [items[i].corr_prefix for i in idx]
            corr_seqs = [Seq(items[i].corr_prefix, c) for i in idx for c in (items[i].c_plus, items[i].c_minus)]
        # recorded prompt states of block b at every position of every group
        rec_rows, rec_cols, where = [], [], {}
        for r, i in enumerate(idx):
            for p in sorted({p for g in groups for p in items[i].pos[g]}):
                where[(r, p)] = len(rec_rows)
                rec_rows.append(r)
                rec_cols.append(p)
        rr, rc = torch.tensor(rec_rows, device=dev), torch.tensor(rec_cols, device=dev)
        states = {"clean": record(lm, prefixes, b, rr, rc, (None, None, None)),
                  "corr": record(lm, corr_prefixes, b, rr, rc, prompt_noise)}
        with torch.no_grad():
            res["clean"].append(pair_scores(lm.token_logprobs(seqs)))
            with embed_noise(lm, *pair_noise):
                res["corr"].append(pair_scores(lm.token_logprobs(corr_seqs)))
            for g in groups:
                rows, cols, sel = pair_index(items, idx, g, where, dev)
                with embed_noise(lm, *pair_noise):  # restore: corrupted run, clean state at (b, G)
                    res[f"restore/{g}"].append(pair_scores(
                        lm.token_logprobs(corr_seqs, b, set_fn(rows, cols, states["clean"][sel]))))
                res[f"remove/{g}"].append(pair_scores(  # remove: clean run, corrupted state
                    lm.token_logprobs(seqs, b, set_fn(rows, cols, states["corr"][sel]))))
    out = {}
    for key, parts in res.items():
        g = key.split("/")[-1]
        missing = [i for i, it in enumerate(items) if "/" in key and not it.pos[g]]
        out[key] = {}
        for j, name in enumerate(PARTS):
            v = torch.cat([p[j] for p in parts]).cpu().numpy()
            v[missing] = np.nan
            out[key][name] = v.tolist()
    return out


def noise_scale(lm, items, mult):
    emb = lm.model.get_input_embeddings().weight
    ids = torch.tensor(sorted({it.prefix[p] for it in items for p in it.pos["span"]}), device=emb.device)
    return float(mult * emb[ids].float().std())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="llama")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--blocks", default="all")
    ap.add_argument("--chunk", default=None)
    ap.add_argument("--groups", nargs="+", default=GROUPS, choices=ALL_GROUPS)
    ap.add_argument("--variant", default=None, help="run name suffix for non-default --groups")
    ap.add_argument("--condition", default="assert_plausible", help="prompt condition with an inserted answer")
    ap.add_argument("--corruption", choices=["noise", "resample"], default="noise")
    ap.add_argument("--noise-mult", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=None, help="first N neutral-correct items only")
    ap.add_argument("--bs", type=int, default=16, help="items per batch (2 sequences each)")
    ap.add_argument("--seed", type=int, default=0, help="noise draw / resample choice")
    args = ap.parse_args()

    root = Path(args.results_root) / args.model
    beh = root / "behavior"
    meta = json.loads((beh / "meta.json").read_text())
    items_all = read_jsonl(beh / "items.jsonl")
    nc = [items_all[i] for i in meta["nc_ids"]][:args.limit]
    neu, bia = per_token_margins(nc, "neutral"), per_token_margins(nc, args.condition)

    lm = LM.load(MODELS[args.model])
    resample = args.corruption == "resample"
    items, rows = assert_items(lm, nc, args.condition, resample_seed=args.seed if resample else None)
    scale = noise_scale(lm, items, args.noise_mult)
    # content-free conditions have no answer tokens: drop groups that are empty everywhere
    args.groups = [g for g in args.groups if any(it.pos[g] for it in items)]
    tag = f"{args.condition}__{args.corruption}{'-' + args.variant if args.variant else ''}__s{args.seed}"
    out_dir = run_dir(root, "tracing", tag, LEGACY_TAG)
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = chunk(parse_blocks(args.blocks, lm.n_layers), args.chunk)
    print(f"{args.model} {tag}: {len(items)}/{len(nc)} items, noise scale {scale:.4f}, blocks {blocks}")
    common = {"model": args.model, "condition": args.condition, "corruption": args.corruption,
              "rows": rows, "noise_scale": scale, "noise_mult": args.noise_mult,
              "seed": args.seed, "groups": args.groups,
              **{f"m_neutral{s}": neu[p][rows].tolist() for p, s in [("mean", ""), ("first", "_first"), ("rest", "_rest")]},
              **{f"m_biased_behavior{s}": bia[p][rows].tolist() for p, s in [("mean", ""), ("first", "_first"), ("rest", "_rest")]},
              **first_token_lps(nc, rows, "neutral", "m_neutral")}
    for b in blocks:
        f = out_dir / f"block{b:02d}.json"
        if f.exists():
            continue
        res = run_block(lm, items, b, args.groups, scale, args.seed, args.bs, args.corruption)
        write_json(f, {**common, "block": b, "margins": res})
        cl, co = np.array(res["clean"]["first"]), np.array(res["corr"]["first"])
        msg = [f"corr removes {np.mean(co - cl) / np.mean(np.array(common['m_neutral_first']) - cl):+.2f} of shift"]
        for d in DIRECTIONS:
            for g in args.groups:
                m = np.array(res[f"{d}/{g}"]["first"])
                ok = np.isfinite(m)
                frac = (np.mean((m - co)[ok]) / np.mean((cl - co)[ok]) if d == "restore"
                        else np.mean((m - cl)[ok]) / np.mean((co - cl)[ok]))
                msg.append(f"{d[:3]}/{g}={frac:+.2f}")
        print(f"block {b:2d} (first token): " + "  ".join(msg))


if __name__ == "__main__":
    main()
