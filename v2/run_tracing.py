"""
Position-resolved causal tracing (Meng et al., 2022) of the assertion inside the biased
prompt. All v1/v2 interventions sat at the last prompt token; this asks WHERE in the prompt,
and from which block on, the information that drives the sycophantic shift is carried.

Runs are on the assert_plausible prompt, so every position exists in both runs:
  clean     : the prompt as is (shows the sycophantic shift)
  corrupted : Gaussian noise added to the input embeddings of the whole assertion span
              (scale = --noise-mult x std of the span-token embeddings), which removes it
For each block b and position group G (see v2/positions.py):
  restore : corrupted run, but the output of block b at G is set to its clean value.
            Fraction restored = (m - m_corr) / (m_clean - m_corr): is the state at (b, G)
            SUFFICIENT to bring the shift back?
  remove  : clean run, but the output of block b at G is set to its corrupted value.
            Fraction removed = (m - m_clean) / (m_corr - m_clean): is the state at (b, G)
            NECESSARY for the shift?
Prediction if the assertion is read late from the context: `last`/`suffix` matter only at
late blocks, while `span` (or `answer`) carries the effect through the middle blocks.

Outputs <results>/<model>/tracing/block*.json with per-item margins (mean / first / rest).
"""
import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from v2.config import MODELS
from v2.das import pair_margins
from v2.data import read_jsonl, write_json
from v2.lm import LM, Seq
from v2.positions import GROUPS, assert_items
from v2.run_intervention import chunk, parse_blocks, per_token_margins

DIRECTIONS = ["restore", "remove"]


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


def run_block(lm, items, b, groups, scale, seed, bs):
    """Margins for every item: clean, corrupted, and each (direction, group)."""
    dev = lm.device
    res = {"clean": [], "corr": [], **{f"{d}/{g}": [] for d in DIRECTIONS for g in groups}}
    for s in range(0, len(items), bs):
        idx = list(range(s, min(s + bs, len(items))))
        prefixes = [items[i].prefix for i in idx]
        seqs = [Seq(items[i].prefix, c) for i in idx for c in (items[i].c_plus, items[i].c_minus)]
        nr, nc_, nz = noise_for(items, idx, lm.d_model, scale, seed, dev)
        # the same noise for the paired (c_plus, c_minus) sequences
        pr, pc = 2 * nr, nc_
        pair_noise = (torch.cat([pr, pr + 1]), torch.cat([pc, pc]), torch.cat([nz, nz]))
        prompt_noise = (nr, nc_, nz)
        # recorded prompt states of block b at every position of every group
        rec_rows, rec_cols, where = [], [], {}
        for r, i in enumerate(idx):
            for p in sorted({p for g in groups for p in items[i].pos[g]}):
                where[(r, p)] = len(rec_rows)
                rec_rows.append(r)
                rec_cols.append(p)
        rr, rc = torch.tensor(rec_rows, device=dev), torch.tensor(rec_cols, device=dev)
        states = {"clean": record(lm, prefixes, b, rr, rc, (None, None, None)),
                  "corr": record(lm, prefixes, b, rr, rc, prompt_noise)}
        with torch.no_grad():
            res["clean"].append(pair_margins(lm.token_logprobs(seqs)))
            with embed_noise(lm, *pair_noise):
                res["corr"].append(pair_margins(lm.token_logprobs(seqs)))
            for g in groups:
                rows, cols, sel = pair_index(items, idx, g, where, dev)
                with embed_noise(lm, *pair_noise):  # restore: corrupted run, clean state at (b, G)
                    res[f"restore/{g}"].append(pair_margins(
                        lm.token_logprobs(seqs, b, set_fn(rows, cols, states["clean"][sel]))))
                res[f"remove/{g}"].append(pair_margins(  # remove: clean run, corrupted state
                    lm.token_logprobs(seqs, b, set_fn(rows, cols, states["corr"][sel]))))
    out = {}
    for key, parts in res.items():
        out[key] = {name: torch.cat([p[j] for p in parts]).cpu().numpy().tolist()
                    for j, name in enumerate(["mean", "first", "rest"])}
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
    ap.add_argument("--groups", nargs="+", default=GROUPS, choices=GROUPS)
    ap.add_argument("--noise-mult", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=None, help="first N neutral-correct items only")
    ap.add_argument("--bs", type=int, default=16, help="items per batch (2 sequences each)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.results_root) / args.model
    beh = root / "behavior"
    meta = json.loads((beh / "meta.json").read_text())
    items_all = read_jsonl(beh / "items.jsonl")
    nc = [items_all[i] for i in meta["nc_ids"]][:args.limit]
    neu, bia = per_token_margins(nc, "neutral"), per_token_margins(nc, "assert_plausible")

    lm = LM.load(MODELS[args.model])
    items, rows = assert_items(lm, nc)
    scale = noise_scale(lm, items, args.noise_mult)
    out_dir = root / "tracing"
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = chunk(parse_blocks(args.blocks, lm.n_layers), args.chunk)
    print(f"{args.model}: {len(items)}/{len(nc)} items, noise scale {scale:.4f}, blocks {blocks}")
    common = {"model": args.model, "rows": rows, "noise_scale": scale, "noise_mult": args.noise_mult,
              "seed": args.seed, "groups": args.groups,
              **{f"m_neutral{s}": neu[p][rows].tolist() for p, s in [("mean", ""), ("first", "_first"), ("rest", "_rest")]},
              **{f"m_biased_behavior{s}": bia[p][rows].tolist() for p, s in [("mean", ""), ("first", "_first"), ("rest", "_rest")]}}
    for b in blocks:
        f = out_dir / f"block{b:02d}.json"
        if f.exists():
            continue
        res = run_block(lm, items, b, args.groups, scale, args.seed, args.bs)
        write_json(f, {**common, "block": b, "margins": res})
        cl, co = np.array(res["clean"]["first"]), np.array(res["corr"]["first"])
        msg = [f"corr removes {np.mean(co - cl) / np.mean(np.array(common['m_neutral_first']) - cl):+.2f} of shift"]
        for d in DIRECTIONS:
            for g in args.groups:
                m = np.array(res[f"{d}/{g}"]["first"])
                frac = np.mean(m - co) / np.mean(cl - co) if d == "restore" else np.mean(m - cl) / np.mean(co - cl)
                msg.append(f"{d[:3]}/{g}={frac:+.2f}")
        print(f"block {b:2d} (first token): " + "  ".join(msg))


if __name__ == "__main__":
    main()
