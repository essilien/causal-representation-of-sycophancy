"""
Item-specific answer-direction interventions: why does a subspace shared across items fall
behind full patching at the last blocks (Qwen blocks 26-27)? Hypothesis: there, the
item-specific part of the last-token change is low-dimensional but points at each item's
OWN answer tokens, so no single shared W can carry it for all items.

At the last prompt token of block b, the interchange h <- h + W_i W_i^T (h_src - h) is applied
with a per-item, untrained W_i built from the unembedding U and the final RMSNorm gain g:
  dir1        : W_i = normalize(g * (U[c_minus_1] - U[c_plus_1]))   (first answer tokens)
  dir2        : W_i = orthonormal basis of {g * U[c_minus_1], g * U[c_plus_1]}
  dir2_other  : dir2 of another test item (does any low-dimensional answer subspace work?)
  dir2_on     : dir2, but the source is another item's NEUTRAL representation (does the
                intervention just disrupt the prediction, like full patching partly does?)
Items whose two candidates share the first token get W_i = 0 (their first-token margin is 0).
Same test splits as the main run, so results line up with DAS and full patching there.

Outputs <results>/<model>/answer_patch/seed*/block*.json (per-item margins).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from v2.config import MAIN_SOURCE, MODELS, build_prompt
from v2.das import BaseItem, pair_margins
from v2.data import read_jsonl, write_json
from v2.lm import LM, Seq
from v2.run_illusion_control import derangement
from v2.run_intervention import chunk, parse_blocks, per_token_margins

VARIANTS = ["dir1", "dir2", "dir2_other", "dir2_on"]


def item_bases(lm, nc):
    """Per-item answer subspaces [n, d, 2] (dir2) and [n, d, 1] (dir1), float32 on CPU."""
    first_m = [lm.encode_answer(it["c_minus"])[0] for it in nc]
    first_p = [lm.encode_answer(it["c_plus"])[0] for it in nc]
    U = lm.model.get_output_embeddings().weight.detach()  # index first: the full matrix is ~2 GB in fp32
    g = lm.model.model.norm.weight.detach().float().cpu()
    um = U[torch.tensor(first_m, device=U.device)].float().cpu() * g
    up = U[torch.tensor(first_p, device=U.device)].float().cpu() * g
    a = um - up
    n = a.norm(dim=-1, keepdim=True)
    same = torch.tensor([m == p for m, p in zip(first_m, first_p)])
    w1 = torch.where(same[:, None], torch.zeros_like(a), a / n.clamp_min(1e-8))[:, :, None]
    q, r = torch.linalg.qr(torch.stack([um, up], dim=-1))  # [n, d, 2]
    keep = (r.diagonal(dim1=-2, dim2=-1).abs() > 1e-6 * r[:, :1, :1].abs().reshape(-1, 1)).float()
    w2 = q * keep[:, None, :]
    w2[same] = 0
    return w1, w2, same


def item_patch_fn(pos, src, W):
    """h <- h + W_i W_i^T (src_i - h) at one position per sequence; W: [B, d, k]."""
    def fn(h):
        rows = torch.arange(h.shape[0], device=h.device)
        base = h[rows, pos].float()
        delta = src.to(h.device).float() - base
        coef = torch.einsum("bd,bdk->bk", delta, W)
        new = base + torch.einsum("bk,bdk->bd", coef, W)
        h = h.clone()
        h[rows, pos] = new.to(h.dtype)
        return h
    return fn


@torch.no_grad()
def evaluate_items(lm, items, idx, layer, src, W, bs=32):
    """Margins (mean, first) with per-item W; src [n_items, d] and W [n_items, d, k] are
    indexed by item row."""
    out = {"mean": [], "first": []}
    for s in range(0, len(idx), bs):
        ch = list(idx[s:s + bs])
        seqs, pos = [], []
        for i in ch:
            it = items[i]
            seqs += [Seq(it.prefix, it.c_plus), Seq(it.prefix, it.c_minus)]
            pos += [len(it.prefix) - 1] * 2
        dev = lm.device
        fn = item_patch_fn(torch.tensor(pos, device=dev),
                           torch.from_numpy(np.ascontiguousarray(src[ch])).repeat_interleave(2, 0).to(dev),
                           W[ch].repeat_interleave(2, 0).to(dev))
        m, f, _ = pair_margins(lm.token_logprobs(seqs, layer, fn))
        out["mean"].append(m.cpu())
        out["first"].append(f.cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="llama")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--blocks", default="all")
    ap.add_argument("--chunk", default=None)
    args = ap.parse_args()

    root = Path(args.results_root) / args.model
    beh = root / "behavior"
    meta = json.loads((beh / "meta.json").read_text())
    items = read_jsonl(beh / "items.jsonl")
    nc = [items[i] for i in meta["nc_ids"]]
    cache_b = np.load(beh / "cache" / f"{MAIN_SOURCE}.npy", mmap_mode="r")
    cache_n = np.load(beh / "cache" / "neutral.npy", mmap_mode="r")
    neu, tgt = per_token_margins(nc, "neutral"), per_token_margins(nc, MAIN_SOURCE)

    lm = LM.load(MODELS[args.model])
    base = [BaseItem(lm.encode_prompt(build_prompt("neutral", it)),
                     lm.encode_answer(it["c_plus"]), lm.encode_answer(it["c_minus"])) for it in nc]
    w1, w2, same = item_bases(lm, nc)
    print(f"{args.model}: {int(same.sum())}/{len(nc)} items share the first answer token (W = 0 for them)")
    blocks = chunk(parse_blocks(args.blocks, lm.n_layers), args.chunk)
    for seed in args.seeds:
        sp = json.loads((root / "main" / f"seed{seed}" / "split.json").read_text())
        test = np.array(sp["test"])
        perm = derangement(len(test), np.random.default_rng(10_000 + seed))  # as in the illusion control
        other = np.arange(len(nc))
        other[test] = test[perm]
        out_dir = root / "answer_patch" / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for b in blocks:
            f = out_dir / f"block{b:02d}.json"
            if f.exists():
                continue
            sb = np.asarray(cache_b[:, b], dtype=np.float32)
            sn = np.asarray(cache_n[:, b], dtype=np.float32)
            src_on = sn.copy()
            src_on[test] = sn[test[perm]]
            runs = {"dir1": (sb, w1), "dir2": (sb, w2), "dir2_other": (sb, w2[torch.from_numpy(other)]),
                    "dir2_on": (src_on, w2)}
            res = {"model": args.model, "seed": seed, "block": b, "test_rows": test.tolist(),
                   "same_first_token": same[test].tolist(),
                   "m_neutral": neu["mean"][test].tolist(), "m_neutral_first": neu["first"][test].tolist(),
                   "m_src": tgt["mean"][test].tolist(), "m_src_first": tgt["first"][test].tolist(),
                   "variants": {}}
            for name, (src, W) in runs.items():
                m = evaluate_items(lm, base, test, b, src, W)
                res["variants"][name] = {"m_int": m["mean"].tolist(), "m_int_first": m["first"].tolist()}
            write_json(f, res)
            rec = {v: np.mean(np.array(res["variants"][v]["m_int_first"]) - neu["first"][test])
                   / np.mean(tgt["first"][test] - neu["first"][test]) for v in VARIANTS}
            print(f"seed {seed} block {b:2d} first-token recovered: " + "  ".join(f"{k}={v:+.2f}" for k, v in rec.items()))


if __name__ == "__main__":
    main()
