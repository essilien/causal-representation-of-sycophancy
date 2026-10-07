"""
Attention knockout (Geva et al., 2023) on the assert_plausible prompt: at the decoder blocks
in a window, positions after the assertion are not allowed to attend to (part of) it.
Tests directly whether, and at which depth, the sycophantic shift is READ from the
assertion tokens via attention.

  keys    : answer | framing | span                     (see v2/positions.py)
  queries : suffix  = prompt tokens after the assertion (incl. the last prompt token)
            after   = suffix + the teacher-forced answer tokens; only these can affect the
                      margin on answer tokens 2..k ("rest"), e.g. by copying c_minus
  window  : from = blocks b..L-1 (no reading from b on);  win = blocks b..b+w-1
Fraction of the shift eliminated = (m_ko - m_biased) / (m_neutral - m_biased), with
m_biased the unintervened run of the same prompt and m_neutral from the behavior stage.
Prediction if the assertion is read late: `from` stays near 1 up to a late block and then
drops, and `win` peaks at those blocks.

Outputs <results>/<model>/knockout/block*.json with per-item margins (mean / first / rest).
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
from v2.positions import assert_items
from v2.run_intervention import chunk, parse_blocks, per_token_margins

KEYS = ["answer", "framing", "span"]
QUERIES = ["suffix", "after"]
WINDOWS = ["from", "win"]


def blocked(items, idx, key, query, T, device):
    """[2B, 1, T, T] bool, True where (query -> key) attention is removed, for the
    interleaved (c_plus, c_minus) batch."""
    m = torch.zeros(2 * len(idx), 1, T, T, dtype=torch.bool)
    for r, i in enumerate(idx):
        it = items[i]
        k = torch.tensor(it.pos[key])  # explicit list: `framing` has a gap where the answer is
        q0 = it.pos["suffix"][0]
        for j, cand in enumerate((it.c_plus, it.c_minus)):
            q1 = len(it.prefix) if query == "suffix" else len(it.prefix) + len(cand)
            m[2 * r + j, 0, q0:q1, k] = True
    return m.to(device)


@contextmanager
def knockout(lm, layers, block_mask, pad_mask):
    """Removes the blocked entries from the attention mask of every block in `layers`.
    The mask reaching the attention modules is boolean (True = attend; transformers 5) or
    additive float (older versions), or None when the batch has no padding, in which case
    the causal + padding mask is rebuilt here."""
    T = block_mask.shape[-1]
    causal = torch.ones(T, T, dtype=torch.bool, device=block_mask.device).tril()
    default = causal[None, None] & pad_mask[:, None, None, :].bool()

    def pre(mod, args, kwargs):
        if "attention_mask" not in kwargs:
            raise RuntimeError("attention module was not called with an attention_mask kwarg")
        am = kwargs["attention_mask"]
        if am is None:
            am = default
        if am.shape[-2:] != block_mask.shape[-2:]:
            raise RuntimeError(f"attention mask shape {tuple(am.shape)} != {tuple(block_mask.shape)}")
        if am.dtype == torch.bool:
            kwargs["attention_mask"] = am & ~block_mask
        else:  # additive float mask (older transformers)
            kwargs["attention_mask"] = am.masked_fill(block_mask, torch.finfo(am.dtype).min)
        return args, kwargs
    hs = [lm.layers[b].self_attn.register_forward_pre_hook(pre, with_kwargs=True) for b in layers]
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def scores(lm, seqs, layers=(), block_mask=None):
    if not layers:
        return pair_margins(lm.token_logprobs(seqs))
    _, pad = lm._collate(seqs)
    with knockout(lm, layers, block_mask, pad):
        return pair_margins(lm.token_logprobs(seqs))


def run_block(lm, items, b, width, bs):
    L = lm.n_layers
    windows = {"from": list(range(b, L)), "win": list(range(b, min(b + width, L)))}
    res = {"biased": [], **{f"{w}/{k}/{q}": [] for w in WINDOWS for k in KEYS for q in QUERIES}}
    for s in range(0, len(items), bs):
        idx = list(range(s, min(s + bs, len(items))))
        seqs = [Seq(items[i].prefix, c) for i in idx for c in (items[i].c_plus, items[i].c_minus)]
        T = max(len(x.prefix) + len(x.cand) for x in seqs)
        res["biased"].append(scores(lm, seqs))
        for k in KEYS:
            for q in QUERIES:
                bm = blocked(items, idx, k, q, T, lm.device)
                for w in WINDOWS:
                    res[f"{w}/{k}/{q}"].append(scores(lm, seqs, windows[w], bm))
    return {key: {name: torch.cat([p[j] for p in parts]).cpu().numpy().tolist()
                  for j, name in enumerate(["mean", "first", "rest"])}
            for key, parts in res.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="llama")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--blocks", default="all")
    ap.add_argument("--chunk", default=None)
    ap.add_argument("--window", type=int, default=4, help="width of the `win` knockout window")
    ap.add_argument("--limit", type=int, default=None, help="first N neutral-correct items only")
    ap.add_argument("--bs", type=int, default=16, help="items per batch (2 sequences each)")
    args = ap.parse_args()

    root = Path(args.results_root) / args.model
    beh = root / "behavior"
    meta = json.loads((beh / "meta.json").read_text())
    items_all = read_jsonl(beh / "items.jsonl")
    nc = [items_all[i] for i in meta["nc_ids"]][:args.limit]
    neu, bia = per_token_margins(nc, "neutral"), per_token_margins(nc, "assert_plausible")

    lm = LM.load(MODELS[args.model])
    items, rows = assert_items(lm, nc)
    out_dir = root / "knockout"
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = chunk(parse_blocks(args.blocks, lm.n_layers), args.chunk)
    print(f"{args.model}: {len(items)}/{len(nc)} items, window {args.window}, blocks {blocks}")
    sfx = [("mean", ""), ("first", "_first"), ("rest", "_rest")]
    common = {"model": args.model, "rows": rows, "window": args.window,
              **{f"m_neutral{s}": neu[p][rows].tolist() for p, s in sfx},
              **{f"m_biased_behavior{s}": bia[p][rows].tolist() for p, s in sfx}}
    mn = np.array(common["m_neutral_first"])
    for b in blocks:
        f = out_dir / f"block{b:02d}.json"
        if f.exists():
            continue
        res = run_block(lm, items, b, args.window, args.bs)
        write_json(f, {**common, "block": b, "margins": res})
        mb = np.array(res["biased"]["first"])
        msg = [f"{w}/{k}/{q}={np.mean(np.array(res[f'{w}/{k}/{q}']['first']) - mb) / np.mean(mn - mb):+.2f}"
               for w in WINDOWS for k in KEYS for q in QUERIES if q == "suffix"]
        print(f"block {b:2d} (first token, eliminated): " + "  ".join(msg))


if __name__ == "__main__":
    main()
