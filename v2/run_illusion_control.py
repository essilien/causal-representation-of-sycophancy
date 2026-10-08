"""
Interpretability-illusion control for the trained DAS subspaces (Makelov et al., 2024).

The main run showed DAS recovering far more of the first-token shift than full patching,
including at early blocks where full patching has no effect at all. A subspace can do that
by pushing along a direction that activates a dormant downstream pathway rather than by
transmitting the item's own assertion. This script re-evaluates the trained W (no training)
on the test split with sources that should NOT reproduce the item's shift:

  matched           : the item's own biased representation (= main run, recomputed here)
  other_biased      : another test item's biased representation. A legitimate variable can
                      carry at most the generic assertion pressure here (behaviorally the
                      irrelevant-answer assertion shifts the margin by ~23% of the full shift).
  other_neutral     : another test item's NEUTRAL representation (no assertion at all).
                      A legitimate assertion variable should recover ~0% here; a large shift
                      means W only encodes "push toward c_minus" relative to this base.
Full patching is evaluated with the same sources as a reference.

--direction reverse is the NECESSITY test (Makelov et al.: an illusory subspace is sufficient
when patched in but not necessary). Base = the BIASED prompt, and the subspace (or the full
vector) is set to a non-biased source. Fraction removed = (m_int - m_biased) /
(m_neutral - m_biased):
  matched       : the item's own NEUTRAL representation (does removing the subspace
                  component remove the shift?)
  other_neutral : another item's neutral representation (is what is removed item-specific?)
  other_biased  : another item's biased representation (should remove only the
                  item-specific part)

Outputs <results>/<model>/illusion[_reverse]/seed*/block*__{das,patch}.json with per-item
margins (mean-token and first-token) per source, stored as m_int, m_src (the margin the
intervention aims at) and m_neutral (the base run's margin; for reverse: the biased run);
`python -m v2.analyze --only illusion` summarizes both directions.

Usage:
    python -m v2.run_illusion_control --model llama --results-root $SYCO_RESULTS
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from v2.config import MAIN_SOURCE, MODELS, build_prompt
from v2.das import BaseItem, evaluate
from v2.data import read_jsonl, write_json
from v2.lm import LM
from v2.run_intervention import chunk, parse_blocks, per_token_margins

CONTROLS = ["matched", "other_biased", "other_neutral"]


def derangement(n: int, rng) -> np.ndarray:
    """Permutation with no fixed point (cyclic shift of a random order)."""
    order = rng.permutation(n)
    perm = np.empty(n, dtype=int)
    perm[order] = np.roll(order, 1)
    return perm


def control_sources(src_biased, src_neutral, test, perm, reverse=False):
    """Source arrays aligned with item rows; only test rows are replaced. `matched` is the
    item's own biased (forward) or neutral (reverse) representation."""
    out = {"matched": src_neutral if reverse else src_biased}
    ob = src_biased.copy()
    ob[test] = src_biased[test[perm]]
    on = src_neutral.copy()
    on[test] = src_neutral[test[perm]]
    out["other_biased"], out["other_neutral"] = ob, on
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="llama")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--from-tag", default="main", help="where the trained W and splits are")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--source", default=MAIN_SOURCE)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--blocks", default="all")
    ap.add_argument("--chunk", default=None)
    ap.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    args = ap.parse_args()
    reverse = args.direction == "reverse"

    root = Path(args.results_root) / args.model
    beh = root / "behavior"
    meta = json.loads((beh / "meta.json").read_text())
    items = read_jsonl(beh / "items.jsonl")
    nc = [items[i] for i in meta["nc_ids"]]
    cache_b = np.load(beh / "cache" / f"{args.source}.npy", mmap_mode="r")
    cache_n = np.load(beh / "cache" / "neutral.npy", mmap_mode="r")
    neu, tgt = per_token_margins(nc, "neutral"), per_token_margins(nc, args.source)

    lm = LM.load(MODELS[args.model])
    base_cond = args.source if reverse else "neutral"
    base = [BaseItem(lm.encode_prompt(build_prompt(base_cond, it)),
                     lm.encode_answer(it["c_plus"]), lm.encode_answer(it["c_minus"])) for it in nc]
    blocks = chunk(parse_blocks(args.blocks, lm.n_layers), args.chunk)
    for seed in args.seeds:
        src_dir = root / args.from_tag / f"seed{seed}"
        test = np.array(json.loads((src_dir / "split.json").read_text())["test"])
        perm = derangement(len(test), np.random.default_rng(10_000 + seed))
        out_dir = root / ("illusion_reverse" if reverse else "illusion") / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for b in blocks:
            w_file = src_dir / f"block{b:02d}__das__k{args.k}__src-{args.source}.W.pt"
            if not w_file.exists():
                print(f"seed {seed} block {b}: no trained W, skipping")
                continue
            srcs = control_sources(np.asarray(cache_b[:, b], dtype=np.float32),
                                   np.asarray(cache_n[:, b], dtype=np.float32), test, perm, reverse)
            # margin of the unintervened base run and the margin the intervention aims at
            m_base, m_aim = (tgt, neu) if reverse else (neu, tgt)
            W = torch.load(w_file).float().to(lm.device)
            for method, w in [("das", W), ("patch", None)]:
                f = out_dir / f"block{b:02d}__{method}.json"
                if f.exists():
                    continue
                res = {"model": args.model, "seed": seed, "block": b, "method": method, "k": args.k,
                       "source": args.source, "direction": args.direction,
                       "test_rows": test.tolist(), "perm": perm.tolist(),
                       "m_neutral": m_base["mean"][test].tolist(), "m_neutral_first": m_base["first"][test].tolist(),
                       "m_src": m_aim["mean"][test].tolist(), "m_src_first": m_aim["first"][test].tolist(),
                       "controls": {}}
                for c in CONTROLS:
                    m = evaluate(lm, base, test, b, srcs[c], w)
                    res["controls"][c] = {"m_int": m["mean"].tolist(), "m_int_first": m["first"].tolist()}
                write_json(f, res)
                rec = {c: np.mean(np.array(res["controls"][c]["m_int_first"]) - m_base["first"][test])
                       / np.mean(m_aim["first"][test] - m_base["first"][test]) for c in CONTROLS}
                print(f"seed {seed} block {b:2d} {method:5s} first-token shift {'removed' if reverse else 'recovered'}: "
                      + "  ".join(f"{c}={v:+.2f}" for c, v in rec.items()))


if __name__ == "__main__":
    main()
