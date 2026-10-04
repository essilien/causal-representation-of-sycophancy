"""
E2 (multi-seed DAS vs. full patching), E3 (rank sweep), E4 (entrainment-ablation sources
and cross-condition transfer), E5 (--model qwen), plus LR calibration on the val split.

For each seed (= data split + DAS init) and each decoder block:
  patch : full-representation patching, evaluated with every --eval-sources condition as
          source (no training).
  das   : for each rank k, trains W on --source (train split, checkpoint chosen on val),
          then evaluates on the test split with EVERY --eval-sources condition as source.
          Evaluating a W trained on one condition with another condition's activations is
          the transfer test: if a subspace learned from sycophancy prompts also carries the
          entrainment-only effect, the two share a causal variable at that block.
Each (seed, block, method, k, source) writes one JSON + W tensor and is skipped if it
already exists, so array tasks can be resubmitted after a timeout.

Usage examples:
  python -m v2.run_intervention --results-root R --tag main --seeds 0 1 2 --blocks all
  python -m v2.run_intervention --results-root R --tag rank --seeds 0 --blocks 5,13,21,29 \
      --methods das --ranks 1 4 16 64 256 1024
  python -m v2.run_intervention --results-root R --tag calibrate --calibrate-lrs 5e-4 2e-3 8e-3 \
      --blocks 13,25 --seeds 0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from v2.config import EVAL_SOURCES, MAIN_SOURCE, MODELS, build_prompt
from v2.das import BaseItem, evaluate, metrics, train_das
from v2.data import read_jsonl, split
from v2.lm import LM


def parse_blocks(spec: str, n_layers: int) -> list[int]:
    if spec == "all":
        return list(range(n_layers))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def chunk(xs: list, spec: str | None) -> list:
    if not spec:
        return xs
    i, n = map(int, spec.split("/"))
    return [x for j, x in enumerate(xs) if j * n // len(xs) == i]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="llama")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--tag", required=True, help="output subfolder: main / rank / ablation / calibrate / smoke")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--blocks", default="all", help='"all", "0-7", or "5,13,21"')
    ap.add_argument("--chunk", default=None, help='"i/n": run only the i-th of n block chunks')
    ap.add_argument("--methods", nargs="+", default=["patch", "das"], choices=["patch", "das"])
    ap.add_argument("--ranks", type=int, nargs="+", default=[64])
    ap.add_argument("--source", default=MAIN_SOURCE, help="condition DAS is trained on")
    ap.add_argument("--eval-sources", nargs="+", default=EVAL_SOURCES)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", default="auto", help='float, or "auto" = <model>/calibrate/best_lr.json (else 2e-3)')
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--calibrate-lrs", type=float, nargs="+", default=None,
                    help="calibration mode: train with each LR, report val MSE only (test untouched)")
    ap.add_argument("--save-w", action="store_true", default=True)
    args = ap.parse_args()

    root = Path(args.results_root) / args.model
    beh = root / "behavior"
    meta = json.loads((beh / "meta.json").read_text())
    items = read_jsonl(beh / "items.jsonl")
    nc = [items[i] for i in meta["nc_ids"]]
    caches = {c: np.load(beh / "cache" / f"{c}.npy", mmap_mode="r")
              for c in set(args.eval_sources) | {args.source}}
    m_neu = np.array([it["margin"]["neutral"] for it in nc])
    tgt = {c: np.array([it["margin"][c] for it in nc]) for c in caches}

    lm = LM.load(MODELS[args.model])
    base = [BaseItem(lm.encode_prompt(build_prompt("neutral", it)),
                     lm.encode_answer(it["c_plus"]), lm.encode_answer(it["c_minus"])) for it in nc]
    blocks = chunk(parse_blocks(args.blocks, lm.n_layers), args.chunk)
    out_dir = root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.model}: {len(nc)} neutral-correct items, blocks {blocks}, seeds {args.seeds}")

    if args.calibrate_lrs:
        return calibrate(args, lm, base, caches, tgt, blocks, out_dir)

    lr = resolve_lr(args.lr, root)
    print(f"LR = {lr}")
    for seed in args.seeds:
        sp = split(len(nc), seed)
        sd = out_dir / f"seed{seed}"
        sd.mkdir(exist_ok=True)
        (sd / "split.json").write_text(json.dumps({k: v.tolist() for k, v in sp.items()}))
        test = sp["test"]

        # Sanity check once per seed: the unintervened batched pipeline must reproduce the
        # behavior-stage neutral margins (differences only from bf16 / batching noise).
        chk = sd / "sanity.json"
        if not chk.exists():
            m0, _ = evaluate(lm, base, test, None)
            diff = float(np.max(np.abs(m0 - m_neu[test])))
            chk.write_text(json.dumps({"max_abs_diff_neutral_margin": diff}))
            print(f"seed {seed}: sanity max |m_neutral diff| = {diff:.4f}")
            if diff > 0.05:
                raise RuntimeError("Intervention pipeline does not reproduce behavior margins.")

        for b in blocks:
            common = {"model": args.model, "seed": seed, "block": b,
                      "test_rows": test.tolist(), "m_neutral": m_neu[test].tolist()}
            if "patch" in args.methods:
                f = sd / f"block{b:02d}__patch.json"
                if not f.exists():
                    res = dict(common, method="patch", eval={})
                    for c in args.eval_sources:
                        m, mf = evaluate(lm, base, test, b, caches[c])
                        res["eval"][c] = {**metrics(m, tgt[c][test], m_neu[test]),
                                          "m_int": m.tolist(), "m_int_first": mf.tolist(),
                                          "m_src": tgt[c][test].tolist()}
                    f.write_text(json.dumps(res))
                    print(f"seed {seed} block {b:2d} patch  "
                          f"IIA={res['eval'][args.eval_sources[0]]['iia']:.3f}")
            if "das" not in args.methods:
                continue
            for k in args.ranks:
                f = sd / f"block{b:02d}__das__k{k}__src-{args.source}.json"
                if f.exists():
                    continue
                t0 = time.time()
                init_seed = 100_000 * seed + 100 * b + k
                W, hist = train_das(lm, base, sp["train"], sp["val"], b, caches[args.source],
                                    tgt[args.source], k, init_seed, steps=args.steps,
                                    bs=args.bs, lr=lr, eval_every=args.eval_every)
                from v2.das import Subspace
                W0 = Subspace(lm.d_model, k, init_seed).to(lm.device)().detach()
                mu, _ = evaluate(lm, base, test, b, caches[args.source], W0)
                res = dict(common, method="das", k=k, source=args.source, lr=lr,
                           steps=args.steps, bs=args.bs, history=hist,
                           untrained=metrics(mu, tgt[args.source][test], m_neu[test]), eval={})
                for c in args.eval_sources:
                    m, mf = evaluate(lm, base, test, b, caches[c], W)
                    res["eval"][c] = {**metrics(m, tgt[c][test], m_neu[test]),
                                      "m_int": m.tolist(), "m_int_first": mf.tolist(),
                                      "m_src": tgt[c][test].tolist()}
                res["seconds"] = time.time() - t0
                if args.save_w:
                    torch.save(W.half().cpu(), f.with_suffix(".W.pt"))
                f.write_text(json.dumps(res))
                print(f"seed {seed} block {b:2d} das k={k:<4d} "
                      f"IIA={res['eval'][args.source]['iia']:.3f} "
                      f"r={res['eval'][args.source]['pearson_r']:.3f} ({res['seconds']:.0f}s)")


def resolve_lr(spec: str, root: Path) -> float:
    if spec != "auto":
        return float(spec)
    f = root / "calibrate" / "best_lr.json"
    if f.exists():
        return float(json.loads(f.read_text())["lr"])
    print("WARNING: no calibrate/best_lr.json found, falling back to lr=2e-3")
    return 2e-3


def calibrate(args, lm, base, caches, tgt, blocks, out_dir):
    """Picks the DAS learning rate by validation MSE (mean over blocks), never using test."""
    sp = split(len(base), args.seeds[0])
    rows = []
    for b in blocks:
        for lr in args.calibrate_lrs:
            _, hist = train_das(lm, base, sp["train"], sp["val"], b, caches[args.source],
                                tgt[args.source], args.ranks[0], 7 + b, steps=args.steps,
                                bs=args.bs, lr=lr, eval_every=args.eval_every)
            rows.append({"block": b, "lr": lr, "best_val_mse": hist["best_val_mse"],
                         "best_step": hist["best_step"], "val_curve": hist["val"]})
            print(f"calibrate block {b} lr {lr:g}: best val MSE {hist['best_val_mse']:.4f} "
                  f"at step {hist['best_step']}")
    (out_dir / f"calibration_blocks{'-'.join(map(str, blocks))}.json").write_text(json.dumps(rows))
    by_lr = {lr: np.mean([r["best_val_mse"] for r in rows if r["lr"] == lr]) for lr in args.calibrate_lrs}
    best = min(by_lr, key=by_lr.get)
    (out_dir / "best_lr.json").write_text(json.dumps({"lr": best, "mean_val_mse": by_lr}))
    print(f"Best LR: {best}  (mean val MSE by LR: {by_lr})")


if __name__ == "__main__":
    main()
