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
      --blocks 13,25 --seeds 0 --ranks 64 1 1024      # one best LR per rank
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from v2.config import EVAL_SOURCES, MAIN_SOURCE, MODELS, build_prompt
from v2.das import BaseItem, Subspace, evaluate, metrics, train_das
from v2.data import read_jsonl, rest_margin, split, write_json
from v2.lm import LM

# Unintervened margins are recomputed here with a different batch composition than in
# run_behavior, so bf16 reduction order differs. Small differences are expected noise;
# only abort if they are large enough to signal a real pipeline mismatch.
# The max is NOT used to abort: with bf16 logits a single item can jump by a whole
# rounding step (Qwen: max 0.50 at mean 0.06), so one outlier would fail a healthy run.
SANITY_MAX_MEAN_ABS = 0.2
SANITY_MAX_SIGN_DISAGREE = 0.02


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
    ap.add_argument("--lr", default="auto",
                    help='float, or "auto" = calibrate/best_lr_k<k>.json, else calibrate/best_lr.json, else 2e-3')
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--calibrate-lrs", type=float, nargs="+", default=None,
                    help="calibration mode: train with each LR (and each --ranks), val MSE only (test untouched)")
    ap.add_argument("--no-save-w", dest="save_w", action="store_false", help="don't save trained W tensors")
    args = ap.parse_args()

    root = Path(args.results_root) / args.model
    beh = root / "behavior"
    meta = json.loads((beh / "meta.json").read_text())
    items = read_jsonl(beh / "items.jsonl")
    nc = [items[i] for i in meta["nc_ids"]]
    caches = {c: np.load(beh / "cache" / f"{c}.npy", mmap_mode="r")
              for c in set(args.eval_sources) | {args.source}}
    neu = per_token_margins(nc, "neutral")
    tgt = {c: per_token_margins(nc, c) for c in caches}

    lm = LM.load(MODELS[args.model])
    base = [BaseItem(lm.encode_prompt(build_prompt("neutral", it)),
                     lm.encode_answer(it["c_plus"]), lm.encode_answer(it["c_minus"])) for it in nc]
    blocks = chunk(parse_blocks(args.blocks, lm.n_layers), args.chunk)
    out_dir = root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.model}: {len(nc)} neutral-correct items, blocks {blocks}, seeds {args.seeds}")

    if args.calibrate_lrs:
        return calibrate(args, lm, base, caches, tgt, blocks, out_dir)

    for seed in args.seeds:
        sp = split(len(nc), seed)
        sd = out_dir / f"seed{seed}"
        sd.mkdir(exist_ok=True)
        write_json(sd / "split.json", {k: v.tolist() for k, v in sp.items()})
        test = sp["test"]
        sanity_check(lm, base, test, neu["mean"], sd / "sanity.json", seed)

        for b in blocks:
            # one block's source vectors for all items, read once (not per training step)
            layer_src = {c: np.asarray(caches[c][:, b], dtype=np.float32) for c in caches}
            common = {"model": args.model, "seed": seed, "block": b, "test_rows": test.tolist(),
                      **{f"m_neutral{sfx}": neu[part][test].tolist()
                         for part, sfx in [("mean", ""), ("first", "_first"), ("rest", "_rest")]}}

            def eval_all(W):
                res = {}
                for c in args.eval_sources:
                    m = evaluate(lm, base, test, b, layer_src[c], W)
                    res[c] = {**metrics(m["mean"], tgt[c]["mean"][test], neu["mean"][test]),
                              "m_int": m["mean"].tolist(), "m_int_first": m["first"].tolist(),
                              "m_int_rest": m["rest"].tolist(), "m_src": tgt[c]["mean"][test].tolist(),
                              "m_src_first": tgt[c]["first"][test].tolist(),
                              "m_src_rest": tgt[c]["rest"][test].tolist()}
                return res

            if "patch" in args.methods:
                f = sd / f"block{b:02d}__patch.json"
                if not f.exists():
                    res = dict(common, method="patch", eval=eval_all(None))
                    write_json(f, res)
                    print(f"seed {seed} block {b:2d} patch  IIA={res['eval'][args.eval_sources[0]]['iia']:.3f}")
            if "das" not in args.methods:
                continue
            for k in args.ranks:
                f = sd / f"block{b:02d}__das__k{k}__src-{args.source}.json"
                if f.exists():
                    continue
                lr = resolve_lr(args.lr, root, k)
                t0 = time.time()
                init_seed = 100_000 * seed + 100 * b + k
                W, hist = train_das(lm, base, sp["train"], sp["val"], b, layer_src[args.source],
                                    tgt[args.source]["mean"], k, init_seed, steps=args.steps,
                                    bs=args.bs, lr=lr, eval_every=args.eval_every)
                W0 = Subspace(lm.d_model, k, init_seed).to(lm.device)().detach()
                mu = evaluate(lm, base, test, b, layer_src[args.source], W0)["mean"]
                res = dict(common, method="das", k=k, source=args.source, lr=lr,
                           steps=args.steps, bs=args.bs, history=hist,
                           untrained=metrics(mu, tgt[args.source]["mean"][test], neu["mean"][test]),
                           eval=eval_all(W))
                res["seconds"] = time.time() - t0
                if args.save_w:
                    torch.save(W.half().cpu(), f.with_suffix(".W.pt"))
                write_json(f, res)  # written last: its existence marks the run as complete
                print(f"seed {seed} block {b:2d} das k={k:<4d} lr={lr:g} "
                      f"IIA={res['eval'][args.source]['iia']:.3f} "
                      f"r={res['eval'][args.source]['pearson_r']:.3f} ({res['seconds']:.0f}s)")


def per_token_margins(nc: list[dict], cond: str) -> dict[str, np.ndarray]:
    """Mean-token, first-token and rest-token margins of each item under `cond`."""
    return {"mean": np.array([it["margin"][cond] for it in nc]),
            "first": np.array([it["margin_first"][cond] for it in nc]),
            "rest": np.array([rest_margin(it["lp"][cond]["c_plus"], it["lp"][cond]["c_minus"])
                              for it in nc])}


def sanity_check(lm, base, test, m_neu, path: Path, seed: int):
    """The unintervened pipeline must reproduce the behavior-stage neutral margins up to
    bf16 noise. Logged every time; aborts only on a systematic mismatch (mean difference
    or share of sign disagreements), never on a single outlier."""
    if path.exists():
        return
    m0 = evaluate(lm, base, test, None)["mean"]
    diff = np.abs(m0 - m_neu[test])
    rep = {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
           "sign_disagree": float(np.mean((m0 > 0) != (m_neu[test] > 0)))}
    write_json(path, rep)
    print(f"seed {seed}: sanity {rep}")
    if rep["mean_abs_diff"] > SANITY_MAX_MEAN_ABS or rep["sign_disagree"] > SANITY_MAX_SIGN_DISAGREE:
        raise RuntimeError(f"Intervention pipeline does not reproduce behavior margins: {rep}")
    if rep["max_abs_diff"] > 0.05:
        print("  WARNING: max difference above 0.05 -- expected bf16 noise if the mean is small.")


def resolve_lr(spec: str, root: Path, k: int) -> float:
    if spec != "auto":
        return float(spec)
    for name in [f"best_lr_k{k}.json", "best_lr.json"]:
        f = root / "calibrate" / name
        if f.exists():
            return float(json.loads(f.read_text())["lr"])
    print("WARNING: no calibrate/best_lr*.json found, falling back to lr=2e-3")
    return 2e-3


def calibrate(args, lm, base, caches, tgt, blocks, out_dir):
    """Picks the DAS learning rate per rank by validation MSE (mean over blocks); the test
    split is never touched. Writes best_lr_k<k>.json for every rank and best_lr.json (the
    k=64 choice, or the first rank's) as the default for ranks that were not calibrated."""
    sp = split(len(base), args.seeds[0])
    rows = []
    for k in args.ranks:
        for b in blocks:
            src = np.asarray(caches[args.source][:, b], dtype=np.float32)
            for lr in args.calibrate_lrs:
                _, hist = train_das(lm, base, sp["train"], sp["val"], b, src, tgt[args.source]["mean"],
                                    k, 7 + b, steps=args.steps, bs=args.bs, lr=lr,
                                    eval_every=args.eval_every)
                rows.append({"k": k, "block": b, "lr": lr, "best_val_mse": hist["best_val_mse"],
                             "best_step": hist["best_step"], "val_curve": hist["val"]})
                print(f"calibrate k={k} block {b} lr {lr:g}: best val MSE {hist['best_val_mse']:.4f} "
                      f"at step {hist['best_step']}/{args.steps}")
        by_lr = {lr: float(np.mean([r["best_val_mse"] for r in rows if r["lr"] == lr and r["k"] == k]))
                 for lr in args.calibrate_lrs}
        best = min(by_lr, key=by_lr.get)
        write_json(out_dir / f"best_lr_k{k}.json", {"lr": best, "mean_val_mse": by_lr})
        print(f"k={k}: best LR {best}  (mean val MSE by LR: {by_lr})")
    write_json(out_dir / f"calibration_blocks{'-'.join(map(str, blocks))}.json", rows)
    default_k = 64 if 64 in args.ranks else args.ranks[0]
    write_json(out_dir / "best_lr.json", json.loads((out_dir / f"best_lr_k{default_k}.json").read_text()))


if __name__ == "__main__":
    main()
