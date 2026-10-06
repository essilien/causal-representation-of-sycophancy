"""
Layer-wise linear probing (CPU only; reads the activation cache).

For each block, predicts the per-item margin shift m(cond) - m(neutral) from the
last-prompt-token representation under `cond`, with ridge regression:
  - outer 5-fold CV; alpha chosen INSIDE each training fold by exact leave-one-out (GCV)
    over a log grid, so the reported score is never tuned on its own test fold;
  - metric: out-of-fold R^2 = 1 - SSE/SST (can be negative; v1 reported squared
    correlation, which cannot);
  - null: the same procedure on permuted targets (default 200 permutations).
Ridge is solved in the dual (n << d), so one eigendecomposition per fold serves every alpha
and every permutation.

Usage:
    python -m v2.run_probe --model llama --results-root $SYCO_RESULTS
"""
import argparse
import json
from pathlib import Path

import numpy as np

from v2.config import ABLATION_SOURCES, MAIN_SOURCE, MODELS
from v2.data import read_jsonl

ALPHAS = np.logspace(-1, 6, 15)


def _fold_solver(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    evals, U = np.linalg.eigh(A @ A.T)
    evals = np.clip(evals, 0, None)
    Kte = B @ A.T

    def predict(ytr):  # ytr: [n_tr, P] (P = number of target vectors, e.g. permutations)
        ym = ytr.mean(0)
        Uy = U.T @ (ytr - ym)
        best_err, best = np.full(ytr.shape[1], np.inf), np.zeros((Xte.shape[0], ytr.shape[1]))
        for a in ALPHAS:
            shrink = evals / (evals + a)
            fit = U @ (shrink[:, None] * Uy)
            hdiag = (U ** 2) @ shrink
            loo = ((ytr - ym - fit) / (1 - hdiag)[:, None]) ** 2
            err = loo.mean(0)
            better = err < best_err
            if better.any():
                coef = U @ (Uy[:, better] / (evals + a)[:, None])
                best[:, better] = Kte @ coef + ym[better]
                best_err[better] = err[better]
        return best
    return predict


def oof_r2(X, Y, folds):
    """X [n, d]; Y [n, P]. Returns out-of-fold R^2 for each of the P target columns."""
    pred = np.zeros_like(Y)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(X)), te)
        pred[te] = _fold_solver(X[tr], X[te])(Y[tr])
    return 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="llama")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--conditions", nargs="+",
                    default=[MAIN_SOURCE, *ABLATION_SOURCES])
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    beh = Path(args.results_root) / args.model / "behavior"
    meta = json.loads((beh / "meta.json").read_text())
    items = read_jsonl(beh / "items.jsonl")
    nc = [items[i] for i in meta["nc_ids"]]
    rng = np.random.default_rng(args.seed)
    folds = np.array_split(rng.permutation(len(nc)), 5)
    out = {}
    for cond in args.conditions:
        y = np.array([it["margin"][cond] - it["margin"]["neutral"] for it in nc])
        Y = np.column_stack([y] + [rng.permutation(y) for _ in range(args.n_perm)])
        cache = np.load(beh / "cache" / f"{cond}.npy", mmap_mode="r")
        rows = []
        for b in range(meta["n_layers"]):
            r2 = oof_r2(np.asarray(cache[:, b], dtype=np.float64), Y, folds)
            null = r2[1:]
            rows.append({"block": b, "r2": float(r2[0]), "null_mean": float(null.mean()),
                         "null_sd": float(null.std()), "null_95": float(np.percentile(null, 95)),
                         "p_perm": float((1 + (null >= r2[0]).sum()) / (1 + len(null)))})
            print(f"{cond:22s} block {b:2d}  R2={r2[0]:+.3f}  null95={rows[-1]['null_95']:+.3f}")
        out[cond] = rows
    (beh.parent / "probe.json").write_text(json.dumps(out, indent=1))
    print(f"Saved {beh.parent / 'probe.json'}")


if __name__ == "__main__":
    main()
