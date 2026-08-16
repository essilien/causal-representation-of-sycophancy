"""
Reruns Phase 2's regression probe using RESIDUAL (biased_shift - copy_shift) as the
target instead of raw margin_shift, to check whether the layers identified as encoding
"how much this example's decision will move" are actually belief-specific or are
contaminated by copying/entrainment (per-example correlation between margin_shift and
copy_shift was r=0.73 -- see project history -- so the original regression target is not
clean of the copying confound).

Reuses phase2_layer_prescreen.run_phase2's regression_r2_for_labels and the already-saved
phase2_hidden_states.npz -- no new model calls, pure CPU/sklearn.

IMPORTANT join-key note: copy_effect_control.py's idx ordering (from random.sample on the
neutral-correct pool) does NOT match run_phase2.py's idx ordering (from enumerate(subset)
in phase1_full_results.json's original order) -- these are different permutations of the
same underlying examples. This script joins by question text, not idx, to avoid silently
pairing the wrong examples together.

Usage:
    python rerun_regression_residual.py \
        --phase1-results /kaggle/working/phase1_out/phase1_full_results.json \
        --copy-control-results /kaggle/working/control_out/copy_control_results.jsonl \
        --hidden-states /kaggle/working/phase2_out/phase2_hidden_states.npz \
        --output-dir /kaggle/working/phase2_out
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2_layer_prescreen.run_phase2 import regression_r2_for_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-results", type=str, default="/kaggle/working/phase1_out/phase1_full_results.json")
    parser.add_argument("--copy-control-results", type=str, default="/kaggle/working/control_out/copy_control_results.jsonl")
    parser.add_argument("--hidden-states", type=str, default="/kaggle/working/phase2_out/phase2_hidden_states.npz")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/phase2_out")
    parser.add_argument("--n-perm", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild the EXACT idx->question mapping run_phase2.py used when it saved
    # phase2_hidden_states.npz (same filter, same iteration order -- must match).
    phase1_results = json.load(open(args.phase1_results))
    subset = [r for r in phase1_results if r["neutral_prefers_correct"]]
    question_to_phase2_idx = {r["question"]: i for i, r in enumerate(subset)}

    # copy_effect_control.py's idx ordering is a different permutation -- join by question.
    copy_records = [json.loads(l) for l in open(args.copy_control_results)]
    residual_by_phase2_idx = {}
    for r in copy_records:
        p2_idx = question_to_phase2_idx.get(r["question"])
        if p2_idx is None:
            continue  # shouldn't happen if copy control ran on the full neutral-correct pool
        biased_shift = r["biased_margin"] - r["neutral_margin"]
        copy_shift = r["copy_control_margin"] - r["neutral_margin"]
        residual_by_phase2_idx[p2_idx] = biased_shift - copy_shift

    npz = np.load(args.hidden_states, allow_pickle=True)
    hidden_idxs = sorted(set(int(k.split("__")[0]) for k in npz.files))

    idxs = [i for i in hidden_idxs if i in residual_by_phase2_idx]
    print(f"Examples with both hidden states and copy-control residual: {len(idxs)} / {len(hidden_idxs)}")
    if len(idxs) < len(hidden_idxs) * 0.9:
        print("WARNING: less than 90% overlap -- check copy_effect_control.py ran on the "
              "full neutral-correct pool (not a subsample) before trusting this result.")

    y_residual = np.array([residual_by_phase2_idx[i] for i in idxs])
    num_layers = npz[f"{idxs[0]}__biased"].shape[0]
    print(f"num_layers = {num_layers}, n = {len(idxs)}")

    regression_r2 = np.array([regression_r2_for_labels(npz, idxs, y_residual, L) for L in range(num_layers)])
    print("\nLayer : held-out R^2 predicting RESIDUAL (biased_shift - copy_shift)")
    for L in range(num_layers):
        print(f"{L:5d} : {regression_r2[L]:.3f}")

    rng = np.random.RandomState(0)
    null_r2 = np.zeros((args.n_perm, num_layers))
    import time
    perm_start = time.time()
    for p in range(args.n_perm):
        y_perm = rng.permutation(y_residual)
        for L in range(num_layers):
            null_r2[p, L] = regression_r2_for_labels(npz, idxs, y_perm, L, seed=p)
        elapsed = time.time() - perm_start
        avg_per_perm = elapsed / (p + 1)
        remaining = avg_per_perm * (args.n_perm - p - 1)
        print(f"Permutation {p + 1}/{args.n_perm} done ({elapsed:.0f}s elapsed, "
              f"~{remaining:.0f}s remaining)")
    null_95th = np.percentile(null_r2, 95, axis=0)

    print("\nLayer : real R^2 : null 95th pct : significant?")
    for L in range(num_layers):
        sig = regression_r2[L] > null_95th[L]
        print(f"{L:5d} : {regression_r2[L]:.3f} : {null_95th[L]:.3f} : {'YES' if sig else 'no'}")

    ranked = sorted(range(num_layers), key=lambda L: -regression_r2[L])
    print("\nRanked by R^2 (residual target), highest first, top 10:", ranked[:10])

    # Side-by-side comparison with the ORIGINAL margin_shift-based curve, if available
    prev_summary_path = out_dir / "phase2_summary.npz"
    if prev_summary_path.exists():
        prev = np.load(prev_summary_path)
        prev_r2 = prev["regression_r2"]
        prev_ranked = sorted(range(len(prev_r2)), key=lambda L: -prev_r2[L])
        print("\nFor comparison -- original margin_shift-based ranking, top 10:", prev_ranked[:10])
        corr = np.corrcoef(prev_r2[:num_layers], regression_r2)[0, 1]
        print(f"Correlation between original R^2 curve and residual R^2 curve across layers: {corr:.3f}")
        print("-> high correlation (curves peak in ~same layers): original candidate layers")
        print("   were already mostly belief-specific, not copying-contaminated.")
        print("   Low correlation / shifted peak: re-derive candidate layers from THIS run's")
        print("   ranking instead of the original margin_shift-based one.")
    else:
        print(f"\n(No {prev_summary_path} found -- skipping side-by-side comparison.)")

    np.savez(
        out_dir / "phase2_summary_residual.npz",
        regression_r2=regression_r2, null_95th=null_95th, ranked=np.array(ranked),
    )
    print("\nSaved to", out_dir / "phase2_summary_residual.npz")


if __name__ == "__main__":
    main()
