"""
Free check (no GPU/model needed) -- reads the already-saved per-step losses from
phase3_results.json to see whether a given layer's training loss had plateaued by the end
of its run, or was still trending down. Answers "would more steps likely help" without
spending any more compute to find out.

Usage:
    python check_loss_plateau.py --layer 15
"""
import argparse
import json

import numpy as np
from scipy.stats import linregress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=str, default="/kaggle/working/phase3_out/phase3_results.json")
    parser.add_argument("--layer", type=int, default=15)
    args = parser.parse_args()

    results = json.load(open(args.results_path))
    losses = results[str(args.layer)]["losses"]
    n = len(losses)
    print(f"Layer {args.layer}: {n} recorded training steps")

    # Compare early vs late portions using a rolling average (raw per-step loss is noisy,
    # same reasoning as the calibration diagnostics earlier in this project)
    window = max(20, n // 10)
    early = np.mean(losses[:window])
    late = np.mean(losses[-window:])
    print(f"Rolling average, first {window} steps: {early:.3f}")
    print(f"Rolling average, last {window} steps:  {late:.3f}")

    # Trend test on the SECOND HALF only -- this is the part that matters for "would more
    # steps help": if the first half did the improving and the second half is flat, more
    # steps of the same kind of training won't add much.
    second_half = losses[n // 2:]
    slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(second_half)), second_half)
    print(f"\nSecond-half trend (steps {n//2}-{n}): slope={slope:.5f}/step, p={p_value:.4f}")
    if p_value < 0.05 and slope < 0:
        print("-> Still decreasing significantly in the second half of training.")
        print("   More steps are plausibly worth trying.")
    else:
        print("-> No significant downward trend in the second half -- loss appears to have")
        print("   plateaued well before the run ended. More steps at the same layer/rank/lr")
        print("   are unlikely to meaningfully improve IIA/r on their own; the ceiling is")
        print("   more likely something else (subspace rank, batch size, or a structural")
        print("   limit on how much a single-layer linear subspace can capture).")


if __name__ == "__main__":
    main()
