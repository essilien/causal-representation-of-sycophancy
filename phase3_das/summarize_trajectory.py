"""
Post-hoc, no-compute-needed analysis: for layers with a saved test_trajectory (from
--eval-every runs), compares the single-final-step IIA/r against the average of the last
K checkpoints -- a more robust summary given that step-to-step IIA bounces around from
both training noise (batch_size=1) and evaluation noise (a ~121-150 example test set means
a couple of examples flipping sign moves IIA by ~1pp on its own).

Usage:
    python summarize_trajectory.py --results-paths /path/to/run1/phase3_results.json /path/to/run2/phase3_results.json ...
"""
import argparse
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-paths", type=str, nargs="+", required=True,
                         help="One or more phase3_results.json files to pool layers from.")
    parser.add_argument("--last-k", type=int, default=3,
                         help="Average the last K test-set checkpoints (from --eval-every) per layer.")
    args = parser.parse_args()

    all_results = {}
    for path in args.results_paths:
        all_results.update(json.load(open(path)))

    print(f"{'layer':>6} {'final_iia':>10} {'final_r':>8} | {'lastK_iia':>10} {'lastK_r':>8} {'n_checkpoints':>14}")
    for layer_str, res in sorted(all_results.items(), key=lambda kv: int(kv[0])):
        traj = res.get("test_trajectory", [])
        if not traj:
            print(f"{layer_str:>6} {res['trained_iia']:>10.1%} {res['trained_r']:>8.3f} | "
                  f"{'(no trajectory saved -- single-point estimate only)':>40}")
            continue
        last_k = traj[-args.last_k:]
        iias = [t[1] for t in last_k]
        rs = [t[2] for t in last_k]
        print(f"{layer_str:>6} {res['trained_iia']:>10.1%} {res['trained_r']:>8.3f} | "
              f"{np.mean(iias):>10.1%} {np.mean(rs):>8.3f} {len(last_k):>14d}")


if __name__ == "__main__":
    main()
