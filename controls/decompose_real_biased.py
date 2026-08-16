"""
Computes the per-candidate log-probability decomposition for the REAL biased condition
(correct_answer's own change, incorrect_answer's own change), to sit alongside the
already-computed decomposition for the controlled/irrelevant condition. No new model
calls -- phase1_full_results.json already stores lp_correct/lp_incorrect for both the
neutral and biased conditions per example.

Usage:
    python decompose_real_biased.py --phase1-results /path/to/phase1_full_results.json
"""
import argparse
import json

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase1-results", type=str, default="/kaggle/working/phase1_out/phase1_full_results.json")
args = parser.parse_args()

results = json.load(open(args.phase1_results))
subset = [r for r in results if r["neutral_prefers_correct"]]

correct_change = np.array([r["biased"]["lp_correct"] - r["neutral"]["lp_correct"] for r in subset])
incorrect_change = np.array([r["biased"]["lp_incorrect"] - r["neutral"]["lp_incorrect"] for r in subset])

print(f"n = {len(subset)}")
print(f"correct_answer's own logprob change (biased - neutral):   mean={correct_change.mean():+.4f}")
print(f"incorrect_answer's own logprob change (biased - neutral): mean={incorrect_change.mean():+.4f}")
print(f"Sum (should match the mean margin shift already reported): {(correct_change - incorrect_change).mean():+.4f}")
