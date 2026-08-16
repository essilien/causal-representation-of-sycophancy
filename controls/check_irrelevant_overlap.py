"""
Per-example overlap between real-biased-condition flips and irrelevant-answer-condition
flips, computed on the ORIGINAL results file (same pool used for every other number in
Section 4.3) -- not the larger replication pool. No GPU/model needed; reads the already-
saved irrelevant_control_results.jsonl directly.

Fixes two issues in an earlier version of this script:
  - field names now match what irrelevant_answer_control.py actually saves
    (real_biased_margin / real_neutral_margin, not biased_margin / neutral_margin)
  - flip conditions now require the correct starting state (neutral margin > 0) before
    counting a negative margin as a "flip", matching the definition used everywhere else
    in this project, rather than a crude shift-sign check

Usage:
    python check_irrelevant_overlap.py --results-path /path/to/irrelevant_control_results.jsonl
"""
import argparse
import json

import numpy as np
from scipy.stats import chi2_contingency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=str,
                         default="/kaggle/working/irrelevant_control_out/irrelevant_control_results.jsonl")
    args = parser.parse_args()

    records = [json.loads(l) for l in open(args.results_path)]
    print(f"n = {len(records)}")

    real_flip = np.array([r["real_neutral_margin"] > 0 and r["real_biased_margin"] < 0 for r in records])
    irrel_flip = np.array([
        (r["lp_correct_neutral"] > r["lp_irrel_neutral"]) and
        (r["lp_correct_biased_irrel"] < r["lp_irrel_biased_irrel"])
        for r in records
    ])

    both = int((real_flip & irrel_flip).sum())
    only_real = int((real_flip & ~irrel_flip).sum())
    only_irrel = int((~real_flip & irrel_flip).sum())
    neither = int((~real_flip & ~irrel_flip).sum())
    union = both + only_real + only_irrel
    jaccard = both / union if union > 0 else float("nan")

    print(f"Flipped under real-biased only:       {only_real}")
    print(f"Flipped under irrelevant-answer only: {only_irrel}")
    print(f"Flipped under BOTH:                   {both}")
    print(f"Flipped under NEITHER:                {neither}")
    print(f"Jaccard overlap (both / union):       {jaccard:.1%}")

    p_real, p_irrel = real_flip.mean(), irrel_flip.mean()
    expected_both = p_real * p_irrel * len(records)
    print(f"Expected 'both' if independent:       {expected_both:.1f}  (observed: {both})")

    table = np.array([[both, only_real], [only_irrel, neither]])
    chi2, p_val, dof, _ = chi2_contingency(table, correction=False)
    phi = np.sqrt(chi2 / len(records))
    print(f"\nChi-square test of independence: chi2={chi2:.3f}, p={p_val:.4f}")
    print(f"Effect size (phi coefficient):    phi={phi:.3f}")
    print("(phi ~0.1 = small, ~0.3 = medium, ~0.5 = large, by conventional thresholds)")


if __name__ == "__main__":
    main()
