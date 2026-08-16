"""
Verifies the "Margin under assertion prompt" row directly from saved fields, instead of
deriving it as (neutral margin + shift). No new model calls -- reads the already-saved
phase1_full_results.json (biased column) and irrelevant_control_results.jsonl (controlled
column).

Usage:
    python verify_assertion_margin.py \
        --phase1-results /path/to/phase1_full_results.json \
        --irrelevant-results /path/to/irrelevant_control_results.jsonl
"""
import argparse
import json

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase1-results", type=str, required=True)
parser.add_argument("--irrelevant-results", type=str, required=True)
args = parser.parse_args()

# --- Biased column: correct vs incorrect_answer, under the assertion prompt ---
phase1 = json.load(open(args.phase1_results))
subset = [r for r in phase1 if r["neutral_prefers_correct"]]
biased_assertion_margin = np.array([r["biased"]["margin"] for r in subset])
biased_neutral_margin = np.array([r["neutral"]["margin"] for r in subset])

print(f"n (biased/phase1) = {len(subset)}")
print(f"Biased column -- margin under assertion prompt (direct read):  mean={biased_assertion_margin.mean():+.4f}")
print(f"Biased column -- margin under neutral prompt (direct read):    mean={biased_neutral_margin.mean():+.4f}")
print(f"Biased column -- derived via (neutral + shift), for comparison: "
      f"{biased_neutral_margin.mean() + (biased_assertion_margin.mean() - biased_neutral_margin.mean()):+.4f}")

# --- Controlled column: correct vs irrelevant_answer, under the assertion prompt ---
irrel_records = [json.loads(l) for l in open(args.irrelevant_results)]
controlled_assertion_margin = np.array([r["biased_margin_irrel"] for r in irrel_records])
controlled_neutral_margin = np.array([r["neutral_margin_irrel"] for r in irrel_records])

print(f"\nn (controlled/irrelevant) = {len(irrel_records)}")
print(f"Controlled column -- margin under assertion prompt (direct read): mean={controlled_assertion_margin.mean():+.4f}")
print(f"Controlled column -- margin under neutral prompt (direct read):   mean={controlled_neutral_margin.mean():+.4f}")

print("\nCompare these direct-read values against the table's +0.55 / -0.99 "
      "(derived via addition) -- small differences (a few hundredths) are expected from "
      "rounding at intermediate steps; a large discrepancy would indicate the derived "
      "numbers were wrong and should be replaced with these direct reads.")
