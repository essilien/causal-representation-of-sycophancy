"""
Diagnostic: distinguishes "the bias prompt has no effect" from "the bias prompt has a
real but sub-threshold effect that the binary margin>0 cutoff hides."

Run this if Phase 1's printed flip rate looks implausibly low. Reads
phase1_full_results.json -- no GPU / model needed.

Usage:
    python diagnose_low_flip_rate.py --results-path /kaggle/working/phase1_out/phase1_full_results.json
"""
import argparse
import json
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=str,
                         default="/kaggle/working/phase1_out/phase1_full_results.json")
    args = parser.parse_args()

    results = json.load(open(args.results_path))

    # ---- 1. Margin shift across ALL examples (not just ones that crossed the binary threshold) ----
    shifts = [r["biased"]["margin"] - r["neutral"]["margin"] for r in results]
    print("=== All examples ===")
    print(f"Mean margin shift (biased - neutral): {statistics.mean(shifts):.4f}")
    print(f"Median margin shift: {statistics.median(shifts):.4f}")
    print(f"Fraction with negative shift (pushed toward the incorrect answer): "
          f"{sum(1 for s in shifts if s < 0) / len(shifts):.1%}")

    # ---- 2. Same, restricted to neutral-correct examples (the only ones sycophancy CAN show up in) ----
    nc = [r for r in results if r["neutral_prefers_correct"]]
    nc_shifts = [r["biased"]["margin"] - r["neutral"]["margin"] for r in nc]
    print(f"\n=== Neutral-correct examples only (n={len(nc)}) ===")
    print(f"Mean margin shift: {statistics.mean(nc_shifts):.4f}")
    print(f"Median margin shift: {statistics.median(nc_shifts):.4f}")
    print(f"Fraction with negative shift: {sum(1 for s in nc_shifts if s < 0) / len(nc_shifts):.1%}")
    print(f"Fraction that fully flipped (crossed zero): "
          f"{sum(r['sycophantic_flip'] for r in nc) / len(nc):.1%}")

    # ---- 3. Eyeball the strongest "near misses": pushed hard toward wrong answer but didn't flip ----
    near_misses = sorted(
        [r for r in nc if not r["sycophantic_flip"]],
        key=lambda r: r["biased"]["margin"] - r["neutral"]["margin"]
    )[:8]
    print("\n=== Strongest near-misses (biggest negative shift, but no full flip) ===")
    for r in near_misses:
        shift = r["biased"]["margin"] - r["neutral"]["margin"]
        print(f"- {r['question'][:70]}")
        print(f"  correct={r['correct_answer']!r}  incorrect={r['incorrect_answer']!r}")
        print(f"  neutral margin={r['neutral']['margin']:.3f}  biased margin={r['biased']['margin']:.3f}  shift={shift:.3f}")

    # ---- 4. Eyeball a few of the actual flips too, for a sanity check on the sign/magnitude ----
    flips = [r for r in results if r["sycophantic_flip"]]
    print(f"\n=== Actual flips (n={len(flips)}) ===")
    for r in flips[:8]:
        print(f"- {r['question'][:70]}")
        print(f"  correct={r['correct_answer']!r}  incorrect={r['incorrect_answer']!r}")
        print(f"  neutral margin={r['neutral']['margin']:.3f}  biased margin={r['biased']['margin']:.3f}")


if __name__ == "__main__":
    main()
