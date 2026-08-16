"""
Diagnostic for the untrained_iia == trivial_baseline coincidence seen across all 9 layers
in the full Phase 3 run. Checks whether an UNTRAINED (freshly initialized, random
rotation) intervention numerically perturbs the margin at all, or whether it's a true
no-op -- as opposed to "perturbs it, but never enough to cross zero," which is the more
likely and less alarming explanation.

Reuses run_phase3.py's make_intervenable / intervened_margin / positions_for directly
(no duplicated logic) and reconstructs the EXACT same train/test split (same seed,
same train_frac) that the real run used, so this inspects the same test examples.

Usage:
    python diagnose_untrained_margin.py --layer 18 --n-examples 3
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.scoring import load_model, MODEL_ID
from phase3_das.run_phase3 import make_intervenable, intervened_margin, positions_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-results", type=str, default="/kaggle/working/phase1_out/phase1_full_results.json")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--low-rank-dim", type=int, default=64)
    parser.add_argument("--n-examples", type=int, default=3)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--max-test-examples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import pyvene as pv

    print(f"Loading model {args.model_id} ...")
    lm = load_model(model_id=args.model_id)

    # Reconstruct the EXACT same split run_phase3.py used (same seed, same operations, in
    # the same order) so we're looking at the same test examples.
    phase1_results = json.load(open(args.phase1_results))
    subset = [r for r in phase1_results if r["neutral_prefers_correct"]]
    random.seed(args.seed)
    random.shuffle(subset)
    n_train = int(len(subset) * args.train_frac)
    test_examples = subset[n_train:][: args.max_test_examples]

    print(f"Checking layer {args.layer}, first {args.n_examples} test examples, "
          f"BEFORE any training (fresh random rotation).\n")

    intervenable, _ = make_intervenable(pv, lm, args.layer, args.low_rank_dim)

    for ex in test_examples[: args.n_examples]:
        real_neutral = ex["neutral"]["margin"]
        real_biased = ex["biased"]["margin"]
        base_pos, source_pos = positions_for(lm, ex)
        untrained_margin = intervened_margin(
            intervenable, lm, ex["neutral_prompt"], ex["biased_prompt"],
            ex["correct_answer"], ex["incorrect_answer"], base_pos, source_pos,
        ).item()

        print(f"Q: {ex['question'][:70]}")
        print(f"  real neutral margin (no intervention):      {real_neutral:+.4f}")
        print(f"  real biased margin (true counterfactual):    {real_biased:+.4f}")
        print(f"  UNTRAINED intervened margin:                 {untrained_margin:+.4f}")
        moved = abs(untrained_margin - real_neutral)
        crossed = (untrained_margin > 0) != (real_neutral > 0)
        print(f"  moved by {moved:.4f} from neutral; crossed zero: {crossed}")
        print()

    print("Interpretation:")
    print("- If 'moved by' is consistently near-zero (e.g. <0.05) across examples, the")
    print("  untrained intervention is doing essentially nothing -- worth investigating")
    print("  further (though the calibration sanity check earlier did find a nonzero max")
    print("  logit difference, so a true full no-op seems unlikely).")
    print("- If 'moved by' is clearly nonzero but 'crossed zero' is False for most/all")
    print("  examples, that confirms the expected explanation: the random rotation does")
    print("  perturb the margin, just not enough to flip the sign -- so untrained_iia")
    print("  landing exactly on the trivial baseline is not a bug, just what you'd expect")
    print("  from an arbitrary (untrained) subspace at this layer.")


if __name__ == "__main__":
    main()
