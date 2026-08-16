"""
Full-representation activation patching baseline -- no training, just swap the ENTIRE
residual stream vector (not a trained low-rank subspace) at a given layer/position, and
evaluate the same way DAS results were evaluated (IIA, r against the real observed
biased-condition margin).

Motivation (see project report / correspondence): this directly parallels Geiger et al.
(2021)'s own comparison between plain interchange intervention (patching the whole
representation) and DAS (patching an aligned, trained subspace) -- DAS's premise is that a
precise, low-dimensional causal variable improves on the cruder full swap. This baseline
also resolves an open question from the late-layer (26/28/30) DAS results: their lower IIA
there could mean either (a) genuinely weaker causal contribution, or (b) the same/greater
causal contribution, just distributed beyond what a rank-64 subspace at one layer/position
can capture. Full-vector patching removes the rank-64 constraint entirely, so:
  - full-patch IIA >> DAS IIA at a layer -> supports (b), the effect is real but diffuse
  - full-patch IIA ~= DAS IIA at a layer -> DAS was already capturing most of what's there
  - full-patch IIA << DAS IIA at a layer -> unexpected, worth double-checking

No training needed (VanillaIntervention has no trainable parameters), so this is cheap:
one forward pass per example per layer, no optimizer loop.

Usage:
    python full_patch_baseline.py --layers 13 14 15 16 17 18 19 20 21 22 26 28 30
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.scoring import load_model, MODEL_ID
from phase3_das.run_phase3 import intervened_margin, positions_for, evaluate


def make_vanilla_intervenable(pv, lm, layer: int):
    pyvene_layer = layer - 1  # same convention as run_phase3.py -- see its docstring
    config = pv.IntervenableConfig(
        representations=[
            pv.RepresentationConfig(
                layer=pyvene_layer, component="block_output", unit="pos", max_number_of_units=1,
            )
        ],
        intervention_types=pv.VanillaIntervention,  # full swap, no training, no low_rank_dimension
    )
    intervenable = pv.IntervenableModel(config, lm.model)
    intervenable.disable_model_gradients()
    intervenable.set_device(lm.device)
    return intervenable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-results", type=str, default="/kaggle/working/phase1_out/phase1_full_results.json")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/phase3_out")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--layers", type=int, nargs="+",
                         default=[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 28, 30])
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--max-test-examples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import pyvene as pv

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model_id} ...")
    lm = load_model(model_id=args.model_id)

    # Reconstruct the SAME split run_phase3.py used (same seed, same ops) -- test set here
    # must match the one DAS was evaluated on for the comparison to be meaningful.
    phase1_results = json.load(open(args.phase1_results))
    subset = [r for r in phase1_results if r["neutral_prefers_correct"]]
    random.seed(args.seed)
    random.shuffle(subset)
    n_train = int(len(subset) * args.train_frac)
    test_examples = subset[n_train:][: args.max_test_examples]

    trivial_baseline = np.mean([ex["biased"]["margin"] > 0 for ex in test_examples])
    print(f"Trivial baseline: {trivial_baseline:.1%}  (n_test={len(test_examples)})")

    results = {}
    for layer in args.layers:
        intervenable = make_vanilla_intervenable(pv, lm, layer)
        iia, r, _, _ = evaluate(intervenable, lm, test_examples)
        print(f"Layer {layer:3d}: full-patch IIA={iia:.1%}  r={r:.3f}")
        results[layer] = {"full_patch_iia": float(iia), "full_patch_r": float(r)}
        json.dump(results, open(out_dir / "full_patch_results.json", "w"))

    print("\n=== Summary ===")
    print(f"{'layer':>6} {'full_patch_iia':>15} {'full_patch_r':>13}")
    for layer in args.layers:
        res = results[layer]
        print(f"{layer:6d} {res['full_patch_iia']:15.1%} {res['full_patch_r']:13.3f}")
    print(f"\nSaved to {out_dir / 'full_patch_results.json'}")
    print("Compare against phase3_results.json's trained_iia/r per layer.")


if __name__ == "__main__":
    main()
