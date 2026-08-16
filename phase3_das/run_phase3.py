"""
Phase 3 -- real DAS training and evaluation. Supersedes calibrate_das.py's placeholder
loss now that the causal graph and candidate layers are finalized (see project
correspondence / report):

    C (candidate-assertion signal) -> R (aligned subspace, layer L) -> ΔMargin(correct_answer, candidate)
    where Margin(correct_answer, candidate) = logP(correct_answer) - logP(candidate)

Training target: for each example we already have a REAL counterfactual pair, not a
synthetic templated one -- base = neutral prompt, source = biased prompt, same question,
same candidate (incorrect_answer). The target is the REAL observed biased-condition margin
for that example (from Phase 1): train the low-rank rotation at layer L so that patching
the source's (biased) representation into the base's (neutral) run reproduces what
actually happens when that example is truly biased, rather than a synthetic label from an
abstract combinatorial causal model (we don't have/need one here -- the "counterfactual"
already exists in the data).

Evaluation (held-out test split): Interchange Intervention Accuracy (IIA) = fraction of
test examples where sign(intervened_margin) matches sign(real_biased_margin) -- does the
intervention correctly reproduce whether this example sycophantically flips. Compared
against:
  - a trivial "always predict no flip" baseline (~1 - Phase 1's flip rate, computed
    directly from the test split's own real margins, no extra compute)
  - an UNTRAINED baseline: same architecture, random-initialized rotation, evaluated before
    any training step -- checks whether training is doing something beyond what an
    arbitrary subspace at this layer would achieve

Known caveats carried over from calibration (see calibrate_das.py docstring for detail):
  - Layer-index correction (Phase-2-convention layer N -> pyvene layer N-1) is a reasoned
    best guess, not independently verified against pyvene internals.
  - Single GPU only -- pyvene does not support device_map="auto" multi-GPU sharding.

Usage:
    python run_phase3.py --layers 15 18 14 16 17 19 20 21 13 --n-steps 500

Requires: pip install pyvene
Requires HF_TOKEN in the environment.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.scoring import load_model, build_inputs, MODEL_ID, SYSTEM_PROMPT


def intervened_margin(intervenable, lm, base_prompt, source_prompt, correct_answer, incorrect_answer,
                       base_pos, source_pos):
    """Runs interchange intervention (patch source's representation into base's run) with
    each candidate answer teacher-forced in turn, returns the resulting margin as a scalar
    tensor (keeps grad if called outside torch.no_grad(), for training)."""
    def score_candidate(candidate_text):
        base_ids, base_prefix_len, cand_len = build_inputs(lm, base_prompt, candidate_text)
        source_ids, _, _ = build_inputs(lm, source_prompt, candidate_text)
        base_ids = base_ids.to(lm.device)
        source_ids = source_ids.to(lm.device)
        unit_locations = {"sources->base": ([[[source_pos]]], [[[base_pos]]])}
        _, intervened_outputs = intervenable(
            {"input_ids": base_ids}, [{"input_ids": source_ids}], unit_locations,
        )
        log_probs = torch.log_softmax(intervened_outputs.logits[0].float(), dim=-1)
        total = log_probs.new_zeros(())
        for k in range(cand_len):
            pos = base_prefix_len - 1 + k
            target = base_ids[0, base_prefix_len + k]
            total = total + log_probs[pos, target]
        return total / cand_len

    lp_correct = score_candidate(correct_answer)
    lp_incorrect = score_candidate(incorrect_answer)
    return lp_correct - lp_incorrect


def positions_for(lm, ex):
    """last-prompt-token positions for the neutral (base) and biased (source) prompts --
    identical regardless of which candidate follows (causal attention), computed once."""
    _, base_prefix_len, _ = build_inputs(lm, ex["neutral_prompt"], ex["correct_answer"])
    _, source_prefix_len, _ = build_inputs(lm, ex["biased_prompt"], ex["correct_answer"])
    return base_prefix_len - 1, source_prefix_len - 1


def make_intervenable(pv, lm, layer: int, low_rank_dim: int):
    pyvene_layer = layer - 1  # see module docstring
    config = pv.IntervenableConfig(
        representations=[
            pv.RepresentationConfig(
                layer=pyvene_layer, component="block_output", unit="pos",
                max_number_of_units=1, low_rank_dimension=low_rank_dim,
            )
        ],
        intervention_types=pv.LowRankRotatedSpaceIntervention,
    )
    intervenable = pv.IntervenableModel(config, lm.model)
    intervenable.disable_model_gradients()
    try:
        intervenable.set_device(lm.device)
    except RuntimeError as e:
        if "offloaded" in str(e):
            raise RuntimeError(
                "set_device() failed because accelerate has offloaded some model modules "
                "to CPU/disk -- this means less GPU memory was free at load time than in "
                "previous successful runs (likely leftover allocation from an earlier "
                "process in the same runtime that didn't fully release memory). Try: "
                "1) check `!nvidia-smi` for stale memory usage, 2) restart the Colab "
                "runtime (Runtime > Restart runtime) for a clean GPU state, then re-run."
            ) from e
        raise
    trainable_params = list(intervenable.get_trainable_parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    assert n_trainable > 0, "No trainable parameters found -- see calibrate_das.py's caveats."
    return intervenable, trainable_params


def train_one_layer(intervenable, trainable_params, lm, train_examples, layer, args, test_examples=None):
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    order = list(range(len(train_examples)))
    random.shuffle(order)
    losses = []
    test_trajectory = []  # list of (step, test_iia, test_r), only populated if args.eval_every > 0
    t0 = time.time()
    for step in range(args.n_steps):
        if step % len(order) == 0 and step > 0:
            random.shuffle(order)
        ex = train_examples[order[step % len(order)]]
        base_pos, source_pos = positions_for(lm, ex)
        target_margin = torch.tensor(ex["biased"]["margin"], device=lm.device, dtype=torch.float32)

        optimizer.zero_grad()
        margin = intervened_margin(
            intervenable, lm, ex["neutral_prompt"], ex["biased_prompt"],
            ex["correct_answer"], ex["incorrect_answer"], base_pos, source_pos,
        )
        loss = (margin - target_margin) ** 2
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 50 == 0:
            recent = np.mean(losses[-50:])
            elapsed = time.time() - t0
            print(f"  layer {layer} step {step+1}/{args.n_steps}  avg_loss(last50)={recent:.4f}  "
                  f"elapsed={elapsed:.0f}s")

        if args.eval_every > 0 and test_examples is not None and (step + 1) % args.eval_every == 0:
            test_iia, test_r, _, _ = evaluate(intervenable, lm, test_examples)
            test_trajectory.append((step + 1, test_iia, test_r))
            print(f"    [test-set check] step {step+1}: IIA={test_iia:.1%}  r={test_r:.3f}")

    return losses, test_trajectory


@torch.no_grad()
def evaluate(intervenable, lm, test_examples):
    correct_sign = 0
    preds, trues = [], []
    for ex in test_examples:
        base_pos, source_pos = positions_for(lm, ex)
        margin = intervened_margin(
            intervenable, lm, ex["neutral_prompt"], ex["biased_prompt"],
            ex["correct_answer"], ex["incorrect_answer"], base_pos, source_pos,
        ).item()
        true_margin = ex["biased"]["margin"]
        preds.append(margin)
        trues.append(true_margin)
        if (margin > 0) == (true_margin > 0):
            correct_sign += 1
    iia = correct_sign / len(test_examples)
    r = float(np.corrcoef(preds, trues)[0, 1]) if len(test_examples) > 1 else float("nan")
    return iia, r, preds, trues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-results", type=str, default="/kaggle/working/phase1_out/phase1_full_results.json")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/phase3_out")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--layers", type=int, nargs="+", default=[15, 18, 14, 16, 17, 19, 20, 21, 13])
    parser.add_argument("--low-rank-dim", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=0,
                         help="If >0, evaluate on the test set every N training steps, within "
                              "the SAME run/random-init, to check for overfitting (test IIA "
                              "rising then falling) without the confound of comparing separate "
                              "runs with different random rotation initializations.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--max-test-examples", type=int, default=150,
                         help="Cap on test-set size to keep eval time bounded.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import pyvene as pv

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model_id} ...")
    lm = load_model(model_id=args.model_id)

    phase1_results = json.load(open(args.phase1_results))
    subset = [r for r in phase1_results if r["neutral_prefers_correct"]]
    random.seed(args.seed)
    torch.manual_seed(args.seed)  # previously unset -- left the rotation's random init
                                    # uncontrolled across runs; re-running "the same" config
                                    # was found to shift IIA by several points on its own,
                                    # independent of any real training-duration effect
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.shuffle(subset)
    n_train = int(len(subset) * args.train_frac)
    train_examples = subset[:n_train]
    test_examples = subset[n_train:][: args.max_test_examples]
    print(f"Train: {len(train_examples)}  Test: {len(test_examples)}")

    trivial_baseline = np.mean([ex["biased"]["margin"] > 0 for ex in test_examples])
    print(f"Trivial 'always predict no flip' baseline accuracy on test set: {trivial_baseline:.1%}")

    all_results = {}
    for layer in args.layers:
        print(f"\n=== Layer {layer} ===")
        intervenable, trainable_params = make_intervenable(pv, lm, layer, args.low_rank_dim)
        untrained_iia, untrained_r, _, _ = evaluate(intervenable, lm, test_examples)
        print(f"Untrained (random rotation) IIA: {untrained_iia:.1%}  r={untrained_r:.3f}")

        losses, test_trajectory = train_one_layer(
            intervenable, trainable_params, lm, train_examples, layer, args, test_examples=test_examples
        )
        trained_iia, trained_r, preds, trues = evaluate(intervenable, lm, test_examples)
        print(f"Trained IIA: {trained_iia:.1%}  r={trained_r:.3f}")
        print(f"(trivial baseline: {trivial_baseline:.1%}, untrained: {untrained_iia:.1%}, "
              f"trained: {trained_iia:.1%})")

        all_results[layer] = {
            "trivial_baseline_iia": float(trivial_baseline),
            "untrained_iia": float(untrained_iia),
            "untrained_r": float(untrained_r),
            "trained_iia": float(trained_iia),
            "trained_r": float(trained_r),
            "final_train_loss_avg50": float(np.mean(losses[-50:])),
            "losses": losses,
            "test_preds": preds,
            "test_trues": trues,
            "test_trajectory": test_trajectory,  # list of (step, test_iia, test_r), empty unless --eval-every > 0
        }
        json.dump(all_results, open(out_dir / "phase3_results.json", "w"))
        print(f"Saved progress to {out_dir / 'phase3_results.json'}")

    print("\n=== Summary across layers ===")
    print(f"{'layer':>6} {'trivial':>8} {'untrained':>10} {'trained':>8} {'r':>7}")
    for layer in args.layers:
        res = all_results[layer]
        print(f"{layer:6d} {res['trivial_baseline_iia']:8.1%} {res['untrained_iia']:10.1%} "
              f"{res['trained_iia']:8.1%} {res['trained_r']:7.3f}")


if __name__ == "__main__":
    main()
