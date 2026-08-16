"""
Phase 3 calibration run -- NOT the real DAS training, just answers: does pyvene's
interchange intervention work on this model, does it fit in memory, and how long does one
training step take (to extrapolate to the full candidate-layer plan, 13-22).

STATUS (updated after real runs, see project history):
- pyvene does NOT support multi-GPU model sharding (confirmed via pyvene's own GitHub
  issue tracker) -- device_map="auto" across multiple GPUs will fail inside pyvene's
  gather/scatter internals. Must run single-GPU only.
- Confirmed working single-GPU on a Colab L4: model loads, trainable parameters are
  found (a 64x4096 rotation matrix), and intervenable.set_device(...) is REQUIRED --
  without it, the intervention's own parameters default-init on CPU while the model runs
  on GPU, causing a device mismatch the first time the intervention actually executes.

REMAINING CAVEATS (read before running):
- The training loss below is a PLACEHOLDER, not the final Phase 3 objective. It pushes the
  intervened base run to assign high probability to incorrect_answer's tokens (mimicking
  the sycophantic flip), which exercises the same forward+backward compute path real
  training will use, but the actual causal-model-grounded target/loss for Phase 3 proper
  still needs to be designed carefully (this script is deliberately NOT that design step).
- The layer-index correction (Phase-2-convention layer N -> pyvene layer N-1) is a
  reasoned best guess, not independently verified against pyvene internals.

Usage:
    python calibrate_das.py --layer 18 --n-steps 200

Requires: pip install pyvene
Requires HF_TOKEN in the environment. Single GPU only (see STATUS above).
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-results", type=str, default="/kaggle/working/phase1_out/phase1_full_results.json")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--low-rank-dim", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import pyvene as pv  # imported here so --help works without pyvene installed

    print(f"Loading model {MODEL_ID} ...")
    lm = load_model(model_id=MODEL_ID)
    if hasattr(lm.model, "hf_device_map"):
        print("Model device map:", lm.model.hf_device_map)

    phase1_results = json.load(open(args.phase1_results))
    subset = [r for r in phase1_results if r["neutral_prefers_correct"]]
    random.seed(args.seed)
    random.shuffle(subset)
    examples = subset[: args.n_steps]
    print(f"Calibration run: {len(examples)} examples, layer {args.layer}, "
          f"low_rank_dimension={args.low_rank_dim}")

    # IMPORTANT: layer-index correction. Phase 2's layer numbering treats hidden_states[k]
    # as "layer k", where hidden_states[0] = embedding output (no transformer block applied
    # yet) and hidden_states[k] = output of transformer block (k-1) for k >= 1. pyvene's
    # RepresentationConfig(layer=N, component="block_output") most likely maps directly to
    # model.layers[N]'s output (0-indexed block number) -- one less than Phase 2's
    # convention for the same physical site. Subtracting 1 here to align them; this is a
    # reasoned best guess, not independently verified against pyvene internals -- if you
    # get a chance, confirm by comparing a captured activation against the corresponding
    # vector in phase2_hidden_states.npz for the same example.
    pyvene_layer = args.layer - 1
    print(f"Phase-2-convention layer {args.layer} -> pyvene layer={pyvene_layer} "
          f"(model.layers[{pyvene_layer}] output)")

    config = pv.IntervenableConfig(
        representations=[
            pv.RepresentationConfig(
                layer=pyvene_layer,
                component="block_output",
                unit="pos",
                max_number_of_units=1,
                low_rank_dimension=args.low_rank_dim,
            )
        ],
        intervention_types=pv.LowRankRotatedSpaceIntervention,
    )
    intervenable = pv.IntervenableModel(config, lm.model)
    intervenable.disable_model_gradients()
    # Now safe: confirmed single-GPU (L4) in practice, so no device_map="auto" sharding
    # conflict. Without this, the intervention's own trainable parameters (the rotation
    # matrix) default-initialize on CPU while the model runs on GPU, causing a device
    # mismatch inside pyvene's RotateLayer.forward() the first time it's actually used.
    intervenable.set_device(lm.device)

    trainable_params = list(intervenable.get_trainable_parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters found: {len(trainable_params)} tensors, {n_trainable} total scalars")
    assert n_trainable > 0, (
        "No trainable parameters found -- intervenable.get_trainable_parameters() likely "
        "isn't the right call for this pyvene version, or the intervention wasn't wired up. "
        "Fix this before running any real steps: training would silently optimize nothing."
    )
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    step_times, losses = [], []
    oom = False
    for step, ex in enumerate(examples):
        t0 = time.time()
        try:
            base_ids, base_prefix_len, _ = build_inputs(lm, ex["neutral_prompt"], ex["incorrect_answer"])
            source_ids, source_prefix_len, _ = build_inputs(lm, ex["biased_prompt"], ex["incorrect_answer"])
            base_ids = base_ids.to(lm.device)
            source_ids = source_ids.to(lm.device)

            base_pos = base_prefix_len - 1
            source_pos = source_prefix_len - 1

            optimizer.zero_grad()
            unit_locations = {"sources->base": ([[[source_pos]]], [[[base_pos]]])}
            # Only request the extra non-intervened forward pass on step 0 (for the sanity
            # check below) -- requesting it every step would inflate per-step timing above
            # what real training actually costs.
            want_original = (step == 0)
            original_outputs, intervened_outputs = intervenable(
                {"input_ids": base_ids},
                [{"input_ids": source_ids}],
                unit_locations,
                output_original_output=want_original,
            )

            if step == 0:
                # One-time check: does the intervention actually change anything? If
                # original and intervened logits are identical (or near-identical), the
                # intervention is a silent no-op -- wrong layer/position/component -- and
                # every subsequent step would be wasted compute measuring nothing.
                with torch.no_grad():
                    diff = (intervened_outputs.logits - original_outputs.logits).abs().max().item()
                print(f"[sanity check] max |intervened_logits - original_logits| at step 1: {diff:.6f}")
                assert diff > 1e-4, (
                    "Intervention appears to be a no-op (logits barely changed). Stop and "
                    "debug the layer index / unit_locations / component before running "
                    "more steps -- see the layer-index caveat in this script's comments."
                )

            # PLACEHOLDER calibration loss -- see module docstring. Pushes the intervened
            # base run toward assigning high probability to incorrect_answer's own tokens.
            logits = intervened_outputs.logits[0]
            target_ids = base_ids[0, base_prefix_len:]
            pred_logits = logits[base_prefix_len - 1: base_prefix_len - 1 + len(target_ids)]
            loss = torch.nn.functional.cross_entropy(pred_logits, target_ids)

            loss.backward()
            optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_times.append(time.time() - t0)
            losses.append(loss.item())

            if (step + 1) % 20 == 0:
                mem_str = ""
                if torch.cuda.is_available():
                    mems = [f"gpu{i}={torch.cuda.max_memory_allocated(i)/1e9:.2f}GB"
                            for i in range(torch.cuda.device_count())]
                    mem_str = "  peak_mem: " + " ".join(mems)
                # rolling average over the last 20 steps, not the single raw step loss --
                # with batch_size=1, per-step loss varies a lot by example difficulty alone,
                # so a single snapshot is a noisy, easily-misleading signal on its own.
                recent_avg_loss = np.mean(losses[-20:])
                print(f"step {step+1}/{len(examples)}  avg_loss(last20)={recent_avg_loss:.4f}  "
                      f"avg_step_time={np.mean(step_times[-20:]):.3f}s{mem_str}")

        except torch.cuda.OutOfMemoryError:
            print(f"OOM at step {step + 1}")
            oom = True
            break
        except Exception as e:
            print(f"Error at step {step + 1}: {type(e).__name__}: {e}")
            print("This is exactly the kind of thing this calibration run is meant to "
                  "surface -- see module docstring caveats about pyvene API uncertainty.")
            raise

    print("\n=== Calibration summary ===")
    print(f"Completed steps: {len(step_times)} / {len(examples)}")
    if step_times:
        print(f"Mean step time: {np.mean(step_times):.3f}s (std {np.std(step_times):.3f}s)")
        print(f"Median step time: {np.median(step_times):.3f}s")
        # Extrapolate to the full candidate-layer plan (13-22, 10 layers)
        for target_steps in [1000, 3000, 5000]:
            per_layer_hours = np.mean(step_times) * target_steps / 3600
            print(f"  Extrapolated: {target_steps} steps/layer x 10 candidate layers "
                  f"= {per_layer_hours * 10:.1f}h total (at {target_steps} steps/layer)")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            peak = torch.cuda.max_memory_allocated(i) / 1e9
            print(f"GPU {i} peak memory: {peak:.2f}GB / {total:.2f}GB")
    print(f"OOM encountered: {oom}")
    if len(losses) >= 10:
        from scipy.stats import linregress
        steps_arr = np.arange(len(losses))
        slope, intercept, r_value, p_value, std_err = linregress(steps_arr, losses)
        first_half_mean = np.mean(losses[: len(losses) // 2])
        second_half_mean = np.mean(losses[len(losses) // 2:])
        print(f"\nLoss trend (linear regression over all {len(losses)} steps):")
        print(f"  slope={slope:.5f} per step  (p={p_value:.4f}, r={r_value:.3f})")
        print(f"  first-half mean={first_half_mean:.3f}  second-half mean={second_half_mean:.3f}")
        if p_value < 0.05 and slope < 0:
            print("  -> statistically significant downward trend")
        elif p_value < 0.05 and slope > 0:
            print("  -> statistically significant UPWARD trend -- worth investigating")
        else:
            print("  -> no statistically significant trend detected at this step count "
                  "(p >= 0.05). With batch_size=1, per-step loss variance is high, so this "
                  "could mean 'not learning yet' or just 'too few/noisy steps to tell' -- "
                  "not conclusive either way. More steps and/or a rolling-average view "
                  "(printed during training above) are more informative than this single "
                  "regression test at n<a few hundred.")
    elif losses:
        print(f"\nOnly {len(losses)} loss values collected -- too few for a trend test; "
              f"use --n-steps >= a few hundred before trying to read a trend into this.")

    if losses:
        loss_log_path = Path(f"calibration_losses_layer{args.layer}.json")
        json.dump(losses, open(loss_log_path, "w"))
        print(f"\nSaved all {len(losses)} per-step losses to {loss_log_path.resolve()} "
              f"for later inspection/plotting.")


if __name__ == "__main__":
    main()
