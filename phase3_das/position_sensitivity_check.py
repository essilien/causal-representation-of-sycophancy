"""
Position-sensitivity check (no training) -- tests whether the causally relevant
information at a late layer is concentrated at the last-prompt-token position (our
default, used everywhere in this project since Phase 1) or has migrated toward the
asserted-answer span itself by that depth.

Motivation: our DAS-vs-full-patch gap grows with depth (see project report). One
candidate explanation, raised in project correspondence, is that assertion-relevant
information may diffuse away from the last-prompt-token position via attention as depth
increases -- in which case BOTH DAS and full-patch at that position would be missing
information that's actually sitting elsewhere in the sequence by that layer. This script
tests that cheaply: full (untrained) patching, same as full_patch_baseline.py, but with
the SOURCE position moved to the end of the asserted-answer span within the biased prompt
(right after "...I think the answer is {incorrect_answer}", before "but I'm really not
sure."), instead of the end of the whole biased prompt. The BASE position (neutral
prompt) is unchanged, since the neutral prompt has no equivalent span to anchor to.

If this alternate-position IIA is notably higher than the last-token-position result
already measured (63.6% at layer 28, from full_patch_baseline.py), that supports the
information-migration hypothesis and would justify retraining DAS at this position. If
similar or lower, the hypothesis is not supported and the last-token position was not the
bottleneck.

Position-finding approach: rather than re-tokenizing a truncated version of the prompt
(which both risks a tokenizer merge-boundary mismatch right at the cut point, and would
use a differently-rendered chat template than the rest of the pipeline), the answer
span's end is located via character-offset mapping within the SAME full-prompt
tokenization (add_generation_prompt=True) used everywhere else in this project. An
earlier version of this script used the truncate-and-retokenize approach and failed on
100% of examples for exactly this reason -- see project history.

Usage:
    python position_sensitivity_check.py --layer 28
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.scoring import load_model, MODEL_ID, SYSTEM_PROMPT
from phase3_das.run_phase3 import intervened_margin, evaluate
from phase3_das.full_patch_baseline import make_vanilla_intervenable


def answer_span_end_position(lm, biased_prompt: str, incorrect_answer: str) -> int:
    """Token index right after the asserted answer's own text, located within the SAME
    full-prompt tokenization (add_generation_prompt=True) used everywhere else in this
    project -- not by re-tokenizing a truncated string separately, which both (a) risks a
    tokenizer merge-boundary mismatch right at the cut point, and (b) would use a
    differently-rendered template (add_generation_prompt=False) than the rest of the
    pipeline, making any resulting position invalid for the real intervention call."""
    chat = lm.tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": biased_prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    assert biased_prompt in chat, "biased_prompt not found verbatim in the rendered chat template."
    prompt_start = chat.index(biased_prompt)
    answer_start_in_prompt = biased_prompt.index(incorrect_answer)
    answer_end_char = prompt_start + answer_start_in_prompt + len(incorrect_answer)

    enc = lm.tokenizer(chat, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"][0].tolist()  # [(char_start, char_end), ...] per token

    token_idx = None
    for i, (start, end) in enumerate(offsets):
        if end <= answer_end_char:
            token_idx = i
        else:
            break
    assert token_idx is not None, "Could not locate the answer span's end within the tokenized prompt."
    return token_idx + 1  # position of the token right after the answer span


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-results", type=str, default="/kaggle/working/phase1_out/phase1_full_results.json")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/phase3_out")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--max-test-examples", type=int, default=150)
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
    random.shuffle(subset)
    n_train = int(len(subset) * args.train_frac)
    test_examples = subset[n_train:][: args.max_test_examples]

    trivial_baseline = np.mean([ex["biased"]["margin"] > 0 for ex in test_examples])
    print(f"Trivial baseline: {trivial_baseline:.1%}  (n_test={len(test_examples)})")

    intervenable = make_vanilla_intervenable(pv, lm, args.layer)

    # Reuse evaluate() from run_phase3.py, but with a CUSTOM source position per example
    # instead of positions_for()'s default (end of whole prompt). Reimplemented inline
    # since evaluate() doesn't take a position-override hook.
    correct_sign = 0
    preds, trues, skipped = [], [], 0
    for ex in test_examples:
        chat_prefix = lm.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ex["neutral_prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        base_pos = lm.tokenizer(chat_prefix, return_tensors="pt", add_special_tokens=False).input_ids.shape[1] - 1
        try:
            source_pos = answer_span_end_position(lm, ex["biased_prompt"], ex["incorrect_answer"]) - 1
        except (ValueError, AssertionError) as e:
            skipped += 1
            continue
        with torch.no_grad():
            margin = intervened_margin(
                intervenable, lm, ex["neutral_prompt"], ex["biased_prompt"],
                ex["correct_answer"], ex["incorrect_answer"], base_pos, source_pos,
            ).item()
        true_margin = ex["biased"]["margin"]
        preds.append(margin)
        trues.append(true_margin)
        if (margin > 0) == (true_margin > 0):
            correct_sign += 1

    n_used = len(preds)
    iia = correct_sign / n_used
    r = float(np.corrcoef(preds, trues)[0, 1])
    print(f"\nLayer {args.layer}, source position = end of asserted-answer span "
          f"(n={n_used}, skipped={skipped}):")
    print(f"  IIA = {iia:.1%}   r = {r:.3f}")
    print(f"  (for comparison, full_patch_baseline.py at the last-prompt-token position "
          f"found IIA=63.6% at layer 28 -- check full_patch_results.json for other layers)")

    json.dump(
        {"layer": args.layer, "position": "answer_span_end", "iia": iia, "r": r,
         "n_used": n_used, "skipped": skipped},
        open(out_dir / f"position_check_layer{args.layer}.json", "w"),
    )


if __name__ == "__main__":
    main()
