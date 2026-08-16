"""
Phase 2 -- candidate-layer prescreen for DAS.

Goal: narrow the ~33-layer search space down to a handful of candidate layers
where a linear probe can predict how much a given bias assertion will move
the model's decision, using a signal that isn't confounded by surface
prompt-length differences (see below). Layers that pass are the ones Phase 3
will spend DAS-training compute on.

Two curves are computed:
  1. shift_curve: layer-wise logit-lens decision margin shift (behavior).
  2. regression_r2: held-out R^2 predicting the *continuous* margin_shift
     (Phase 1's ground truth) from the BIASED-condition hidden state alone,
     at each layer, validated against a per-layer permutation null.

     This replaces an earlier, discarded approach: classifying neutral-vs-
     biased prompts from hidden states, which read ~1.0 accuracy from layer 1
     onward regardless of any real representation -- biased prompts are
     simply longer / contain different surface tokens than neutral ones, so
     a probe can solve that from sequence form alone. Every biased-condition
     prompt in this dataset shares the identical template, so the regression
     target here isn't solvable via a trivial surface cue.

Candidate layers are selected from regression_r2 alone, NOT intersected with
shift_curve. An earlier version required both to be extreme at the same
layer, which produced an empty set: representation and behavioral-effect
magnitude are expected to peak at different layers if the former causally
drives the latter (a mid-network belief representation causing a decision
effect that keeps compounding through later layers is a completely
consistent story, not a contradiction).

Usage:
    python run_phase2.py \
        --phase1-results /kaggle/working/phase1_out/phase1_full_results.json \
        --output-dir /kaggle/working/phase2_out

Requires HF_TOKEN in the environment. Reuses common/scoring.py's SYSTEM_PROMPT
-- do not override it here, or Phase 2's margins become incomparable with
Phase 1's.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.scoring import load_model, MODEL_ID, SYSTEM_PROMPT, LoadedModel


@torch.no_grad()
def layerwise_score(lm: LoadedModel, prompt_text: str, correct_answer: str, incorrect_answer: str) -> dict:
    """Returns:
      - final_margin: Phase-1-equivalent final-layer margin (sanity-check anchor)
      - per_layer_margin: np.array [num_layers], logit-lens (correct_lp - incorrect_lp)
        at each layer, length-normalized per token
      - last_prompt_hidden: np.array [num_layers, hidden_dim], residual stream at the
        last prompt token (decision point), one vector per layer -- identical regardless
        of which candidate follows (causal attention), computed once as a byproduct of
        scoring the correct_answer pass
    """
    chat_prefix = lm.tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": prompt_text}],
        tokenize=False, add_generation_prompt=True,
    )
    prefix_ids = lm.tokenizer(chat_prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(lm.device)
    prefix_len = prefix_ids.shape[1]

    def score_candidate(candidate_text: str, want_hidden: bool = False):
        cand_ids = lm.tokenizer(" " + candidate_text, return_tensors="pt", add_special_tokens=False).input_ids.to(lm.device)
        full_ids = torch.cat([prefix_ids, cand_ids], dim=1)
        cand_len = cand_ids.shape[1]

        out = lm.model(full_ids, output_hidden_states=True)

        final_lp = torch.log_softmax(out.logits[0].float(), dim=-1)
        final_margin_component = sum(
            final_lp[prefix_len + k - 1, full_ids[0, prefix_len + k]].item() for k in range(cand_len)
        ) / cand_len

        # Batched logit-lens across all layers. IMPORTANT (dtype): keep hidden states in
        # the model's native bfloat16 through norm+lm_head (matching their weight dtype),
        # only cast to float32 right before log_softmax -- casting earlier crashes with a
        # dtype mismatch against the bf16 lm_head weights.
        hs_stack = torch.stack([h[0] for h in out.hidden_states], dim=0)  # [L, T, H]
        L, T, H = hs_stack.shape
        normed = lm.model.model.norm(hs_stack.reshape(-1, H))
        layer_logits = lm.model.lm_head(normed).view(L, T, -1).float()
        layer_logprobs = torch.log_softmax(layer_logits, dim=-1)  # [L, T, V]

        per_layer_lp = np.zeros(L)
        for k in range(cand_len):
            pos = prefix_len + k - 1
            target = full_ids[0, prefix_len + k]
            per_layer_lp += layer_logprobs[:, pos, target].cpu().numpy()
        per_layer_lp /= cand_len

        hidden = None
        if want_hidden:
            hidden = np.stack([h[0, prefix_len - 1].float().cpu().numpy() for h in out.hidden_states])
        return final_margin_component, per_layer_lp, hidden

    final_correct, layer_correct, hidden = score_candidate(correct_answer, want_hidden=True)
    final_incorrect, layer_incorrect, _ = score_candidate(incorrect_answer, want_hidden=False)

    return {
        "final_margin": final_correct - final_incorrect,
        "per_layer_margin": layer_correct - layer_incorrect,
        "last_prompt_hidden": hidden,
    }


def run_scoring(lm: LoadedModel, subset: list[dict], margin_path: Path, hidden_path: Path) -> None:
    done_ids = set()
    if margin_path.exists():
        with open(margin_path) as f:
            for line in f:
                done_ids.add(json.loads(line)["idx"])
        print(f"Resuming: {len(done_ids)} examples already done")

    hidden_records = {}
    if hidden_path.exists():
        npz = np.load(hidden_path, allow_pickle=True)
        for key in npz.files:
            idx, cond = key.split("__")
            hidden_records.setdefault(int(idx), {})[cond] = npz[key]

    with open(margin_path, "a") as fout:
        for i, r in enumerate(subset):
            if i in done_ids:
                continue
            neutral_out = layerwise_score(lm, r["neutral_prompt"], r["correct_answer"], r["incorrect_answer"])
            biased_out = layerwise_score(lm, r["biased_prompt"], r["correct_answer"], r["incorrect_answer"])

            record = {
                "idx": i,
                "neutral_final_margin": neutral_out["final_margin"],
                "biased_final_margin": biased_out["final_margin"],
                "neutral_per_layer_margin": neutral_out["per_layer_margin"].tolist(),
                "biased_per_layer_margin": biased_out["per_layer_margin"].tolist(),
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            hidden_records[i] = {"neutral": neutral_out["last_prompt_hidden"], "biased": biased_out["last_prompt_hidden"]}
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(subset)} done")
                flat = {f"{k}__{c}": v for k, r2 in hidden_records.items() for c, v in r2.items()}
                np.savez_compressed(hidden_path, **flat)

    flat = {f"{k}__{c}": v for k, r2 in hidden_records.items() for c, v in r2.items()}
    np.savez_compressed(hidden_path, **flat)


def regression_r2_for_labels(npz, idxs, y_vec, layer: int, seed: int = 0, n_splits: int = 5) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr

    X = np.array([npz[f"{i}__biased"][layer] for i in idxs])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(y_vec, dtype=float)
    for train_idx, test_idx in kf.split(X):
        scaler = StandardScaler().fit(X[train_idx])
        reg = Ridge(alpha=10.0).fit(scaler.transform(X[train_idx]), y_vec[train_idx])
        preds[test_idx] = reg.predict(scaler.transform(X[test_idx]))
    r, _ = pearsonr(y_vec, preds)
    return r ** 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-results", type=str, default="/kaggle/working/phase1_out/phase1_full_results.json")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/phase2_out")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--n-perm", type=int, default=20, help="Number of permutations for the significance null.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1_results = json.load(open(args.phase1_results))
    subset = [r for r in phase1_results if r["neutral_prefers_correct"]]
    for r in subset:
        r["final_margin_shift"] = r["biased"]["margin"] - r["neutral"]["margin"]
    print(f"Neutral-correct examples carried into Phase 2: {len(subset)} / {len(phase1_results)}")

    print(f"Loading model {args.model_id} ...")
    lm = load_model(model_id=args.model_id)
    print("Num layers (incl. embedding output):", lm.num_layers)

    margin_path = out_dir / "phase2_layer_margins.jsonl"
    hidden_path = out_dir / "phase2_hidden_states.npz"
    run_scoring(lm, subset, margin_path, hidden_path)
    print("Phase 2 scoring complete.")

    # ---- Sanity check: logit-lens final layer should closely match Phase 1's final margin ----
    margins = sorted((json.loads(l) for l in open(margin_path)), key=lambda r: r["idx"])
    check_deltas = [abs(m["neutral_per_layer_margin"][-1] - subset[m["idx"]]["neutral"]["margin"]) for m in margins]
    print(f"Sanity check -- mean |logit-lens final layer margin - Phase 1 final margin|: "
          f"{np.mean(check_deltas):.4f} (small = good; if not, see kaggle.md's gotchas section)")

    # ---- Decision-margin shift curve (behavior localization) ----
    neutral_curves = np.array([m["neutral_per_layer_margin"] for m in margins])
    biased_curves = np.array([m["biased_per_layer_margin"] for m in margins])
    shift_curve = (biased_curves - neutral_curves).mean(axis=0)
    shift_curve_std = (biased_curves - neutral_curves).std(axis=0)
    print("\nLayer : mean margin shift (biased - neutral)")
    for L in range(len(shift_curve)):
        print(f"{L:5d} : {shift_curve[L]:+.3f}  (std {shift_curve_std[L]:.3f})")

    # ---- Regression probe (deconfounded) + per-layer permutation null ----
    npz = np.load(hidden_path, allow_pickle=True)
    idxs = sorted(set(int(k.split("__")[0]) for k in npz.files))
    y = np.array([subset[i]["final_margin_shift"] for i in idxs])

    regression_r2 = np.array([regression_r2_for_labels(npz, idxs, y, L) for L in range(lm.num_layers)])
    print("\nLayer : held-out R^2 predicting margin_shift from biased-condition hidden state")
    for L in range(lm.num_layers):
        print(f"{L:5d} : {regression_r2[L]:.3f}")

    rng = np.random.RandomState(0)
    null_r2 = np.zeros((args.n_perm, lm.num_layers))
    import time
    perm_start = time.time()
    for p in range(args.n_perm):
        y_perm = rng.permutation(y)
        for L in range(lm.num_layers):
            null_r2[p, L] = regression_r2_for_labels(npz, idxs, y_perm, L, seed=p)
        elapsed = time.time() - perm_start
        avg_per_perm = elapsed / (p + 1)
        remaining = avg_per_perm * (args.n_perm - p - 1)
        print(f"Permutation {p + 1}/{args.n_perm} done ({elapsed:.0f}s elapsed, "
              f"~{remaining:.0f}s remaining)")
    null_95th = np.percentile(null_r2, 95, axis=0)

    print("\nLayer : real R^2 : null 95th pct : significant?")
    for L in range(lm.num_layers):
        sig = regression_r2[L] > null_95th[L]
        print(f"{L:5d} : {regression_r2[L]:.3f} : {null_95th[L]:.3f} : {'YES' if sig else 'no'}")

    # ---- Candidate layers: from regression_r2 alone, see module docstring for why ----
    candidate_layers = [L for L in range(lm.num_layers) if regression_r2[L] > null_95th[L]]
    ranked = sorted(candidate_layers, key=lambda L: -regression_r2[L])
    print("\nCandidate layers (representation significantly above permutation null):", candidate_layers)
    print("Ranked by R^2, highest first (prioritize these for Phase 3 DAS):", ranked)

    # ---- Plot ----
    try:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(shift_curve, color="tab:red", label="decision margin shift (behavior)")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Mean margin shift (biased - neutral)", color="tab:red")
        ax2 = ax1.twinx()
        ax2.plot(regression_r2, color="tab:blue", label="regression R^2 (representation)")
        ax2.plot(null_95th, color="tab:blue", linestyle="--", linewidth=1, label="permutation null (95th pct)")
        ax2.set_ylabel("R^2", color="tab:blue")
        plt.title("Phase 2: representation (deconfounded) vs. behavior localization")
        fig.tight_layout()
        plt.savefig(out_dir / "phase2_layer_curves.png", dpi=150)
        print(f"Saved plot to {out_dir / 'phase2_layer_curves.png'}")
    except ImportError:
        print("matplotlib not available, skipping plot")

    np.savez(
        out_dir / "phase2_summary.npz",
        shift_curve=shift_curve, regression_r2=regression_r2,
        null_95th=null_95th, candidate_layers=np.array(candidate_layers),
    )
    print("Saved Phase 2 summary to", out_dir / "phase2_summary.npz")


if __name__ == "__main__":
    main()
