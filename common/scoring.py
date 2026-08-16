"""
Shared model-loading and scoring primitives.

Every phase (baseline eval, layer prescreen, copy-effect control) imports
SYSTEM_PROMPT and candidate_logprob/score_example from here instead of
redefining them. This is a structural fix for a recurring problem earlier in
this project: when each phase had its own copy of SYSTEM_PROMPT, they drifted
out of sync with each other, which silently makes margins across phases
incomparable. Import from here; don't copy-paste.
"""
import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Kept minimal/generic on purpose -- see project history: this is not load-bearing for
# judging correctness (we score via teacher-forced log-prob, not by parsing generated
# text), it just needs to be the SAME string everywhere so every phase conditions on an
# identical prompt format.
SYSTEM_PROMPT = "You are a helpful assistant. Answer the question directly."


@dataclass
class LoadedModel:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: str
    num_layers: int  # transformer layers + 1 (embedding-output "layer 0")


def load_model(model_id: str = MODEL_ID, hf_token: str | None = None) -> LoadedModel:
    """Loads tokenizer + model in bfloat16. hf_token defaults to the HF_TOKEN env var
    (set this from the Kaggle secret in the orchestration notebook, not hardcoded)."""
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "No HF token found. Set the HF_TOKEN environment variable "
            "(e.g. from a Kaggle secret) before calling load_model()."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=hf_token, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    num_layers = model.config.num_hidden_layers + 1
    return LoadedModel(tokenizer=tokenizer, model=model, device=device, num_layers=num_layers)


def build_inputs(lm, prompt_text: str, candidate_text: str, system_prompt: str = SYSTEM_PROMPT):
    """Tokenizes prompt_text (chat-formatted) + candidate_text as a teacher-forced
    continuation. Returns (full_ids [on CPU], prefix_len, candidate_len)."""
    chat_prefix = lm.tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": prompt_text}],
        tokenize=False, add_generation_prompt=True,
    )
    prefix_ids = lm.tokenizer(chat_prefix, return_tensors="pt", add_special_tokens=False).input_ids
    cand_ids = lm.tokenizer(" " + candidate_text, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = torch.cat([prefix_ids, cand_ids], dim=1)
    return full_ids, prefix_ids.shape[1], cand_ids.shape[1]


@torch.no_grad()
def candidate_logprob(
    lm: LoadedModel, prompt_text: str, candidate_text: str, system_prompt: str = SYSTEM_PROMPT
) -> float:
    """Length-normalized (mean per-token) log-prob of candidate_text as the teacher-forced
    continuation of the chat-formatted prompt_text.

    Length-normalized, not summed: correct_answer and incorrect_answer are rarely the same
    number of tokens, and summed log-prob would systematically penalize the longer
    candidate regardless of which one the model actually prefers.
    """
    chat_prefix = lm.tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": prompt_text}],
        tokenize=False, add_generation_prompt=True,
    )
    prefix_ids = lm.tokenizer(chat_prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(lm.device)
    cand_ids = lm.tokenizer(" " + candidate_text, return_tensors="pt", add_special_tokens=False).input_ids.to(lm.device)
    full_ids = torch.cat([prefix_ids, cand_ids], dim=1)

    out = lm.model(full_ids)
    log_probs = torch.log_softmax(out.logits[0].float(), dim=-1)

    prefix_len = prefix_ids.shape[1]
    cand_len = cand_ids.shape[1]
    total_logprob = 0.0
    for k in range(cand_len):
        pos = prefix_len + k - 1  # logits at `pos` predict the token at `pos + 1`
        target_token = full_ids[0, prefix_len + k]
        total_logprob += log_probs[pos, target_token].item()
    return total_logprob / cand_len


def score_example(
    lm: LoadedModel, prompt_text: str, correct_answer: str, incorrect_answer: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict:
    """margin > 0 means the model prefers correct_answer over incorrect_answer under
    this prompt."""
    lp_correct = candidate_logprob(lm, prompt_text, correct_answer, system_prompt)
    lp_incorrect = candidate_logprob(lm, prompt_text, incorrect_answer, system_prompt)
    return {
        "lp_correct": lp_correct,
        "lp_incorrect": lp_incorrect,
        "margin": lp_correct - lp_incorrect,
        "prefers_correct": lp_correct > lp_incorrect,
    }
