"""
Model wrapper: loading, chat-template tokenization, batched teacher-forced scoring, and
residual-stream hooks for recording and interchange interventions.

Replaces pyvene from v1. A plain forward hook on model.model.layers[b] is all that the
interventions here need, and owning it (a) allows batching with per-example positions,
(b) avoids pyvene's layer-index convention ambiguity flagged in v1's run_phase3.py, and
(c) works for any HF decoder that exposes model.model.layers (Llama, Qwen2, ...).
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from v2.config import SYSTEM_PROMPT


@dataclass
class Seq:
    prefix: list[int]  # chat-formatted prompt, ends right before the assistant's answer
    cand: list[int]    # teacher-forced answer tokens (may be empty for prompt-only runs)

    @property
    def last_prompt_pos(self) -> int:
        return len(self.prefix) - 1


class LM:
    def __init__(self, model, tokenizer=None, system_prompt: str = SYSTEM_PROMPT):
        self.model = model
        self.tok = tokenizer
        self.system_prompt = system_prompt
        self.layers = model.model.layers
        self.n_layers = len(self.layers)
        self.d_model = model.config.hidden_size
        self.device = next(model.parameters()).device
        pad = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        if pad is None and tokenizer is not None:
            pad = tokenizer.eos_token_id
            pad = pad[0] if isinstance(pad, list) else pad
        self.pad_id = pad if pad is not None else 0

    @classmethod
    def load(cls, model_id: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        kwargs = dict(device_map={"": 0} if torch.cuda.is_available() else None,
                      attn_implementation="sdpa")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, **kwargs)
        except TypeError:  # transformers < 4.56
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kwargs)
        model.eval().requires_grad_(False)
        return cls(model, tok)

    # ---- tokenization ------------------------------------------------------------------
    def encode_prompt(self, user_text: str) -> list[int]:
        chat = self.tok.apply_chat_template(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True)
        return self.tok(chat, add_special_tokens=False).input_ids

    def encode_answer(self, text: str) -> list[int]:
        # No leading space (v1 prepended " "): after the assistant header the model's own
        # first token has no leading space, so this is the natural continuation.
        return self.tok(text, add_special_tokens=False).input_ids

    # ---- forward passes ----------------------------------------------------------------
    def _collate(self, seqs: list[Seq]):
        lens = [len(s.prefix) + len(s.cand) for s in seqs]
        T = max(lens)
        ids = torch.full((len(seqs), T), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):  # right padding: real tokens keep positions 0..len-1
            ids[i, :lens[i]] = torch.tensor(s.prefix + s.cand)
            mask[i, :lens[i]] = 1
        return ids.to(self.device), mask.to(self.device)

    def token_logprobs(self, seqs: list[Seq], layer: int | None = None, fn=None) -> list[torch.Tensor]:
        """Per-token log P(cand_j | prefix, cand_<j) for each sequence (float32, keeps grad
        if called with grad enabled). If `fn` is given, it rewrites the output of decoder
        block `layer` (see patch_fn). The LM head is applied only at the positions that
        predict candidate tokens, which keeps memory small during DAS training."""
        ids, mask = self._collate(seqs)
        with self.hook(layer, fn):
            h = self.model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        rows, cols, tgt, sizes = [], [], [], []
        for i, s in enumerate(seqs):
            P, k = len(s.prefix), len(s.cand)
            rows += [i] * k
            cols += list(range(P - 1, P - 1 + k))
            tgt += s.cand
            sizes.append(k)
        logits = self.model.lm_head(h[rows, cols]).float()
        tgt = torch.tensor(tgt, device=logits.device)
        lp = logits.log_softmax(-1).gather(1, tgt[:, None]).squeeze(1)
        return list(torch.split(lp, sizes))

    @torch.no_grad()
    def record_last_prompt(self, prefixes: list[list[int]]) -> torch.Tensor:
        """Output of every decoder block at each prompt's last token -> [B, n_layers, d]
        (float16, CPU). Recorded with the same hooks the interventions use, so the source
        vectors are by construction the representations that get patched."""
        seqs = [Seq(p, []) for p in prefixes]
        pos = torch.tensor([s.last_prompt_pos for s in seqs], device=self.device)
        rows = torch.arange(len(seqs), device=self.device)
        out = torch.empty(len(seqs), self.n_layers, self.d_model, dtype=torch.float16)
        handles = []
        for b, layer in enumerate(self.layers):
            def rec(mod, inp, o, b=b):
                hs = o[0] if isinstance(o, tuple) else o
                out[:, b] = hs[rows, pos].detach().to(torch.float16).cpu()
            handles.append(layer.register_forward_hook(rec))
        try:
            ids, mask = self._collate(seqs)
            self.model.model(input_ids=ids, attention_mask=mask)
        finally:
            for h in handles:
                h.remove()
        return out

    @contextmanager
    def hook(self, layer: int | None, fn):
        if layer is None or fn is None:
            yield
            return

        def _hook(mod, inp, out):
            if isinstance(out, tuple):
                return (fn(out[0]),) + tuple(out[1:])
            return fn(out)

        handle = self.layers[layer].register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()


def patch_fn(pos: torch.Tensor, src: torch.Tensor, W: torch.Tensor | None = None):
    """Interchange intervention at one position per sequence.
    pos: [B] positions; src: [B, d] source representations at the same block.
      W is None -> full-representation patching:  h <- src
      W [d, k] with orthonormal columns -> DAS:   h <- h + W W^T (src - h)
    Computed in float32; gradients flow into W only (the model is frozen)."""
    def fn(h):
        rows = torch.arange(h.shape[0], device=h.device)
        base = h[rows, pos]
        s = src.to(h.device)
        if W is None:
            new = s.to(h.dtype)
        else:
            delta = s.float() - base.float()
            new = (base.float() + (delta @ W) @ W.T).to(h.dtype)
        h = h.clone()
        h[rows, pos] = new
        return h
    return fn
