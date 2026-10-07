"""
Interchange interventions on the answer margin: full-representation patching and DAS.

Base input  = neutral prompt; source input = a biased/control prompt of the same question;
site        = output of decoder block b at the last prompt token.
Margin m    = mean-token log P(c_plus) - mean-token log P(c_minus)  (teacher-forced).
DAS objective: (m_int - m_src)^2, i.e. the intervened neutral run should reproduce the margin
of the real source run -- equivalent to matching the intervened shift to the observed shift.
"""
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from v2.lm import LM, Seq, patch_fn


@dataclass
class BaseItem:
    prefix: list[int]   # tokenized neutral prompt
    c_plus: list[int]
    c_minus: list[int]


class Subspace(torch.nn.Module):
    """W = Q factor of an unconstrained d x k matrix, so W always has orthonormal columns.
    Equivalent to DAS's rotation restricted to its first k rows."""
    def __init__(self, d: int, k: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.A = torch.nn.Parameter(torch.randn(d, k, generator=g))

    def forward(self) -> torch.Tensor:
        return torch.linalg.qr(self.A).Q


def margins(lm: LM, items: list[BaseItem], idx, layer: int | None = None,
            src: torch.Tensor | None = None, W: torch.Tensor | None = None):
    """Margins for items[idx] as (mean-token, first-token, rest) tensors. With src=None no
    intervention is applied. First token and the remaining tokens are split out because
    near the output the patched last-prompt vector directly produces the FIRST answer
    token; `rest` (mean over tokens 2..k, NaN if either candidate is a single token) is
    the part of the margin that is not read off the patched position itself."""
    seqs, pos = [], []
    for i in idx:
        it = items[i]
        seqs += [Seq(it.prefix, it.c_plus), Seq(it.prefix, it.c_minus)]
        pos += [len(it.prefix) - 1] * 2
    fn = None
    if src is not None:
        pos_t = torch.tensor(pos, device=lm.device)
        fn = patch_fn(pos_t, src.repeat_interleave(2, dim=0), W)
    return pair_margins(lm.token_logprobs(seqs, layer, fn))


def pair_margins(lps: list[torch.Tensor]):
    """Per-token log-probs of interleaved (c_plus, c_minus) sequences -> (mean-token,
    first-token, rest) margins, one per pair."""
    mean = torch.stack([lp.mean() for lp in lps])
    first = torch.stack([lp[0] for lp in lps])
    nan = mean.new_tensor(float("nan"))
    rest = torch.stack([lp[1:].mean() if len(lp) > 1 else nan for lp in lps])
    return mean[0::2] - mean[1::2], first[0::2] - first[1::2], rest[0::2] - rest[1::2]


def _src(src_layer: np.ndarray, idx, device):
    """src_layer: one block's source vectors for all items, [n_items, d] (preloaded)."""
    return torch.from_numpy(np.ascontiguousarray(src_layer[np.asarray(idx)], dtype=np.float32)).to(device)


@torch.no_grad()
def evaluate(lm, items, idx, layer, src_layer=None, W=None, bs=32) -> dict:
    """Margins under intervention for every index in idx (src_layer=None: no intervention).
    Returns {"mean", "first", "rest"} numpy arrays."""
    out = {"mean": [], "first": [], "rest": []}
    for s in range(0, len(idx), bs):
        chunk = idx[s:s + bs]
        src = _src(src_layer, chunk, lm.device) if src_layer is not None else None
        for key, v in zip(out, margins(lm, items, chunk, layer, src, W)):
            out[key].append(v.cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def train_das(lm, items, train_idx, val_idx, layer, src_layer, targets, k, seed,
              steps=600, bs=16, lr=2e-3, eval_every=50, log=print):
    """Trains W on train_idx; returns (best W by validation MSE, history). src_layer: this
    block's source vectors [n_items, d]; targets: real source-run margins [n_items]."""
    torch.manual_seed(seed)
    sub = Subspace(lm.d_model, k, seed).to(lm.device)
    opt = torch.optim.Adam(sub.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    order, ptr = rng.permutation(train_idx), 0
    tgt_val = targets[val_idx]

    def val_mse():
        m = evaluate(lm, items, val_idx, layer, src_layer, sub().detach())["mean"]
        return float(np.mean((m - tgt_val) ** 2))

    best = {"step": 0, "val_mse": val_mse(), "A": sub.A.detach().clone()}
    hist = {"train_loss": [], "val": [(0, best["val_mse"])]}
    t0 = time.time()
    for step in range(1, steps + 1):
        if ptr + bs > len(order):
            order, ptr = rng.permutation(train_idx), 0
        batch = order[ptr:ptr + bs]
        ptr += bs
        m = margins(lm, items, batch, layer, _src(src_layer, batch, lm.device), sub())[0]
        loss = ((m - torch.tensor(targets[batch], device=m.device, dtype=m.dtype)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist["train_loss"].append(loss.item())
        if step % eval_every == 0 or step == steps:
            v = val_mse()
            hist["val"].append((step, v))
            if v < best["val_mse"]:
                best = {"step": step, "val_mse": v, "A": sub.A.detach().clone()}
            log(f"    block {layer} k={k} step {step}/{steps} "
                f"train={np.mean(hist['train_loss'][-eval_every:]):.3f} val={v:.3f} "
                f"({time.time() - t0:.0f}s)")
    with torch.no_grad():
        sub.A.copy_(best["A"])
        W = sub().detach()
    hist["best_step"], hist["best_val_mse"] = best["step"], best["val_mse"]
    return W, hist


def metrics(m_int, m_src, m_neu) -> dict:
    """IIA = sign agreement with the real source run, i.e. does the intervention correctly
    predict whether the item flips. Balanced accuracy and r are reported because flip
    classes are imbalanced (an always-flip predictor can beat sign IIA)."""
    m_int, m_src, m_neu = map(np.asarray, (m_int, m_src, m_neu))
    flip, pred = m_src < 0, m_int < 0
    out = {"n": int(len(m_src)), "iia": float(np.mean(pred == flip)),
           "flip_rate": float(np.mean(flip))}
    out["balanced_acc"] = (float(0.5 * (np.mean(pred[flip]) + np.mean(~pred[~flip])))
                           if flip.any() and (~flip).any() else math.nan)
    out["pearson_r"] = (float(np.corrcoef(m_int, m_src)[0, 1])
                        if np.std(m_int) > 0 and np.std(m_src) > 0 else math.nan)
    denom = np.mean(m_src - m_neu)
    out["shift_recovered"] = float(np.mean(m_int - m_neu) / denom) if abs(denom) > 1e-8 else math.nan
    out["mse"] = float(np.mean((m_int - m_src) ** 2))
    return out
