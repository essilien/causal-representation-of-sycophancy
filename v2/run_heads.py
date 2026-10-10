"""
E13: which attention heads read the inserted answer into the last prompt token, and how does
an assertion amplify that read relative to a mere mention?

Phase 1 (recording). For every neutral-correct item and every condition in --conditions, one
teacher-forced forward pass of the (c_plus, c_minus) pair. At every block and head, for the
query at the last prompt token p, the attention weights alpha_t and values v_t are recomputed
from the inputs of the attention module (checked against the module's own output on the
first batch), and the head's output is split by source position group G:
    o_h^G = W_O^h sum_{t in G} alpha_{h,t} v_{h,t}
Stored per (item, block, head), see STATS:
    attn_G : sum_{t in G} alpha_{h,t}                   (G = answer, pre, post)
    dla_G  : o_h^G . a / rms                            (G = answer, pre, post, all positions)
with a = g * (U[c_minus_1] - U[c_plus_1]) (first answer tokens, final RMSNorm gain g) and rms
the RMS of the final residual stream at p. dla_G is the direct contribution of the head's
read of G to the first-token logit difference (positive = toward c_minus). Items whose
candidates share their first token have a = 0; their dla_* are NaN.
A condition "<cond>+tag" is recorded under the tag knockout (the answer tokens cannot attend
to the framing before them, at every block; run_knockout's (pre, answer) route): does the
mark change how much the late heads attend to the answer, or what they read from it?

Phase 2 (head knockout). Items are split by row parity. Heads are ranked on the even rows by
mean dla_answer under --select-condition (and, separately, under --mention-condition); on the
odd rows, the selected heads' last-token query is blocked from the answer tokens. Head sets:
top-k by assertion, top-k by mention, k random heads from the second half of the network.
The first-token shift eliminated, per --ko-conditions, tells whether the heads that carry the
asserted answer also carry the mentioned one, and the answer in the other assertion
templates.

Outputs <results>/<model>/heads/<condition>.npz (phase 1) and heads/knockout.json (phase 2).
"""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from v2.config import CONDITIONS, MODELS, build_prompt
from v2.das import PARTS, pair_scores
from v2.data import read_jsonl, write_json
from v2.lm import LM, Seq
from v2.positions import ALL_GROUPS, AssertItem, assert_items
from v2.run_knockout import blocked, knockout, scores

SRC_GROUPS = ["answer", "pre", "post"]
STATS = [f"attn_{g}" for g in SRC_GROUPS] + [f"dla_{g}" for g in SRC_GROUPS] + ["dla_all"]
MAX_REL_ERR = 0.05  # recomputed vs. actual attention output (bf16 on GPU)


def neutral_items(lm, nc):
    """The plain question: no inserted text, only the last position is defined."""
    out = []
    for it in nc:
        ids = lm.encode_prompt(build_prompt("neutral", it))
        pos = {g: [] for g in ALL_GROUPS} | {"last": [len(ids) - 1], "suffix": [len(ids) - 1]}
        out.append(AssertItem(ids, lm.encode_answer(it["c_plus"]), lm.encode_answer(it["c_minus"]), pos))
    return out, list(range(len(nc)))


def items_for(lm, nc, cond):
    return neutral_items(lm, nc) if cond == "neutral" else assert_items(lm, nc, cond)


def answer_dirs(lm, items):
    """[n, d] float32 a = g * (U[c_minus_1] - U[c_plus_1]); zero rows where the first tokens agree."""
    U = lm.model.get_output_embeddings().weight
    g = lm.model.model.norm.weight.float()
    fp = torch.tensor([it.c_plus[0] for it in items], device=U.device)
    fm = torch.tensor([it.c_minus[0] for it in items], device=U.device)
    A = (U[fm].float() - U[fp].float()) * g
    A[fp == fm] = 0
    return A, (fp == fm).cpu().numpy()


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


class HeadRecorder:
    """Hooks that split each head's last-token output by source group (see module docstring).
    Rows 0, 2, 4, ... of the pair batch are recorded (the c_plus sequence of each item; the
    prompt, and hence the last-token state, is the same for c_minus)."""

    def __init__(self, lm, items, idx, A, check=False):
        self.lm, self.check = lm, check
        n = len(idx)
        dev = lm.device
        self.sr = torch.arange(0, 2 * n, 2, device=dev)
        self.p = torch.tensor([len(items[i].prefix) - 1 for i in idx], device=dev)
        T = max(len(items[i].prefix) + max(len(items[i].c_plus), len(items[i].c_minus)) for i in idx)
        self.gmask = {}
        for g in SRC_GROUPS:
            m = torch.zeros(n, T, dtype=torch.bool)
            for r, i in enumerate(idx):
                m[r, items[i].pos.get(g, [])] = True
            self.gmask[g] = m.to(dev)
        self.A = A  # [n, d]
        self.buf, self.z = {}, {}
        self.rms = None
        self.max_err = 0.0

    def _attn_pre(self, b):
        def hook(mod, args, kwargs):
            h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            cos, sin = kwargs["position_embeddings"]
            am = kwargs.get("attention_mask")
            sr, p = self.sr, self.p
            n, T = len(sr), h.shape[1]
            H = mod.config.num_attention_heads
            KV = mod.config.num_key_value_heads
            dh = mod.head_dim
            hs = h[sr]
            q = mod.q_proj(hs[torch.arange(n), p]).float().view(n, H, dh)
            k = mod.k_proj(hs).float().view(n, T, KV, dh)
            v = mod.v_proj(hs).float().view(n, T, KV, dh)
            c = (cos[sr] if cos.shape[0] > 1 else cos.expand(n, -1, -1)).float()
            s = (sin[sr] if sin.shape[0] > 1 else sin.expand(n, -1, -1)).float()
            cq, sq = c[torch.arange(n), p][:, None], s[torch.arange(n), p][:, None]
            q = q * cq + rotate_half(q) * sq
            k = k * c[:, :, None] + rotate_half(k) * s[:, :, None]
            k = k.repeat_interleave(H // KV, dim=2)  # = transformers' repeat_kv
            v = v.repeat_interleave(H // KV, dim=2)
            sc = torch.einsum("nhd,nthd->nht", q, k) * mod.scaling
            t = torch.arange(T, device=h.device)
            if am is None:
                allowed = (t[None] <= p[:, None])[:, None]
            else:
                row = am[sr][torch.arange(n), :, p]  # [n, 1 or H, T]
                allowed = row if row.dtype == torch.bool else row > torch.finfo(row.dtype).min / 2
            alpha = sc.masked_fill(~allowed, float("-inf")).softmax(-1)  # [n, H, T]
            Wo = mod.o_proj.weight.float()  # [d, H*dh]
            Wa = (self.A @ Wo).view(n, H, dh)
            contrib = alpha * torch.einsum("nthd,nhd->nht", v, Wa)
            out = torch.zeros(n, H, len(STATS), device=h.device)
            for j, g in enumerate(SRC_GROUPS):
                gm = self.gmask[g][:, None, :T]
                out[..., j] = (alpha * gm).sum(-1)
                out[..., len(SRC_GROUPS) + j] = (contrib * gm).sum(-1)
            out[..., -1] = contrib.sum(-1)
            self.buf[b] = out
            if self.check:
                self.z[b] = torch.einsum("nht,nthd->nhd", alpha, v)
            return args, kwargs
        return hook

    def _o_pre(self, b):
        def hook(mod, args):
            x = args[0][self.sr, self.p].float().view(self.z[b].shape)
            err = float((x - self.z[b]).norm() / x.norm().clamp_min(1e-8))
            self.max_err = max(self.max_err, err)
        return hook

    def _norm_pre(self, mod, args):
        x = args[0][self.sr, self.p].float()
        self.rms = (x.pow(2).mean(-1) + mod.variance_epsilon).sqrt()

    def __enter__(self):
        self.handles = [self.lm.model.model.norm.register_forward_pre_hook(self._norm_pre)]
        for b, layer in enumerate(self.lm.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._attn_pre(b), with_kwargs=True))
            if self.check:
                self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._o_pre(b)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()

    def result(self):
        """[n, n_layers, H, len(STATS)] float32, dla_* divided by the final-residual RMS."""
        out = torch.stack([self.buf[b] for b in range(len(self.buf))], dim=1)
        nd = len(SRC_GROUPS)
        out[..., nd:] = out[..., nd:] / self.rms[:, None, None, None]
        return out.cpu().numpy()


def pair_seqs(items, idx):
    return [Seq(items[i].prefix, c) for i in idx for c in (items[i].c_plus, items[i].c_minus)]


@torch.no_grad()
def record_condition(lm, items, bs, tag_ko=False, log=print):
    A_all, same = answer_dirs(lm, items)
    stats, parts = [], {name: [] for name in PARTS}
    for s in range(0, len(items), bs):
        idx = list(range(s, min(s + bs, len(items))))
        seqs = pair_seqs(items, idx)
        T = max(len(x.prefix) + len(x.cand) for x in seqs)
        with ExitStack() as stack:
            if tag_ko:  # registered first, so the recorder sees the modified mask
                _, pad = lm._collate(seqs)
                stack.enter_context(knockout(lm, list(range(lm.n_layers)),
                                             blocked(items, idx, "pre", "answer", T, lm.device), pad))
            rec = stack.enter_context(HeadRecorder(lm, items, idx, A_all[idx], check=(s == 0)))
            res = pair_scores(lm.token_logprobs(seqs))
        if s == 0:
            log(f"  recomputed vs. actual attention output: max relative error {rec.max_err:.4f}")
            if rec.max_err > MAX_REL_ERR:
                raise RuntimeError(f"head decomposition does not match the model (rel. err {rec.max_err:.3f})")
        stats.append(rec.result())
        for name, v in zip(PARTS, res):
            parts[name].append(v.float().cpu().numpy())
    stats = np.concatenate(stats)
    stats[same, ..., len(SRC_GROUPS):] = np.nan
    return stats, {k: np.concatenate(v) for k, v in parts.items()}, same


def head_mask(items, idx, heads, n_heads, T, device):
    """[2B, H, T, T] bool: the listed heads' last-prompt-token query cannot attend to the answer."""
    m = torch.zeros(2 * len(idx), n_heads, T, T, dtype=torch.bool)
    for r, i in enumerate(idx):
        it = items[i]
        p = len(it.prefix) - 1
        for j in (0, 1):
            for h in heads:
                m[2 * r + j, h, p, it.pos["answer"]] = True
    return m.to(device)


def select_heads(stats, rows, k):
    """Top-k heads by mean dla_answer on even rows; returns [(block, head), ...]."""
    even = np.asarray(rows) % 2 == 0
    score = np.nanmean(stats[even][..., STATS.index("dla_answer")], axis=0)  # [L, H]
    order = np.argsort(np.nan_to_num(score, nan=-np.inf), axis=None)[::-1][:k]
    return [tuple(int(x) for x in np.unravel_index(o, score.shape)) for o in order]


def random_heads(k, n_layers, n_heads, rng):
    pool = [(b, h) for b in range(n_layers // 2, n_layers) for h in range(n_heads)]
    return [pool[i] for i in sorted(rng.choice(len(pool), k, replace=False))]


@torch.no_grad()
def head_knockout(lm, items, rows, head_sets, bs):
    """Per head set: margins on the odd rows with the set's heads blocked from the answer."""
    odd = [j for j, r in enumerate(rows) if r % 2 == 1]
    H = lm.model.config.num_attention_heads
    res = {"rows": [rows[j] for j in odd], "biased": {n: [] for n in PARTS},
           **{name: {n: [] for n in PARTS} for name in head_sets}}
    for s in range(0, len(odd), bs):
        idx = odd[s:s + bs]
        seqs = pair_seqs(items, idx)
        T = max(len(x.prefix) + len(x.cand) for x in seqs)
        for n, v in zip(PARTS, scores(lm, seqs)):
            res["biased"][n] += v.float().cpu().tolist()
        for name, heads in head_sets.items():
            by_block = {}
            for b, h in heads:
                by_block.setdefault(b, []).append(h)
            masks = {b: head_mask(items, idx, hs, H, T, lm.device) for b, hs in by_block.items()}
            for n, v in zip(PARTS, scores(lm, seqs, sorted(masks), masks)):
                res[name][n] += v.float().cpu().tolist()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="llama")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--conditions", nargs="+",
                    default=["neutral", "assert_plausible", "mention_plausible_1", "assert_irrelevant",
                             "assert_plausible+tag", "mention_plausible_1+tag",
                             "assert_end_plausible", "assert_alt_plausible"])
    ap.add_argument("--select-condition", default="assert_plausible")
    ap.add_argument("--mention-condition", default="mention_plausible_1")
    ap.add_argument("--ko-conditions", nargs="+",
                    default=["assert_plausible", "mention_plausible_1", "assert_end_plausible", "assert_alt_plausible"])
    ap.add_argument("--k", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--limit", type=int, default=None, help="first N neutral-correct items only")
    ap.add_argument("--bs", type=int, default=16, help="items per batch (2 sequences each)")
    args = ap.parse_args()

    root = Path(args.results_root) / args.model
    beh = root / "behavior"
    meta = json.loads((beh / "meta.json").read_text())
    items_all = read_jsonl(beh / "items.jsonl")
    nc = [items_all[i] for i in meta["nc_ids"]][:args.limit]
    out_dir = root / "heads"
    out_dir.mkdir(parents=True, exist_ok=True)
    lm = LM.load(MODELS[args.model])

    for cond in args.conditions:
        f = out_dir / f"{cond}.npz"
        if f.exists():
            continue
        base, _, tag = cond.partition("+")
        assert base in CONDITIONS and tag in ("", "tag"), cond
        items, rows = items_for(lm, nc, base)
        print(f"{args.model} {cond}: {len(items)} items")
        stats, parts, same = record_condition(lm, items, args.bs, tag_ko=bool(tag))
        np.savez_compressed(f, stats=stats.astype(np.float32), rows=np.array(rows), same_first=same,
                            stats_names=np.array(STATS),
                            m_neutral_first=np.array([nc[r]["margin_first"]["neutral"] for r in rows]),
                            **{f"m_{k}": v for k, v in parts.items()})
        dla = np.nanmean(stats[..., STATS.index("dla_answer")], axis=0).sum(-1)  # [L]
        print("  sum over heads of dla_answer by block: " + " ".join(f"{x:+.2f}" for x in dla))

    f = out_dir / "knockout.json"
    if f.exists():
        return
    rng = np.random.default_rng(0)
    L, H = lm.n_layers, lm.model.config.num_attention_heads
    sel = np.load(out_dir / f"{args.select_condition}.npz")
    men = np.load(out_dir / f"{args.mention_condition}.npz")
    head_sets = {}
    for k in args.k:
        head_sets[f"top_assert_k{k}"] = select_heads(sel["stats"], sel["rows"], k)
        head_sets[f"top_mention_k{k}"] = select_heads(men["stats"], men["rows"], k)
        head_sets[f"random_k{k}"] = random_heads(k, L, H, rng)
    print("heads:", json.dumps(head_sets))
    out = {"model": args.model, "heads": head_sets, "conditions": {}}
    for cond in args.ko_conditions:
        items, rows = assert_items(lm, nc, cond)
        res = head_knockout(lm, items, rows, head_sets, args.bs)
        res["m_neutral_first"] = [nc[r]["margin_first"]["neutral"] for r in res["rows"]]
        out["conditions"][cond] = res
        mn, mb = np.array(res["m_neutral_first"]), np.array(res["biased"]["first"])
        print(f"{cond}: " + "  ".join(
            f"{name}={np.mean(np.array(res[name]['first']) - mb) / np.mean(mn - mb):+.2f}" for name in head_sets))
    write_json(f, out)


if __name__ == "__main__":
    main()
