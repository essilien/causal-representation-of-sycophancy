"""
CPU tests with a tiny randomly initialized Llama (no downloads, ~1 min):
  unit:        batching/padding, recording vs. hooks, patching semantics, DAS gradients,
               ridge probe, metrics
  integration: run_behavior -> run_probe -> run_intervention (calibrate/main/rank/ablation)
               -> analyze, end to end on a few real SycophancyEval questions.

    python -m v2.tests.test_pipeline [--dataset-path answer.jsonl]
"""
import argparse
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import torch

import v2.lm as lm_mod
from v2.das import BaseItem, Subspace, evaluate, margins, metrics, train_das
from v2.lm import LM, Seq, patch_fn

VOCAB = 97


def tiny_lm(seed=0) -> LM:
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, intermediate_size=64, max_position_embeddings=512)
    model = LlamaForCausalLM(cfg).eval().requires_grad_(False)
    return FakeTokLM(model)


class FakeTokLM(LM):
    """Deterministic word-hash 'tokenizer' so the pipeline runs without a real one."""
    def encode_prompt(self, text):
        return [1] + [zlib.crc32(w.encode()) % (VOCAB - 3) + 3 for w in text.split()] + [2]

    def encode_answer(self, text):
        return [zlib.crc32(w.encode()) % (VOCAB - 3) + 3 for w in text.split()] or [3]


def rand_seq(rng, P, k):
    return Seq(list(rng.integers(3, VOCAB, P)), list(rng.integers(3, VOCAB, k)))


def test_units():
    lm = tiny_lm()
    rng = np.random.default_rng(0)
    seqs = [rand_seq(rng, int(rng.integers(4, 12)), int(rng.integers(1, 4))) for _ in range(6)]

    with torch.no_grad():
        batched = lm.token_logprobs(seqs)
        single = [lm.token_logprobs([s])[0] for s in seqs]
    assert all(torch.allclose(a, b, atol=1e-4) for a, b in zip(batched, single)), "padding changes scores"

    rec = lm.record_last_prompt([s.prefix for s in seqs])  # [B, L, d]
    captured = {}
    for b in range(lm.n_layers):
        pos = torch.tensor([s.last_prompt_pos for s in seqs])

        def grab(h, b=b, pos=pos):
            captured[b] = h[torch.arange(len(seqs)), pos].detach().clone()
            return h
        with torch.no_grad():
            lm.token_logprobs(seqs, b, grab)
        assert torch.allclose(rec[:, b].float(), captured[b].float(), atol=1e-2), f"record != hook at block {b}"

    pos = torch.tensor([s.last_prompt_pos for s in seqs])
    with torch.no_grad():
        same = lm.token_logprobs(seqs, 2, patch_fn(pos, captured[2]))
    assert all(torch.allclose(a, b, atol=1e-4) for a, b in zip(same, batched)), "self-patch is not a no-op"

    other = torch.randn_like(captured[2])
    Wfull = torch.linalg.qr(torch.randn(lm.d_model, lm.d_model)).Q
    with torch.no_grad():
        full = lm.token_logprobs(seqs, 2, patch_fn(pos, other))
        das_full = lm.token_logprobs(seqs, 2, patch_fn(pos, other, Wfull))
        das_zero_effect = lm.token_logprobs(seqs, 2, patch_fn(pos, captured[2], Wfull[:, :4]))
    assert all(torch.allclose(a, b, atol=1e-3) for a, b in zip(full, das_full)), "DAS k=d != full patch"
    assert all(torch.allclose(a, b, atol=1e-4) for a, b in zip(das_zero_effect, batched)), "DAS self-source not no-op"
    assert not all(torch.allclose(a, b, atol=1e-4) for a, b in zip(full, batched)), "patching had no effect"

    # gradients reach W only, and only through the patched block
    sub = Subspace(lm.d_model, 4, 0)
    lp = lm.token_logprobs(seqs, 1, patch_fn(pos, other, sub()))
    torch.stack([x.mean() for x in lp]).sum().backward()
    assert sub.A.grad is not None and sub.A.grad.abs().sum() > 0
    assert all(p.grad is None for p in lm.model.parameters())
    W = sub().detach()
    assert torch.allclose(W.T @ W, torch.eye(4), atol=1e-5)

    # DAS training runs and does not get worse on val than its init
    items = [BaseItem(s.prefix, s.cand, list(rng.integers(3, VOCAB, 2))) for s in seqs * 4]
    cache = np.random.default_rng(1).standard_normal((len(items), lm.n_layers, lm.d_model)).astype(np.float16)
    with torch.no_grad():
        tgt, _ = evaluate(lm, items, np.arange(len(items)), 2, cache)  # full-patch margins as targets
    W, hist = train_das(lm, items, np.arange(16), np.arange(16, 24), 2, cache, tgt, k=8, seed=0,
                        steps=40, bs=4, lr=5e-2, eval_every=10, log=lambda *a: None)
    assert hist["best_val_mse"] <= hist["val"][0][1] + 1e-9

    from v2.run_probe import oof_r2
    X = rng.standard_normal((200, 50))
    y = X[:, :3] @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.standard_normal(200)
    folds = np.array_split(rng.permutation(200), 5)
    r2 = oof_r2(X, np.column_stack([y, rng.permutation(y)]), folds)
    assert r2[0] > 0.9 and r2[1] < 0.1, r2

    m = metrics(np.array([-1, 1, -1, 1.0]), np.array([-2, 2, 1, 1.0]), np.array([1, 1, 1, 1.0]))
    assert m["iia"] == 0.75 and abs(m["balanced_acc"] - 5 / 6) < 1e-9
    print("unit tests passed")


def test_integration(dataset_path):
    import v2.data
    from v2 import analyze, run_behavior, run_intervention, run_probe

    orig_load = v2.data.load_raw
    v2.data.load_raw = lambda path=None: orig_load(dataset_path)
    LM.load = classmethod(lambda cls, model_id: tiny_lm())
    with tempfile.TemporaryDirectory() as tmp:
        def run(mod, *argv):
            sys.argv = ["x", "--results-root", tmp, *argv]
            mod.main()
        run(run_behavior, "--limit", "60", "--bs", "16")
        root = Path(tmp) / "llama"
        n_nc = len(__import__("json").loads((root / "behavior/meta.json").read_text())["nc_ids"])
        assert n_nc >= 10, f"too few neutral-correct items for the test ({n_nc})"
        run(run_probe, "--n-perm", "5")
        run(run_intervention, "--tag", "calibrate", "--blocks", "1", "--calibrate-lrs", "1e-2", "5e-2",
            "--steps", "10", "--bs", "4", "--eval-every", "5", "--ranks", "4")
        run(run_intervention, "--tag", "main", "--seeds", "0", "1", "--blocks", "all", "--chunk", "0/2",
            "--ranks", "4", "--steps", "10", "--bs", "4", "--eval-every", "5")
        run(run_intervention, "--tag", "main", "--seeds", "0", "1", "--blocks", "all", "--chunk", "1/2",
            "--ranks", "4", "--steps", "10", "--bs", "4", "--eval-every", "5")
        run(run_intervention, "--tag", "rank", "--seeds", "0", "--blocks", "1,3", "--methods", "das",
            "--ranks", "1", "4", "16", "--steps", "10", "--bs", "4", "--eval-every", "5")
        for src in ["assert_irrelevant", "mention_plausible_1"]:
            run(run_intervention, "--tag", "ablation", "--seeds", "0", "1", "--blocks", "all",
                "--methods", "das", "--ranks", "4", "--source", src, "--steps", "10", "--bs", "4",
                "--eval-every", "5")
        n_main = len(list((root / "main").glob("seed*/block*.json")))
        assert n_main == 2 * 4 * 2, n_main  # seeds x blocks x {patch, das}
        sys.argv = ["x", "--results-root", tmp]
        analyze.main()
        # analyze_transfer hardcodes k=64; exercise it with the tiny rank too
        analyze.analyze_transfer(root, root / "analysis", k=4)
        analyze.analyze_main(root, root / "analysis", k=4)
        analyze.analyze_rank(root, root / "analysis")
        produced = sorted(p.name for p in (root / "analysis").iterdir())
        print("analysis outputs:", produced)
        for f in ["behavior.md", "main_k4.csv", "fig_main_k4.png", "rank.csv", "transfer_k4.csv",
                  "subspace_overlap_k4.csv", "fig_probe.png", "table_main_k4.tex"]:
            assert f in produced, f"missing {f}"
    print("integration test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", default=None)
    a = ap.parse_args()
    test_units()
    if a.dataset_path:
        test_integration(a.dataset_path)
