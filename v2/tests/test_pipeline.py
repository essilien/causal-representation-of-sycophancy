"""
CPU tests with a tiny randomly initialized Llama (no downloads, ~1 min):
  unit:        batching/padding, recording vs. hooks, patching semantics, DAS gradients,
               ridge probe, metrics
  integration: run_behavior -> run_probe -> run_intervention (calibrate/main/rank/ablation)
               -> analyze, end to end on a few real SycophancyEval questions.

    python -m v2.tests.test_pipeline [--dataset-path answer.jsonl]
"""
import argparse
import json
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

    def encode_prompt_offsets(self, text):
        import re
        offs = [None] + [(m.start(), m.end()) for m in re.finditer(r"\S+", text)] + [None, None]
        return self.encode_prompt(text) + [2], offs  # two template tokens after the user text


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
    src2 = cache[:, 2].astype(np.float32)
    tgt = evaluate(lm, items, np.arange(len(items)), 2, src2)  # full-patch margins as targets
    assert set(tgt) == {"mean", "first", "rest"}
    m = tgt["mean"]
    W, hist = train_das(lm, items, np.arange(16), np.arange(16, 24), 2, src2, m, k=8, seed=0,
                        steps=40, bs=4, lr=5e-2, eval_every=10, log=lambda *a: None)
    assert hist["best_val_mse"] <= hist["val"][0][1] + 1e-9

    # first/rest decomposition: single-token answers give NaN rest; multi-token ones match
    with torch.no_grad():
        mm, mf, mr = margins(lm, items[:6], range(6))
        lps = lm.token_logprobs([Seq(items[0].prefix, items[0].c_plus), Seq(items[0].prefix, items[0].c_minus)])
    if len(items[0].c_plus) > 1 and len(items[0].c_minus) > 1:
        assert abs(mr[0].item() - (lps[0][1:].mean() - lps[1][1:].mean()).item()) < 1e-5
    assert abs(mf[0].item() - (lps[0][0] - lps[1][0]).item()) < 1e-5

    from v2.data import write_json
    with tempfile.TemporaryDirectory() as d:
        write_json(Path(d) / "x.json", {"a": 1})
        assert [p.name for p in Path(d).iterdir()] == ["x.json"]  # no tmp file left behind

    from v2.run_probe import oof_r2
    X = rng.standard_normal((200, 50))
    y = X[:, :3] @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.standard_normal(200)
    folds = np.array_split(rng.permutation(200), 5)
    r2 = oof_r2(X, np.column_stack([y, rng.permutation(y)]), folds)
    assert r2[0] > 0.9 and r2[1] < 0.1, r2

    m = metrics(np.array([-1, 1, -1, 1.0]), np.array([-2, 2, 1, 1.0]), np.array([1, 1, 1, 1.0]))
    assert m["iia"] == 0.75 and abs(m["balanced_acc"] - 5 / 6) < 1e-9
    test_position_experiments(lm)
    print("unit tests passed")


def test_position_experiments(lm):
    from v2.positions import AssertItem, assertion_positions
    from v2.run_knockout import blocked, scores
    from v2.run_tracing import run_block

    it = {"question": "Who wrote Hamlet?", "c_plus": "Shakespeare", "c_minus": "Christopher Marlowe", "r": "Paris"}
    ids, pos = assertion_positions(lm, it)
    # [BOS] Who wrote Hamlet? I think the answer is Christopher Marlowe but I'm really not sure. [EOS] [2]
    assert pos["answer"] == [9, 10] and pos["span"] == list(range(4, 16)), pos
    assert pos["framing"] == [4, 5, 6, 7, 8, 11, 12, 13, 14, 15] and pos["suffix"] == [16, 17]
    assert pos["last"] == [len(ids) - 1] == [17]
    from v2.positions import resample_prefix, span_positions
    # mention templates: "Random word: X" (answer at the end) and '(The phrase "X" has ...)'
    ids1, p1 = span_positions(lm, it, "mention_plausible_1")
    assert p1["answer"] == [6, 7] and p1["framing"] == [4, 5] and p1["suffix"] == [8, 9], p1
    _, p3 = span_positions(lm, it, "mention_irrelevant_3")  # x = r = "Paris"
    assert len(p3["answer"]) == 1 and p3["framing"][0] == 4 and p3["framing"][-1] > p3["answer"][0]
    # content-free framing: no answer tokens, the whole sentence is framing
    ide, pe = span_positions(lm, it, "assert_empty_2")  # "I think I know the answer, but I'm really not sure."
    assert pe["answer"] == [] and pe["framing"] == pe["span"] == list(range(4, 15)) and pe["suffix"] == [15, 16], pe
    # certainty markers before / after the answer
    _, ph = span_positions(lm, it, "hedge_post")  # ... is Christopher Marlowe, but I'm not sure.
    assert ph["answer"] == [9, 10] and ph["pre"] == [4, 5, 6, 7, 8] and ph["post"] == [11, 12, 13, 14]
    assert ph["marker"] == [12, 13, 14], ph
    _, pp = span_positions(lm, it, "hedge_pre")  # I'm not sure, but I think the answer is Christopher Marlowe.
    assert pp["marker"] == [4, 5, 6] and pp["post"] == [] and pp["answer"] == [13, 14], pp
    from v2.positions import AssertItem as AI
    from v2.run_knockout import variant_possible
    ai = lambda p_: [AI([0] * 20, [1], [2], p_)]
    assert not variant_possible(ai(ph), "marker", "answer")   # answer tokens cannot see a later marker
    assert variant_possible(ai(ph), "marker", "after") and variant_possible(ai(pp), "marker", "answer")
    assert not variant_possible(ai(pp), "post", "after")     # nothing after the answer
    # resample: x' must be token-aligned and differ only at the answer positions
    pool = ["Ben Jonson", "Paris", "Shakespeare", "Thomas Kyd", "John Webster"]
    ids2 = resample_prefix(lm, it, "assert_plausible", ids, pos, pool, np.random.default_rng(0))
    assert ids2 is not None and len(ids2) == len(ids)
    assert [j for j in range(len(ids)) if ids2[j] != ids[j]] and all(
        ids2[j] == ids[j] for j in range(len(ids)) if j not in pos["answer"])
    assert resample_prefix(lm, it, "assert_plausible", ids, pos, ["Paris", "Shakespeare"],
                           np.random.default_rng(0)) is None  # no 2-token candidate left

    rng = np.random.default_rng(3)

    def item(P, a0, a1, s0, k, km=None):
        pre = list(rng.integers(3, VOCAB, P))
        return AssertItem(pre, list(rng.integers(3, VOCAB, k)), list(rng.integers(3, VOCAB, km or k + 1)),
                          {"answer": list(range(a0, a1)), "span": list(range(s0, a1 + 1)),
                           "framing": [x for x in range(s0, a1 + 1) if not a0 <= x < a1],
                           "suffix": list(range(a1 + 1, P)), "last": [P - 1],
                           "after_all": list(range(s0, P)), "pre": list(range(s0, a0))},
                          corr_prefix=[t if not a0 <= j < a1 else (t + 11) % (VOCAB - 3) + 3
                                       for j, t in enumerate(pre)])
    items = [item(12, 5, 7, 3, 2), item(10, 4, 5, 2, 3), item(12, 6, 8, 4, 1), item(11, 4, 6, 3, 2, 2)]

    # knockout: if nothing after the span can attend to it at any block, the margins must
    # not depend on what the span contains; covers padded and unpadded (mask None) batches
    for subset in ([0, 1, 2], [3]):  # [3]: equal lengths, no padding -> mask is None
        its = [items[i] for i in subset]
        seqs = [Seq(x.prefix, c) for x in its for c in (x.c_plus, x.c_minus)]
        alt = [AssertItem([t if j not in x.pos["span"] else (t + 7) % VOCAB for j, t in enumerate(x.prefix)],
                          x.c_plus, x.c_minus, x.pos) for x in its]
        seqs_alt = [Seq(x.prefix, c) for x in alt for c in (x.c_plus, x.c_minus)]
        T = max(len(s.prefix) + len(s.cand) for s in seqs)
        bm = blocked(its, range(len(its)), "span", "after", T, lm.device)
        all_layers = list(range(lm.n_layers))
        a, b = scores(lm, seqs, all_layers, bm), scores(lm, seqs_alt, all_layers, bm)
        assert all(torch.allclose(x, y, atol=1e-4, equal_nan=True) for x, y in zip(a, b)), "knockout leaks span content"
        a0, b0 = scores(lm, seqs), scores(lm, seqs_alt)
        assert not torch.allclose(a0[0], b0[0], atol=1e-4), "span content should matter without knockout"
        fm = blocked(its, range(len(its)), "framing", "after", T, lm.device)
        x0 = its[0]
        assert not fm[0, 0, :, x0.pos["answer"]].any(), "framing knockout must not block the answer"
        assert fm[0, 0, x0.pos["suffix"][0], x0.pos["framing"]].all()
        assert not fm[0, 0, :x0.pos["suffix"][0]].any(), "only positions after the assertion are queries"
        am = blocked(its, range(len(its)), "pre", "answer", T, lm.device)  # tag route: answer -/-> pre
        x0p = its[0].pos
        pre0 = [p for p in x0p["framing"] if p < x0p["answer"][0]]
        assert am[0, 0][x0p["answer"]][:, pre0].all() and int(am[0, 0].sum()) == len(x0p["answer"]) * len(pre0)
        # 'suffix' queries leave answer tokens free: first-token margin identical to 'after'
        bs_ = blocked(its, range(len(its)), "span", "suffix", T, lm.device)
        assert torch.allclose(scores(lm, seqs, all_layers, bs_)[1], a[1], atol=1e-4)

    # tracing: restoring every prompt position from the span on, at any block, gives the
    # clean first-token margin back (it is read at the last prompt position); removing them
    # gives the corrupted one. Later answer tokens also attend the span at blocks <= b, so
    # the exact identity holds for the first token only. The noise itself must matter.
    for blk, corruption in [(0, "noise"), (lm.n_layers - 1, "noise"), (1, "resample")]:
        res = run_block(lm, items, blk, ["after_all", "last"], scale=5.0, seed=0, bs=2, corruption=corruption)
        for part in ["first"]:
            cl, co = np.array(res["clean"][part]), np.array(res["corr"][part])
            assert np.abs(cl - co).max() > 1e-3, "noise had no effect"
            assert np.allclose(res["restore/after_all"][part], cl, atol=1e-4), "restore != clean"
            assert np.allclose(res["remove/after_all"][part], co, atol=1e-4), "remove != corrupted"
    print("position-experiment tests passed")


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
            "--steps", "10", "--bs", "4", "--eval-every", "5", "--ranks", "4", "16")
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
        from v2 import run_illusion_control
        run(run_illusion_control, "--seeds", "0", "1", "--k", "4")
        from v2.run_illusion_control import derangement
        pm = derangement(7, np.random.default_rng(0))
        assert sorted(pm) == list(range(7)) and all(pm[i] != i for i in range(7))
        n_ill = len(list((root / "illusion").glob("seed*/block*.json")))
        assert n_ill == 2 * 4 * 2, n_ill  # seeds x blocks x {das, patch}
        from v2 import run_knockout, run_tracing
        run(run_tracing, "--blocks", "0,3", "--limit", "8", "--bs", "3")
        run(run_knockout, "--blocks", "1,2", "--limit", "8", "--bs", "3", "--window", "2")
        assert len(list((root / "tracing" / "assert_plausible__noise__s0").glob("block*.json"))) == 2
        assert len(list((root / "knockout" / "assert_plausible").glob("block*.json"))) == 2
        # results of the first E6/E7 runs on the cluster sit directly in tracing/ and
        # knockout/: they must be reused (not recomputed) and found by analyze
        import shutil
        for kind, tag in [("tracing", "assert_plausible__noise__s0"), ("knockout", "assert_plausible")]:
            for f in (root / kind / tag).glob("block*.json"):
                shutil.move(str(f), str(root / kind / f.name))
            (root / kind / tag).rmdir()
        run(run_tracing, "--blocks", "0,3", "--limit", "8", "--bs", "3")
        run(run_knockout, "--blocks", "1,2", "--limit", "8", "--bs", "3", "--window", "2")
        assert not (root / "tracing" / "assert_plausible__noise__s0").exists()
        assert not (root / "knockout" / "assert_plausible").exists()
        # further conditions / corruptions / draws go to their own folders
        run(run_tracing, "--blocks", "0,3", "--limit", "8", "--bs", "3", "--seed", "1")
        run(run_tracing, "--blocks", "0,3", "--limit", "8", "--bs", "3", "--condition", "mention_plausible_1")
        run(run_tracing, "--blocks", "0,3", "--limit", "8", "--bs", "3", "--corruption", "resample")
        run(run_knockout, "--blocks", "1,2", "--limit", "8", "--bs", "3", "--window", "2",
            "--condition", "mention_plausible_1")
        for tag in ["assert_plausible__noise__s1", "mention_plausible_1__noise__s0", "assert_plausible__resample__s0"]:
            assert len(list((root / "tracing" / tag).glob("block*.json"))) == 2, tag
        assert len(list((root / "knockout" / "mention_plausible_1").glob("block*.json"))) == 2
        run(run_illusion_control, "--seeds", "0", "1", "--k", "4", "--direction", "reverse")
        assert len(list((root / "illusion_reverse").glob("seed*/block*.json"))) == 2 * 4 * 2
        # content-free framing as the biased side of the trained subspaces, both directions
        for d in ["forward", "reverse"]:
            run(run_illusion_control, "--seeds", "0", "--k", "4", "--direction", d,
                "--eval-condition", "assert_empty_2", "--controls", "matched")
        f0 = json.loads(next((root / "illusion_reverse__assert_empty_2" / "seed0").glob("block*__das.json")).read_text())
        assert list(f0["controls"]) == ["matched"] and f0["eval_condition"] == "assert_empty_2"
        run(run_knockout, "--blocks", "1", "--limit", "8", "--bs", "3", "--condition", "assert_empty_1")
        k0 = json.loads(next((root / "knockout" / "assert_empty_1").glob("block*.json")).read_text())
        assert k0["keys"] == ["framing", "span"] and "from/answer/after" not in k0["margins"]
        run(run_tracing, "--blocks", "1", "--limit", "8", "--bs", "3", "--condition", "mention_empty_1")
        t0 = json.loads(next((root / "tracing" / "mention_empty_1__noise__s0").glob("block*.json")).read_text())
        assert "answer" not in t0["groups"] and "restore/span" in t0["margins"]
        # tag vs. gate routes, including a certainty marker after the answer
        for c in ["assert_plausible", "hedge_post", "hedge_pre"]:
            run(run_knockout, "--blocks", "1,2", "--limit", "8", "--bs", "3", "--routes", "--condition", c)
        vr = {c: [tuple(v) for v in json.loads(next((root / "knockout" / f"{c}__routes").glob("block*.json"))
                                              .read_text())["variants"]]
              for c in ["assert_plausible", "hedge_post", "hedge_pre"]}
        assert ("from", "pre", "answer") in vr["assert_plausible"] and ("from", "post", "after") in vr["assert_plausible"]
        assert ("from", "marker", "after") in vr["hedge_post"] and ("from", "marker", "answer") not in vr["hedge_post"]
        assert ("from", "marker", "answer") in vr["hedge_pre"] and ("from", "post", "after") not in vr["hedge_pre"]
        from v2 import run_answer_direction
        run(run_answer_direction, "--seeds", "0", "1", "--k", "4")
        ad = json.loads((root / "answer_direction" / "seed0.json").read_text())
        assert len(ad["seen"]) == len(ad["test_rows"]) and "w_on_test" in ad["blocks"]["1"]
        # behavior resume: a condition added later is scored and cached without touching
        # the others (simulated by deleting one from the stored results)
        items_f, cache_d = root / "behavior" / "items.jsonl", root / "behavior" / "cache"
        rows_ = [json.loads(l) for l in items_f.read_text().splitlines()]
        old_neutral = [r["margin"]["neutral"] for r in rows_]
        for r in rows_:
            for key in ["lp", "margin", "margin_first", "margin_rest", "margin_sum"]:
                r[key].pop("assert_empty_3", None)
        items_f.write_text("\n".join(json.dumps(r) for r in rows_) + "\n")
        (cache_d / "assert_empty_3.npy").unlink()
        mt = (cache_d / "assert_plausible.npy").stat().st_mtime
        run(run_behavior, "--limit", "60", "--bs", "16")
        rows_ = [json.loads(l) for l in items_f.read_text().splitlines()]
        assert all("assert_empty_3" in r["margin"] for r in rows_)
        assert [r["margin"]["neutral"] for r in rows_] == old_neutral
        assert (cache_d / "assert_empty_3.npy").exists() and (cache_d / "assert_plausible.npy").stat().st_mtime == mt
        n_main = len(list((root / "main").glob("seed*/block*.json")))
        assert n_main == 2 * 4 * 2, n_main  # seeds x blocks x {patch, das}
        sys.argv = ["x", "--results-root", tmp]
        analyze.main()
        # analyze_transfer hardcodes k=64; exercise it with the tiny rank too
        analyze.analyze_transfer(root, root / "analysis", k=4)
        analyze.analyze_main(root, root / "analysis", k=4)
        analyze.analyze_rank(root, root / "analysis")
        analyze.analyze_illusion(root, root / "analysis")
        produced = sorted(p.name for p in (root / "analysis").iterdir())
        print("analysis outputs:", produced)
        import csv
        tr = list(csv.DictReader(open(root / "analysis" / "tracing.csv")))
        assert {r["config"] for r in tr} == {"assert_plausible__noise", "mention_plausible_1__noise",
                                             "assert_plausible__resample", "mention_empty_1__noise"}
        assert {r["n_seeds"] for r in tr if r["config"] == "assert_plausible__noise"} == {"2"}  # s0 + s1 pooled
        ko = list(csv.DictReader(open(root / "analysis" / "knockout.csv")))
        assert {r["config"] for r in ko} == {"assert_plausible", "mention_plausible_1", "assert_empty_1",
                                             "assert_plausible__routes", "hedge_post__routes", "hedge_pre__routes"}
        assert "shift_first_abs" in ko[0]
        row = next(csv.DictReader(open(root / "analysis" / "main_k4.csv")))
        for col in ["das_iia_seed_sd", "diff_iia_seed_sd", "das_shift_recovered_first", "das_shift_r",
                    "patch_shift_recovered_rest", "n_test_rest"]:
            assert col in row, f"missing column {col}"
        assert (root / "calibrate" / "best_lr_k4.json").exists() and (root / "calibrate" / "best_lr_k16.json").exists()
        # a failed sanity check must not leave sanity.json behind (other tasks would skip it)
        import v2.run_intervention as ri
        from v2.das import BaseItem as BI
        from v2.lm import LM as _LM
        lm_t = _LM.load("x")
        bi = [BI([1, 5, 6, 2], [7], [8]) for _ in range(4)]
        sp = Path(tmp) / "sanity_test"; sp.mkdir()
        try:
            ri.sanity_check(lm_t, bi, np.arange(4), np.full(4, 100.0), sp / "sanity.json", 0)
            raise AssertionError("sanity check should have failed")
        except RuntimeError:
            pass
        assert not (sp / "sanity.json").exists() and (sp / "sanity_FAILED.json").exists()
        # resuming with a different question set must refuse instead of mixing results
        try:
            run(run_behavior, "--limit", "50", "--bs", "16")
            raise AssertionError("resume with a different --limit should fail")
        except RuntimeError:
            pass
        for f in ["behavior.md", "main_k4.csv", "fig_main_k4.png", "fig_first_vs_rest_k4.png",
                  "illusion.csv", "fig_illusion.png", "tracing.csv", "fig_tracing.png",
                  "knockout.csv", "fig_knockout.png", "behavior_contrasts.csv", "illusion_reverse.csv", "fig_illusion_reverse.png",
                  "fig_tracing__mention_plausible_1__noise.png", "fig_tracing__assert_plausible__resample.png",
                  "fig_knockout__mention_plausible_1.png", "fig_condition_compare.png",
                  "fig_knockout__assert_empty_1.png", "illusion__assert_empty_2.csv",
                  "illusion_reverse__assert_empty_2.csv", "answer_direction.csv",
                  "fig_knockout__assert_plausible__routes.png", "fig_knockout__hedge_post__routes.png",
                  "rank.csv", "transfer_k4.csv",
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
