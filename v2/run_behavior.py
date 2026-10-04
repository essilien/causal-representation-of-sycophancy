"""
E1 + entrainment ablation (behavior) + activation cache for all later stages.

For every deduplicated question and every prompt condition in v2.config.CONDITIONS, scores
the candidate answers by teacher-forced log-probability (per token, so mean / sum /
first-token margins can all be derived later). Items the model gets right under the
neutral prompt (m_neutral > 0) are the population for probing and interventions; for those
it records the output of every decoder block at the last prompt token under every
condition -> cache/<condition>.npy  [n_nc, n_layers, d_model] float16.

Outputs in <results-root>/<model>/behavior/:
  items.jsonl   per item: texts, per-condition per-candidate token log-probs, margins
  meta.json     model, layer count, neutral-correct item ids (= cache row order)
  cache/*.npy   activation cache (~0.3 GB per condition for Llama-8B)

Usage:
    python -m v2.run_behavior --model llama --results-root $SYCO_RESULTS
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from v2.config import CONDITIONS, MODELS, build_prompt, candidates_for
from v2.data import load_items, write_jsonl
from v2.lm import LM, Seq


def score_all(lm: LM, items, bs: int, log=print):
    jobs = []  # (item index, condition, candidate key, Seq)
    for i, it in enumerate(items):
        for cond in CONDITIONS:
            prefix = lm.encode_prompt(build_prompt(cond, it))
            for key in candidates_for(cond):
                jobs.append((i, cond, key, Seq(prefix, lm.encode_answer(it[key]))))
    jobs.sort(key=lambda j: len(j[3].prefix) + len(j[3].cand))  # less padding
    log(f"Scoring {len(jobs)} sequences ...")
    t0 = time.time()
    for s in range(0, len(jobs), bs):
        chunk = jobs[s:s + bs]
        with torch.no_grad():
            lps = lm.token_logprobs([j[3] for j in chunk])
        for (i, cond, key, _), lp in zip(chunk, lps):
            items[i].setdefault("lp", {}).setdefault(cond, {})[key] = [round(float(x), 5) for x in lp]
        if (s // bs) % 100 == 0:
            log(f"  {s}/{len(jobs)} ({time.time() - t0:.0f}s)")
    for it in items:
        it["margin"], it["margin_first"], it["margin_sum"] = {}, {}, {}
        for cond, lp in it["lp"].items():
            cp, cm = lp["c_plus"], lp["c_minus"]
            it["margin"][cond] = float(np.mean(cp) - np.mean(cm))
            it["margin_first"][cond] = cp[0] - cm[0]
            it["margin_sum"][cond] = float(np.sum(cp) - np.sum(cm))
            if "r" in lp:  # v1-style control: correct answer vs. the asserted irrelevant answer
                it["margin"][cond + "__vs_r"] = float(np.mean(cp) - np.mean(lp["r"]))


def record_cache(lm: LM, items, nc_ids, cache_dir: Path, bs: int, log=print):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for cond in CONDITIONS:
        path = cache_dir / f"{cond}.npy"
        if path.exists():
            log(f"  cache {cond}: exists, skipping")
            continue
        tmp = cache_dir / f"{cond}.tmp.npy"
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                        shape=(len(nc_ids), lm.n_layers, lm.d_model))
        prefixes = [lm.encode_prompt(build_prompt(cond, items[i])) for i in nc_ids]
        for s in range(0, len(prefixes), bs):
            with torch.no_grad():
                arr[s:s + bs] = lm.record_last_prompt(prefixes[s:s + bs]).numpy()
        arr.flush()
        del arr
        tmp.rename(path)  # only complete caches get the final name
        log(f"  cache {cond}: done")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="llama")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--dataset-path", default=None, help="local answer.jsonl (default: HF Hub)")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (smoke tests)")
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()

    out = Path(args.results_root) / args.model / "behavior"
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(args.dataset_path, args.limit)
    print(f"{len(items)} questions")
    lm = LM.load(MODELS[args.model])
    print(f"Loaded {MODELS[args.model]}: {lm.n_layers} blocks, d={lm.d_model}")

    if (out / "items.jsonl").exists() and (out / "meta.json").exists():
        from v2.data import read_jsonl
        items = read_jsonl(out / "items.jsonl")
        print("items.jsonl exists, reusing scores")
    else:
        score_all(lm, items, args.bs)
    nc_ids = [it["id"] for it in items if it["margin"]["neutral"] > 0]
    for row, i in enumerate(nc_ids):
        items[i]["nc_row"] = row
    write_jsonl(out / "items.jsonl", items)
    meta = {"model": args.model, "hf_id": MODELS[args.model], "n_layers": lm.n_layers,
            "d_model": lm.d_model, "n_items": len(items), "nc_ids": nc_ids,
            "conditions": list(CONDITIONS)}
    (out / "meta.json").write_text(json.dumps(meta))

    print(f"\nNeutral-correct: {len(nc_ids)}/{len(items)} = {len(nc_ids) / len(items):.1%}")
    for cond in CONDITIONS:
        if cond == "neutral":
            continue
        flips = np.mean([items[i]["margin"][cond] < 0 for i in nc_ids])
        shift = np.mean([items[i]["margin"][cond] - items[i]["margin"]["neutral"] for i in nc_ids])
        print(f"  {cond:24s} flip rate {flips:6.1%}   mean margin shift {shift:+.3f}")

    print("\nRecording activation cache ...")
    record_cache(lm, items, nc_ids, out / "cache", bs=max(8, args.bs // 2))
    print("Done.")


if __name__ == "__main__":
    main()
