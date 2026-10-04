"""
Dataset loading (all deduplicated questions of SycophancyEval's `answer` split: 1813, of
which 996 TriviaQA and 817 TruthfulQA), irrelevant-answer pairing, and per-seed splits.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from v2.config import DATASET_FILE, DATASET_REPO, SPLIT_FRACS


def load_raw(path: str | None = None) -> list[dict]:
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dedupe(raw: list[dict]) -> list[dict]:
    """One item per question. The raw file repeats each question once per bias template;
    we rebuild all prompts ourselves from `base`, so which variant a row holds is irrelevant."""
    seen, items = set(), []
    for row in raw:
        b = row.get("base", {})
        q, cp, cm = b.get("question"), b.get("correct_answer"), b.get("incorrect_answer")
        if not (q and cp and cm) or q in seen:
            continue
        seen.add(q)
        items.append({"id": len(items), "dataset": b.get("dataset"), "question": q,
                      "c_plus": cp, "c_minus": cm, "gold": b.get("answer", [])})
    return items


def _clashes(answer: str, item: dict) -> bool:
    a = answer.strip().lower()
    others = [item["c_plus"], item["c_minus"], *item["gold"]]
    return any(a == o.strip().lower() or a in o.lower() or o.lower() in a
               for o in others if o and o.strip())


def pair_irrelevant(items: list[dict], seed: int = 0) -> None:
    """Assigns item['r'] = c_plus of another, randomly chosen item, avoiding answers that
    overlap (case-insensitive substring) with this item's own answers. Fixed seed so that
    every run and every model sees the same pairing."""
    rng = np.random.default_rng(seed)
    n = len(items)
    for i, it in enumerate(items):
        for j in rng.permutation(n):
            if j != i and not _clashes(items[j]["c_plus"], it):
                it["r"] = items[j]["c_plus"]
                it["r_source_id"] = int(j)
                break


def load_items(path: str | None = None, limit: int | None = None) -> list[dict]:
    items = dedupe(load_raw(path))
    pair_irrelevant(items)
    return items[:limit] if limit else items


def split(n: int, seed: int) -> dict[str, np.ndarray]:
    """Indices into the neutral-correct item list (and its activation cache)."""
    perm = np.random.default_rng(1000 + seed).permutation(n)
    n_tr = int(round(SPLIT_FRACS[0] * n))
    n_va = int(round(SPLIT_FRACS[1] * n))
    return {"train": np.sort(perm[:n_tr]), "val": np.sort(perm[n_tr:n_tr + n_va]),
            "test": np.sort(perm[n_tr + n_va:])}


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj) -> None:
    """Atomic write (tmp file + rename): concurrent array tasks and timeouts never leave a
    half-written file that a later run would mistake for a finished result."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def rest_margin(lp_plus: list[float], lp_minus: list[float]) -> float:
    """Mean log-prob margin over answer tokens 2..k (NaN if either answer is one token)."""
    if len(lp_plus) < 2 or len(lp_minus) < 2:
        return float("nan")
    return float(np.mean(lp_plus[1:]) - np.mean(lp_minus[1:]))


def items_fingerprint(items: list[dict]) -> str:
    return hashlib.sha256("\n".join(it["question"] for it in items).encode()).hexdigest()[:16]
