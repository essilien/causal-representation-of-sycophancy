"""
Token positions of the inserted text x inside the tokenized prompt of a condition (see
v2/config.CONDITIONS), for the position-resolved experiments (run_tracing, run_knockout).
For assert_plausible, "{question} I think the answer is {c_minus} but I'm really not sure.":

  answer  : tokens of x (c_minus, or r for the irrelevant-content conditions)
  framing : the rest of the inserted sentence ("I think the answer is", "but I'm ...";
            for mention conditions e.g. "Random word:")
  span    : answer + framing
  suffix  : every prompt token after the span (chat-template tokens up to the assistant
            header), including the last prompt token where all v1/v2 interventions sat
  last    : the last prompt token only
  pre     : framing tokens before the answer (the answer tokens can attend to these)
  post    : framing tokens after the answer (only later positions can attend to these)
  marker  : tokens of the certainty phrase of config.MARKERS (e.g. "I'm not sure"), if any
  tmpl    : suffix without the last token (end-of-turn and assistant-header tokens)

Content-free conditions (config.CONTENT_FREE) insert no answer: `answer` is empty and the
whole inserted sentence counts as `framing`.

`resample` corruption (run_tracing) needs a second prompt in which x is replaced by a
different answer x' that tokenizes to the same number of tokens and leaves every other
token unchanged, so the two runs are position-aligned (symmetric token replacement).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from v2.config import CONDITIONS, MARKERS, build_prompt

GROUPS = ["answer", "framing", "span", "suffix", "last"]
ALL_GROUPS = GROUPS + ["pre", "post", "tmpl", "marker"]


@dataclass
class AssertItem:
    prefix: list[int]          # tokenized prompt of the condition
    c_plus: list[int]
    c_minus: list[int]
    pos: dict[str, list[int]]  # group -> positions in prefix
    corr_prefix: list[int] | None = None  # resample corruption: x replaced by x'


def _x_key(cond: str) -> str:
    content = CONDITIONS[cond][1]
    if content is None:
        raise ValueError(f"condition {cond} inserts no answer text")
    return {"plausible": "c_minus", "irrelevant": "r"}[content]


def span_positions(lm, item: dict, cond: str = "assert_plausible", x: str | None = None):
    """(token ids, group -> positions) for the prompt of `cond`; x overrides the inserted
    answer (used to build the resample corruption)."""
    template = CONDITIONS[cond][0]
    q = item["question"]
    assert template.startswith("{question}")
    if "{x}" in template:
        x = item[_x_key(cond)] if x is None else x
        user = template.format(question=q, x=x)
        pre = template.split("{x}")[0].format(question=q)
        assert user.startswith(pre + x)
        if user != build_prompt(cond, {**item, _x_key(cond): x}):
            raise AssertionError("prompt construction diverged from config.build_prompt")
        a0, x0, x1 = len(q), len(pre), len(pre) + len(x)
    else:  # content-free: no answer tokens
        user = build_prompt(cond, item)
        a0, x0, x1 = len(q), -1, -1
    ids, offs = lm.encode_prompt_offsets(user)
    answer, framing = [], []
    for i, o in enumerate(offs):
        if o is None:
            continue
        s, e = o
        if s < x1 and e > x0:
            answer.append(i)
        elif e > a0:
            framing.append(i)
    span = sorted(answer + framing)
    if not span or (not answer and "{x}" in template) or span != list(range(span[0], span[-1] + 1)):
        raise ValueError(f"could not locate a contiguous span for {q[:60]!r}")
    suffix = list(range(span[-1] + 1, len(ids)))
    if not suffix or any(offs[i] is not None for i in suffix):
        raise ValueError(f"unexpected tokens after the span for {q[:60]!r}")
    pre = [i for i in framing if not answer or i < answer[0]]
    post = [i for i in framing if answer and i > answer[-1]]
    marker = []
    if cond in MARKERS:
        m0 = user.find(MARKERS[cond], a0)
        if m0 < 0:
            raise ValueError(f"marker {MARKERS[cond]!r} not found for {q[:60]!r}")
        m1 = m0 + len(MARKERS[cond])
        marker = [i for i in span if offs[i] is not None and offs[i][0] < m1 and offs[i][1] > m0]
    return ids, {"answer": answer, "framing": framing, "span": span, "suffix": suffix,
                 "last": [len(ids) - 1], "pre": pre, "post": post, "marker": marker,
                 "tmpl": suffix[:-1]}


def assertion_positions(lm, item: dict):
    return span_positions(lm, item, "assert_plausible")


def resample_prefix(lm, item, cond, ids, pos, pool, rng, max_tries=400):
    """Prompt ids with x replaced by another item's answer x' that is token-aligned with
    the original (same length, only the answer positions differ), or None."""
    key = _x_key(cond)
    taken = {item[k].strip().lower() for k in ("c_plus", "c_minus", "r")}
    n_ans = len(pos["answer"])
    keep = [j for j in range(len(ids)) if j not in set(pos["answer"])]
    for j in rng.permutation(len(pool))[:max_tries]:
        x2 = pool[j]
        if x2.strip().lower() in taken or abs(len(lm.encode_answer(x2)) - n_ans) > 1:
            continue
        try:
            ids2, pos2 = span_positions(lm, item, cond, x=x2)
        except ValueError:
            continue
        if len(ids2) == len(ids) and pos2["answer"] == pos["answer"] and all(ids2[k] == ids[k] for k in keep):
            return ids2
    return None


def assert_items(lm, nc: list[dict], cond: str = "assert_plausible", resample_seed: int | None = None):
    """AssertItems for the neutral-correct items whose span could be located (and, with
    resample_seed, a token-aligned x' found), plus their row indices into nc."""
    out, rows, skipped = [], [], 0
    pool = sorted({it[_x_key(cond)] for it in nc}) if resample_seed is not None else []
    for r, it in enumerate(nc):
        try:
            ids, pos = span_positions(lm, it, cond)
        except ValueError as e:
            print(f"  skipping item {r}: {e}")
            skipped += 1
            continue
        corr = None
        if resample_seed is not None:
            corr = resample_prefix(lm, it, cond, ids, pos, pool, np.random.default_rng(10_000 * resample_seed + r))
            if corr is None:
                skipped += 1
                continue
        out.append(AssertItem(ids, lm.encode_answer(it["c_plus"]), lm.encode_answer(it["c_minus"]), pos, corr))
        rows.append(r)
    print(f"  {cond}: {len(out)} items, {skipped} skipped")
    return out, rows


def run_dir(root: Path, kind: str, tag: str, legacy_tag: str) -> Path:
    """Output folder of one run configuration. The first E6/E7 runs (assert_plausible, noise,
    seed 0) wrote straight into <kind>/; they are kept there and read as `legacy_tag`."""
    base = root / kind
    if tag == legacy_tag and any(base.glob("block*.json")):
        return base
    return base / tag


def run_dirs(root: Path, kind: str, legacy_tag: str) -> dict[str, Path]:
    base = root / kind
    out = {}
    if any(base.glob("block*.json")):
        out[legacy_tag] = base
    if base.exists():
        for d in sorted(base.iterdir()):
            if d.is_dir() and any(d.glob("block*.json")):
                out[d.name] = d
    return out
