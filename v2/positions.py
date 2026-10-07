"""
Token positions of the user's assertion inside the tokenized assert_plausible prompt,
"{question} I think the answer is {c_minus} but I'm really not sure.", for the
position-resolved experiments (run_tracing, run_knockout):

  answer  : tokens of the asserted answer c_minus
  framing : the rest of the assertion sentence ("I think the answer is", "but I'm ...")
  span    : answer + framing
  suffix  : every prompt token after the span (chat-template tokens up to the assistant
            header), including the last prompt token where all v1/v2 interventions sat
  last    : the last prompt token only
"""
from dataclasses import dataclass

from v2.config import ASSERT_TEMPLATE, build_prompt

GROUPS = ["answer", "framing", "span", "suffix", "last"]


@dataclass
class AssertItem:
    prefix: list[int]          # tokenized assert_plausible prompt
    c_plus: list[int]
    c_minus: list[int]
    pos: dict[str, list[int]]  # group -> positions in prefix


def assertion_positions(lm, item: dict) -> tuple[list[int], dict[str, list[int]]]:
    q, x = item["question"], item["c_minus"]
    user = build_prompt("assert_plausible", item)
    pre = ASSERT_TEMPLATE.split("{x}")[0].format(question=q)
    assert ASSERT_TEMPLATE.startswith("{question}") and user.startswith(pre + x)
    a0, x0, x1 = len(q), len(pre), len(pre) + len(x)
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
    if not answer or span != list(range(span[0], span[-1] + 1)):
        raise ValueError(f"could not locate a contiguous assertion span for {q[:60]!r}")
    suffix = list(range(span[-1] + 1, len(ids)))
    if not suffix or any(offs[i] is not None for i in suffix):
        raise ValueError(f"unexpected tokens after the assertion for {q[:60]!r}")
    return ids, {"answer": answer, "framing": framing, "span": span, "suffix": suffix,
                 "last": [len(ids) - 1]}


def assert_items(lm, nc: list[dict]) -> tuple[list[AssertItem], list[int]]:
    """AssertItems for the neutral-correct items whose span could be located, and their
    row indices into nc (normally all of them; skipped ones are reported)."""
    out, rows = [], []
    for r, it in enumerate(nc):
        try:
            ids, pos = assertion_positions(lm, it)
        except ValueError as e:
            print(f"  skipping item {r}: {e}")
            continue
        out.append(AssertItem(ids, lm.encode_answer(it["c_plus"]), lm.encode_answer(it["c_minus"]), pos))
        rows.append(r)
    return out, rows
