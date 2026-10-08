"""
Why does DAS fall behind full patching near the output, even at k = 1024? Hypothesis: there
the last-token change Delta = h_bias - h_neu mainly raises the logit of the item's OWN
asserted answer token. That direction differs between items, and W is shared across items
and trained on other items, so test items' answer directions lie outside span(W).

For every neutral-correct item, its answer direction at the last prompt token is
    a = g * (U[c_minus_1] - U[c_plus_1])
(U: unembedding, g: final RMSNorm gain, c_*_1: first answer token), i.e. the direction
whose growth raises the first-token logit of c_minus over c_plus (up to the norm's scale).
Per seed and block (main run, k = --k, source assert_plausible):
  delta_on_own   : (Delta . a_own)^2 / |Delta|^2 / |a_own|^2, vs. delta_on_other with another
                   item's answer direction: does Delta point at the item's own answer?
  w_on_test / w_on_train / w_chance : |W^T a|^2 / |a|^2 for test-item and training-item
                   directions, and k/d
  seen           : per test item, whether its c_minus first token occurs among the training
                   items' c_minus first tokens (analyze compares DAS recovery for both groups)
Outputs <results>/<model>/answer_direction/seed*.json; no training, one forward-free pass.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from v2.config import MAIN_SOURCE, MODELS
from v2.data import read_jsonl, write_json
from v2.lm import LM


def unit_rows(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="llama")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    root = Path(args.results_root) / args.model
    beh = root / "behavior"
    meta = json.loads((beh / "meta.json").read_text())
    items = read_jsonl(beh / "items.jsonl")
    nc = [items[i] for i in meta["nc_ids"]]
    cache_b = np.load(beh / "cache" / f"{MAIN_SOURCE}.npy", mmap_mode="r")
    cache_n = np.load(beh / "cache" / "neutral.npy", mmap_mode="r")

    lm = LM.load(MODELS[args.model])
    first_m = [lm.encode_answer(it["c_minus"])[0] for it in nc]
    first_p = [lm.encode_answer(it["c_plus"])[0] for it in nc]
    U = lm.model.get_output_embeddings().weight.float()
    g = lm.model.model.norm.weight.float()
    A = (U[first_m] - U[first_p]) * g  # [n, d]
    same = torch.tensor([m == p for m, p in zip(first_m, first_p)])
    A_unit = unit_rows(A).cpu()
    rng = np.random.default_rng(0)
    other = torch.from_numpy(rng.permutation(len(nc)))
    other = torch.where(other == torch.arange(len(nc)), (other + 1) % len(nc), other)

    out_dir = root / "answer_direction"
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        f = out_dir / f"seed{seed}.json"
        if f.exists():
            continue
        src_dir = root / "main" / f"seed{seed}"
        sp = json.loads((src_dir / "split.json").read_text())
        train, test = np.array(sp["train"]), np.array(sp["test"])
        seen_tokens = {first_m[i] for i in train}
        res = {"model": args.model, "seed": seed, "k": args.k, "test_rows": test.tolist(),
               "seen": [first_m[i] in seen_tokens for i in test],
               "same_first_token": [bool(same[i]) for i in test], "blocks": {}}
        for b in range(lm.n_layers):
            D = torch.from_numpy(np.asarray(cache_b[:, b], dtype=np.float32)
                                 - np.asarray(cache_n[:, b], dtype=np.float32))
            Du = unit_rows(D)
            own = (Du * A_unit).sum(-1) ** 2
            oth = (Du * A_unit[other]).sum(-1) ** 2
            row = {"delta_on_own": own[test].tolist(), "delta_on_other": oth[test].tolist()}
            w_file = src_dir / f"block{b:02d}__das__k{args.k}__src-{MAIN_SOURCE}.W.pt"
            if w_file.exists():
                W = torch.load(w_file).float()
                cap = ((A_unit @ W) ** 2).sum(-1)
                row |= {"w_on_test": cap[test].tolist(), "w_on_train": cap[train].tolist(),
                        "w_chance": args.k / W.shape[0],
                        "delta_in_w": (((Du @ W) ** 2).sum(-1))[test].tolist()}
            res["blocks"][str(b)] = row
        write_json(f, res)
        print(f"seed {seed}: {sum(res['seen'])}/{len(test)} test items have a c_minus first token seen in training")


if __name__ == "__main__":
    main()
