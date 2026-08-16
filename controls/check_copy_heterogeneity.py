"""
Checks whether the ~55.6% copy_control/biased ratio is a stable per-example pattern
(copying dominates roughly equally on every question) or an average masking real
heterogeneity (some questions are almost entirely copying, others almost entirely
belief-driven). Reads copy_control_results.jsonl -- no GPU needed.
"""
import json
import numpy as np
from scipy.stats import pearsonr

records = [json.loads(l) for l in open("/kaggle/working/control_out/copy_control_results.jsonl")]

biased_shift = np.array([r["biased_margin"] - r["neutral_margin"] for r in records])
copy_shift = np.array([r["copy_control_margin"] - r["neutral_margin"] for r in records])

# Per-example ratio (only defined where biased_shift is meaningfully non-zero -- ratios
# near a zero denominator are unstable/uninterpretable, exclude them explicitly rather
# than letting them blow up the summary stats)
valid = np.abs(biased_shift) > 0.5
ratio = copy_shift[valid] / biased_shift[valid]

print(f"n = {len(records)}, n with |biased_shift| > 0.5: {valid.sum()}")

r, p = pearsonr(biased_shift, copy_shift)
print(f"\nPer-example correlation (biased_shift vs copy_shift): r={r:.3f}, p={p:.4g}")
print("-> high r means copying tracks belief-shift consistently across questions (a global")
print("   constant-ratio story); low r means the split varies a lot by question (copying")
print("   dominates some questions, belief dominates others -- not a single fixed ratio)")

print(f"\nPer-example ratio (copy_shift / biased_shift), n={valid.sum()}:")
print(f"  mean={ratio.mean():.3f}  median={np.median(ratio):.3f}  std={ratio.std():.3f}")
for pct in [10, 25, 50, 75, 90]:
    print(f"  {pct}th percentile: {np.percentile(ratio, pct):.3f}")

# residual = the part of biased_shift NOT explained by copying alone -- this is a cleaner
# per-example "belief effect" estimate than the global 44.4% figure, useful if Phase 2's
# regression target gets revised to isolate belief from copying
residual = biased_shift - copy_shift
print(f"\nResidual (biased_shift - copy_shift), i.e. the belief-framing-specific component:")
print(f"  mean={residual.mean():.3f}  median={np.median(residual):.3f}  std={residual.std():.3f}")
print(f"  fraction of examples where residual is still negative (belief framing adds an")
print(f"  effect beyond copying, in the expected direction): {(residual < 0).mean():.1%}")

# quick outlier check: examples where copy_shift alone explains MORE than the full biased
# shift (ratio > 100%) -- worth eyeballing a few, since these are cases where the neutral
# "exposure" framing moved the answer more than the actual user assertion did
over_100 = [(r, cs, bs) for r, cs, bs in zip(records, copy_shift, biased_shift)
            if abs(bs) > 0.5 and cs / bs > 1.0]
print(f"\nExamples where copy_control shift exceeded biased shift (n={len(over_100)}):")
for r, cs, bs in over_100[:5]:
    print(f"- {r['question'][:70]}")
    print(f"  biased_shift={bs:.3f}  copy_shift={cs:.3f}")
