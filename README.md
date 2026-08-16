# A Depth-Dependent Causal Geometry of Sycophantic Agreement in Llama-3.1-8B-Instruct

Mechanistic-interpretability investigation of sycophancy in Llama-3.1-8B-Instruct,
combining linear probing, full-representation activation patching, and Distributed
Alignment Search (DAS) to test whether sycophantic agreement is driven by a compact,
low-dimensional causal variable, and whether that compactness holds uniformly across
network depth.

See [`paper/paper_draft.md`](paper/paper_draft.md) for the full write-up (abstract,
methods, results, discussion). The core finding: a 64-dimensional trained subspace
matches or exceeds full-representation patching in mid-network blocks (12–21), but falls
increasingly behind it at greater depth (blocks 22–31) — evidence that the causal
geometry of the effect changes qualitatively with depth, not a single uniform mechanism.

## Repository structure

```
common/                  Shared model-loading, scoring, and dataset utilities used by
                          every phase below (SYSTEM_PROMPT, teacher-forced margin
                          scoring, dataset loading/deduplication).
phase1_baseline/          Baseline sycophancy measurement (neutral vs. biased flip rate).
phase2_layer_prescreen/   Layer-wise regression probing to localize a candidate range of
                          blocks for causal validation.
phase3_das/               Interchange-intervention experiments: DAS training/evaluation,
                          full-representation patching baseline, rank- and
                          position-sensitivity follow-ups, and figure generation.
controls/                 Copy-effect and irrelevant-answer controls testing whether the
                          effect is separable from contextual entrainment, plus
                          per-example overlap/decomposition analyses.
figures/                  Generated figures referenced in the paper.
paper/                    The paper draft and a running results summary.
```

Every script is a standalone, argparse-driven module — runnable with
`python <script>.py --help` on any machine with the right dependencies, not
notebook-cell fragments. `common/scoring.py` holds the single definition of
`SYSTEM_PROMPT` and the teacher-forcing scorer that every other script imports, so
results stay comparable across phases without needing to keep multiple copies in sync.

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=...   # needs access to meta-llama/Llama-3.1-8B-Instruct (gated)
```

A single GPU with at least ~20GB is recommended (model is loaded in bf16). Note:
**pyvene, used for Phase 3's interchange interventions, does not support multi-GPU model
sharding** — if you see a device-mismatch error, run on a single visible GPU
(`CUDA_VISIBLE_DEVICES=0`) rather than a multi-GPU `device_map="auto"` split.

## Running the pipeline

```bash
# 1. Baseline
python phase1_baseline/run_phase1.py --n-samples 800 --output-dir ./out/phase1

# 2. Layer localization
python phase2_layer_prescreen/run_phase2.py \
    --phase1-results ./out/phase1/phase1_full_results.json --output-dir ./out/phase2

# 3a. DAS training/evaluation across candidate (or all) blocks
python phase3_das/run_phase3.py --layers 12 13 14 15 16 17 18 19 20 21 \
    --phase1-results ./out/phase1/phase1_full_results.json --output-dir ./out/phase3

# 3b. Full-representation patching baseline, for comparison
python phase3_das/full_patch_baseline.py \
    --phase1-results ./out/phase1/phase1_full_results.json --output-dir ./out/phase3

# 4. Controls
python controls/copy_effect_control.py \
    --phase1-results ./out/phase1/phase1_full_results.json --output-dir ./out/controls
python controls/irrelevant_answer_control.py \
    --phase1-results ./out/phase1/phase1_full_results.json --output-dir ./out/controls
```

Every script's own `--help` documents its full argument list; module docstrings explain
the reasoning behind non-obvious design choices (e.g., why batch size is 1, why the
layer-numbering convention is what it is, why full-representation patching's cross-layer
comparison needs a specific caveat). Several diagnostic/follow-up scripts in
`phase3_das/` and `controls/` (loss-plateau checks, rank/position sensitivity, per-example
overlap, margin decomposition) are meant to be run selectively once you have a specific
question, not as part of the main pipeline — their own docstrings explain when to reach
for them.

## Known limitations

See the Discussion section of the paper for the full account. In short: the copy-effect
control does not converge across template wordings, so contextual entrainment and
sycophantic agreement cannot be cleanly separated in this design; the late-layer
DAS/patching gap is established only negatively (two candidate explanations were ruled
out, not a mechanism confirmed); several training/evaluation choices (batch size 1, a
few-hundred-example test set) introduce quantified but real variance; and all results are
specific to one model, one task format, and the "sycophantic agreement" sub-behavior in
the taxonomy of Vennemeyer et al. (2025).
