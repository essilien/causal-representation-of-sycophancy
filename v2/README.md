# v2: rerun of all experiments

A full rerun of the project with the revisions planned for the EACL 2027 SRW submission. It is self-contained:
v1 code (`common/`, `phase*/`, `controls/`) is untouched and not imported.

| ID | Question | Where |
|----|----------|-------|
| E1 | More statistical power | All 1813 deduplicated questions (996 TriviaQA, 817 TruthfulQA) instead of 1000 sampled; 70/10/20 train/val/test split, giving about 3x more test items. |
| E2 | Is the mid-network "DAS ≥ patching" real? | 3 seeds (each seed sets both the data split and the DAS init), all 32 blocks. Reports bootstrap 95% CIs, paired DAS−patch differences, IIA, balanced accuracy, Pearson r, and the fraction of the shift recovered. |
| E3 | Is the late-layer gap a rank or optimization artifact? | Rank sweep k ∈ {1, 4, 16, 64, 256, 1024} at 8 blocks with 3 seeds. LR is calibrated on val. Checkpoints are selected by val loss. |
| E4 | Sycophancy vs. contextual entrainment | 2×2 behavioral design (framing × content, 3 mention wordings). DAS is also trained on the entrainment-only and assertion-only sources. Outputs a transfer matrix and the subspace overlap. |
| E5 | Generalization | Same pipeline with `MODEL=qwen` (Qwen2.5-7B-Instruct). |
| (limitation 1) | Is late-layer patching just "copying the output"? | Every intervention also logs the first-answer-token margin separately. |

## Changes from v1

- **Layer indexing.** `block b` is the output of decoder block `b`, recorded with a hook. In v1, "layer l" was `hidden_states[l]`, which is block l−1, and the last layer was post-norm. So v1 layer 14 is v2 block 13.
- **pyvene is gone.** Interventions use plain forward hooks (`v2/lm.py`). This allows batching (batch 16 instead of 1) and works for any HF decoder.
- **Answer tokenization.** Candidates are tokenized without the leading space v1 added. A model's first answer token after the chat header has no leading space.
- **IIA reporting.** IIA is reported together with balanced accuracy and r. The test set is imbalanced between flips and non-flips, so an "always flip" predictor can beat sign IIA.
- **Probing.** Uses true out-of-fold R² (it can be negative), with alpha selected inside each fold and 200 permutations.

## Running on bwUniCluster 3.0

```bash
ssh <zx-id>@uc3.scc.kit.edu
git clone <repo> && cd <repo>
bash v2/slurm/setup_env.sh               # once, on the login node: venv, models, CPU tests
sbatch v2/slurm/smoke.sbatch             # 30 min check on the real model; read logs/ first
bash v2/slurm/submit_all.sh llama        # full pipeline with job dependencies
bash v2/slurm/submit_all.sh qwen         # E5
squeue -u $USER                          # monitor
source v2/slurm/env.sh && python -m v2.analyze --model llama --results-root $SYCO_RESULTS
```

Results go to `$(ws_find syco)/results/<model>/`:
- `behavior/`: items, scores, activation cache
- `probe.json`
- `calibrate/`, `main/`, `rank/`, `ablation/`: one JSON per seed, block, method and rank, plus the trained W
- `analysis/`: CSV, LaTeX and PNG outputs

Finished outputs are skipped on rerun, so a job that timed out can simply be resubmitted.

## Local test

`python -m v2.tests.test_pipeline --dataset-path answer.jsonl` runs unit and end-to-end tests on a tiny random model on CPU in about 10 s.
