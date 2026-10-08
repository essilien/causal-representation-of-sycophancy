# v2: rerun of all experiments

A full rerun of the project with the revisions planned for the EACL 2027 SRW submission. It is self-contained:
v1 code (`common/`, `phase*/`, `controls/`) is untouched and not imported.

| ID | Question | Where |
|----|----------|-------|
| E1 | More statistical power | All 1813 deduplicated questions (996 TriviaQA, 817 TruthfulQA) instead of 1000 sampled; 70/10/20 train/val/test split. Assuming ~60% neutral-correct, the test split grows from 121 to ~218 items per seed (about 1.8x; the binomial SE of IIA drops from ~4.5 to ~3.4 points). |
| E2 | Is the mid-network "DAS ≥ patching" real? | 3 seeds (each seed sets both the data split and the DAS init), all 32 blocks. Reports IIA, balanced accuracy, Pearson r and the fraction of the shift recovered, each with a 95% item-bootstrap CI plus the across-seed SD (init/split variability is not inside the CI), and paired DAS−patch differences. |
| E3 | Is the late-layer gap a rank or optimization artifact? | Rank sweep k ∈ {1, 4, 16, 64, 256, 1024} at 8 blocks with 3 seeds. LR is calibrated on val separately for k = 1, 64 and 1024. Checkpoints are selected by val loss. |
| E4 | Sycophancy vs. contextual entrainment | 2×2 behavioral design (framing × content, 3 mention wordings). DAS is also trained on the entrainment-only and assertion-only sources. Outputs a transfer matrix and the subspace overlap. |
| E5 | Generalization | Same pipeline with `MODEL=qwen` (Qwen2.5-7B-Instruct). |
| E6 | Where in the prompt is the assertion carried, and from which block on? | `run_tracing.py`: causal tracing inside the biased prompt. The assertion span's input embeddings are noised, then the clean state of one (block, position group) is restored, or in the other direction the corrupted state is inserted into the clean run. Groups: asserted answer, framing, whole span, post-assertion template tokens, last token. `fig_tracing.png` |
| E7 | Is the shift read from the assertion via attention, and at which depth? | `run_knockout.py`: positions after the assertion cannot attend to the answer / framing / span, in blocks b..L−1 or in a sliding window of 4 blocks. `fig_knockout.png` |
| E8 | Is the late read sycophancy or entrainment? Are E6/E7 robust? Is the DAS subspace necessary? | E6/E7 rerun with `mention_plausible_1/2` and `assert_irrelevant` prompts (`fig_condition_compare.png`), two more noise draws, and a noise-free corruption that swaps the asserted answer for a token-aligned other answer (`--corruption resample`). `run_illusion_control --direction reverse`: the trained subspace is set to the neutral value in the biased run; an illusory subspace is sufficient but not necessary (`fig_illusion_reverse.png`). All submitted by `submit_mechanism.sh` |
| (limitation 1) | Is late-layer patching just "copying the output"? | Every intervention logs the margin on the first answer token and on tokens 2..k separately; `analyze` reports the shift recovered on each (`fig_first_vs_rest_k64.png`). If the late-layer patching advantage vanishes on tokens 2..k, it is carried by the token read directly off the patched position. |

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

E6/E7 need only the behavior stage and run separately (about 1 GPU-hour each, as 4 array tasks):

```bash
sbatch --mail-user="$SYCO_MAIL" --export=ALL,MODEL=llama v2/slurm/tracing.sbatch
sbatch --mail-user="$SYCO_MAIL" --export=ALL,MODEL=llama v2/slurm/knockout.sbatch
bash v2/slurm/submit_mechanism.sh llama   # E8: 10 jobs + a final analysis job that mails once
```

Results go to `$(ws_find syco)/results/<model>/`:
- `behavior/`: items, scores, activation cache
- `probe.json`
- `calibrate/`, `main/`, `rank/`, `ablation/`: one JSON per seed, block, method and rank, plus the trained W
- `illusion/`, `illusion_reverse/`, `tracing/<condition>__<corruption>__s<seed>/`, `knockout/<condition>/`: one JSON per block with per-item margins (the first assert_plausible E6/E7 runs sit directly in `tracing/` and `knockout/`)
- `analysis/`: CSV, LaTeX and PNG outputs

Finished outputs are skipped on rerun, so a job that timed out can simply be resubmitted.

## Local test

`python -m v2.tests.test_pipeline --dataset-path answer.jsonl` runs unit and end-to-end tests on a tiny random model on CPU in about 10 s.
