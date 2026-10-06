#!/bin/bash
# E4: DAS trained on the entrainment-ablation conditions (assert_irrelevant: assertion
# without the wrong answer's string; mention_plausible_1: the wrong answer's string without
# assertion), evaluated on all conditions -> transfer matrix + subspace overlap with main.
# Array task = source x seed x block chunk (2 x 3 x 2).
#SBATCH --job-name=syco-ablation
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL   # recipient: --mail-user from submit_all.sh (SYCO_MAIL)
source v2/slurm/env.sh
SOURCES=(assert_irrelevant mention_plausible_1)
NCHUNK=2
SRC=${SOURCES[$((SLURM_ARRAY_TASK_ID / 6))]}
SEED=$(((SLURM_ARRAY_TASK_ID % 6) / NCHUNK))
CH=$((SLURM_ARRAY_TASK_ID % NCHUNK))
python -m v2.run_intervention --model "$MODEL" --results-root "$SYCO_RESULTS" --tag ablation \
    --seeds "$SEED" --blocks all --chunk "$CH/$NCHUNK" --methods das --ranks 64 \
    --source "$SRC" --lr auto
