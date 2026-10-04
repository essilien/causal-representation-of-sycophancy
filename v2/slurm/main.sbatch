#!/bin/bash
# E2: DAS (k=64) vs. full patching at every block, 3 seeds. Array task = seed x block chunk.
# Each task evaluates every EVAL_SOURCES condition, so E4's patching results come for free.
#SBATCH --job-name=syco-main
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x-%A_%a.out
source v2/slurm/env.sh
NCHUNK=4
SEED=$((SLURM_ARRAY_TASK_ID / NCHUNK))
CH=$((SLURM_ARRAY_TASK_ID % NCHUNK))
python -m v2.run_intervention --model "$MODEL" --results-root "$SYCO_RESULTS" --tag main \
    --seeds "$SEED" --blocks all --chunk "$CH/$NCHUNK" --ranks 64 --lr auto
