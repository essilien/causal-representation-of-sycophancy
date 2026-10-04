#!/bin/bash
# E3: DAS rank sweep at 8 blocks spanning the network, 3 seeds. Array task = seed x rank.
#SBATCH --job-name=syco-rank
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/%x-%A_%a.out
source v2/slurm/env.sh
RANKS=(1 4 16 64 256 1024)
K=${RANKS[$((SLURM_ARRAY_TASK_ID % 6))]}
SEED=$((SLURM_ARRAY_TASK_ID / 6))
python -m v2.run_intervention --model "$MODEL" --results-root "$SYCO_RESULTS" --tag rank \
    --seeds "$SEED" --blocks "${RANK_BLOCKS:-5,9,13,17,21,25,29,31}" --methods das --ranks "$K" --lr auto
