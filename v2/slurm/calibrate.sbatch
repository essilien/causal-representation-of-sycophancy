#!/bin/bash
# Chooses the DAS learning rate on the VALIDATION split (blocks 13 and 25 = v1 layers 14
# and 26), separately for k = 64, 1 and 1024 (the extremes of the rank sweep, so a poor
# large-rank result cannot be blamed on a learning rate tuned for k = 64). Writes
# <model>/calibrate/best_lr_k<k>.json + best_lr.json (k=64); later jobs use --lr auto,
# and ranks 4/16/256 fall back to the k=64 choice.
#SBATCH --job-name=syco-calibrate
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
source v2/slurm/env.sh
python -m v2.run_intervention --model "$MODEL" --results-root "$SYCO_RESULTS" --tag calibrate \
    --seeds 0 --blocks 13,25 --ranks 64 1 1024 --calibrate-lrs 5e-4 2e-3 8e-3
