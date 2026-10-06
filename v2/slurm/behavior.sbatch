#!/bin/bash
# E1 + entrainment-ablation behavior on all 1813 questions, activation cache, probing.
#SBATCH --job-name=syco-behavior
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL   # recipient: --mail-user from submit_all.sh (SYCO_MAIL)
source v2/slurm/env.sh
python -m v2.run_behavior --model "$MODEL" --results-root "$SYCO_RESULTS"
python -m v2.run_probe --model "$MODEL" --results-root "$SYCO_RESULTS" --n-perm 200
