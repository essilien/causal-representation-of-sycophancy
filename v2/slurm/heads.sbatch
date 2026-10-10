#!/bin/bash
# E13: per-head reads of the inserted answer at the last prompt token, and head knockout
# (v2/run_heads.py). No training; needs the behavior stage (incl. the E14 conditions).
#SBATCH --job-name=syco-heads
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL   # recipient: --mail-user (SYCO_MAIL)
source v2/slurm/env.sh
python -m v2.run_heads --model "$MODEL" --results-root "$SYCO_RESULTS"
