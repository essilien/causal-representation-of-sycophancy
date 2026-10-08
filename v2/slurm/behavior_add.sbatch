#!/bin/bash
# Scores and caches conditions added to config.CONDITIONS after the behavior stage ran
# (existing scores and caches are reused, the neutral-correct set is unchanged). No probing.
#SBATCH --job-name=syco-behavior-add
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL   # recipient: --mail-user (SYCO_MAIL)
source v2/slurm/env.sh
python -m v2.run_behavior --model "$MODEL" --results-root "$SYCO_RESULTS"
