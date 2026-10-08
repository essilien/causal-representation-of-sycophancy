#!/bin/bash
# Interpretability-illusion control: re-evaluates the trained main-run DAS subspaces with
# mismatched sources (no training). Run after main has finished:
#   sbatch --mail-user="$SYCO_MAIL" --export=ALL,MODEL=llama v2/slurm/illusion.sbatch
# DIRECTION=reverse runs the necessity test (biased base, subspace set to the neutral source).
#SBATCH --job-name=syco-illusion
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL   # recipient: --mail-user (SYCO_MAIL)
source v2/slurm/env.sh
python -m v2.run_illusion_control --model "$MODEL" --results-root "$SYCO_RESULTS" --direction "${DIRECTION:-forward}"
python -m v2.analyze --model "$MODEL" --results-root "$SYCO_RESULTS" --only main illusion
