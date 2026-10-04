#!/bin/bash
# 30-minute end-to-end check on the real model with 80 questions. Run this FIRST:
#   sbatch v2/slurm/smoke.sbatch
# then read logs/syco-smoke-<jobid>.out. Check that (1) the sanity line reports a small
# "max |m_neutral diff|", (2) DAS val loss goes down, (3) the per-run seconds are plausible
# -- multiply them out to check the time limits in main/rank/ablation.sbatch.
#SBATCH --job-name=syco-smoke
#SBATCH --partition=dev_gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
source v2/slurm/env.sh
R="$SYCO_WS/smoke"
python -m v2.run_behavior --model "$MODEL" --results-root "$R" --limit 80
python -m v2.run_probe --model "$MODEL" --results-root "$R" --n-perm 5
python -m v2.run_intervention --model "$MODEL" --results-root "$R" --tag smoke --seeds 0 \
    --blocks 13,25 --ranks 64 --steps 60 --eval-every 20 --lr 2e-3
python -m v2.analyze --model "$MODEL" --results-root "$R" --only behavior probe
echo "Smoke test finished."
