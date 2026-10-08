#!/bin/bash
# The trained main-run DAS subspaces (k=64, no training) evaluated with other conditions as
# the biased side: inserted into the neutral run and removed from that condition's run.
# CONDS: space-separated condition names. Matched sources only.
#SBATCH --job-name=syco-das-eval
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL   # recipient: --mail-user (SYCO_MAIL)
source v2/slurm/env.sh
for c in $CONDS; do
    for d in forward reverse; do
        python -m v2.run_illusion_control --model "$MODEL" --results-root "$SYCO_RESULTS" \
            --eval-condition "$c" --direction "$d" --controls matched
    done
done
