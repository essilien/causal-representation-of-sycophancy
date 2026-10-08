#!/bin/bash
# Final aggregation after a batch of jobs (CPU work; runs on the GPU partition only so it can
# be chained with --dependency). Sends the single completion mail for submit_mechanism.sh.
#SBATCH --job-name=syco-analyze
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL   # recipient: --mail-user (SYCO_MAIL)
source v2/slurm/env.sh
python -m v2.analyze --model "$MODEL" --results-root "$SYCO_RESULTS" --only ${PARTS:-illusion tracing knockout}
