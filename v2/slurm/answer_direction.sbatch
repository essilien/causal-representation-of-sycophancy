#!/bin/bash
# Do test items' answer directions lie in span(W)? (v2/run_answer_direction.py; no training)
#SBATCH --job-name=syco-answer-dir
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL   # recipient: --mail-user (SYCO_MAIL)
source v2/slurm/env.sh
python -m v2.run_answer_direction --model "$MODEL" --results-root "$SYCO_RESULTS"
