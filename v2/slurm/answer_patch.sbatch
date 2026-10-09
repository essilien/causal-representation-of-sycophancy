#!/bin/bash
# Item-specific answer-direction interventions (v2/run_answer_patch.py; no training), then
# the summary next to DAS and full patching. Submit from the repository root:
#   sbatch --mail-user="$SYCO_MAIL" --export=ALL,MODEL=llama v2/slurm/answer_patch.sbatch
#SBATCH --job-name=syco-answer-patch
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL   # recipient: --mail-user (SYCO_MAIL)
source v2/slurm/env.sh
python -m v2.run_answer_patch --model "$MODEL" --results-root "$SYCO_RESULTS"
python -m v2.analyze --model "$MODEL" --results-root "$SYCO_RESULTS" --only answer_patch
