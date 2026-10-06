#!/bin/bash
# Submits the full pipeline for one model with dependencies:
#   behavior(+probe) -> calibrate -> { main, ablation, rank* }       (*llama only)
# Usage (from anywhere):
#   bash v2/slurm/submit_all.sh llama
#   bash v2/slurm/submit_all.sh qwen          # E5 replication
#   PARTITION=gpu_h100 bash v2/slurm/submit_all.sh llama   # different GPU partition
# Jobs whose outputs already exist skip that work, so resubmitting after a timeout is safe.
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL=${1:-llama}
P=${PARTITION:-gpu_a100_il}
mkdir -p logs
# Email on END/FAIL of each job (array jobs: one mail for the whole array). The address
# is kept out of this public repo: put `export SYCO_MAIL=you@example.com` in ~/.bashrc.
MAIL=()
[ -n "${SYCO_MAIL:-}" ] && MAIL=(--mail-user="$SYCO_MAIL")
[ -z "${SYCO_MAIL:-}" ] && echo "Note: SYCO_MAIL not set, no email notifications."
sub() { sbatch --parsable -p "$P" "${MAIL[@]}" --export=ALL,MODEL="$MODEL" "$@"; }

J1=$(sub v2/slurm/behavior.sbatch)
J2=$(sub --dependency=afterok:"$J1" v2/slurm/calibrate.sbatch)
J3=$(sub --dependency=afterok:"$J2" v2/slurm/main.sbatch)
J4=$(sub --dependency=afterok:"$J2" v2/slurm/ablation.sbatch)
echo "$MODEL on $P: behavior=$J1 calibrate=$J2 main=$J3 ablation=$J4"
if [ "$MODEL" = "llama" ]; then
    J5=$(sub --dependency=afterok:"$J2" v2/slurm/rank.sbatch)
    echo "rank=$J5"
fi
echo "Monitor with: squeue -u \$USER     Logs in: $(pwd)/logs"
echo "When done:    source v2/slurm/env.sh && python -m v2.analyze --model $MODEL --results-root \$SYCO_RESULTS"
