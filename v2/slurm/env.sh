# Sourced by every job script. Jobs must be submitted from the repository root
# (submit_all.sh takes care of that).
set -eo pipefail
WS=$(ws_find syco)
export SYCO_WS="$WS"
export HF_HOME="$WS/hf"
export HF_HUB_OFFLINE=1          # everything was pre-downloaded by setup_env.sh
export TOKENIZERS_PARALLELISM=false
export SYCO_RESULTS="${SYCO_RESULTS:-$WS/results}"
export MODEL="${MODEL:-llama}"
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
source "$WS/venv/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-none} model=$MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
