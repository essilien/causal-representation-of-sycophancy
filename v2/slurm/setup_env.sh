#!/bin/bash
# One-time setup on a bwUniCluster 3.0 LOGIN node (has internet; compute jobs run offline).
# Run from the repository root:   bash v2/slurm/setup_env.sh
#
# Creates a workspace "syco" (60 days, extend later with: ws_extend syco 60), a Python 3.12
# venv inside it, downloads both models + the dataset into $WS/hf, and runs the CPU test
# suite. You will be asked ONCE for your Hugging Face token (needs accepted access to
# meta-llama/Llama-3.1-8B-Instruct). The token is stored by huggingface_hub in $WS/hf.
set -euo pipefail

WS=$(ws_find syco 2>/dev/null || true)
if [ -z "$WS" ]; then
    WS=$(ws_allocate syco 60)
fi
echo "Workspace: $WS"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
[ -d "$WS/venv" ] || uv venv --python 3.12 "$WS/venv"
source "$WS/venv/bin/activate"
uv pip install -r v2/requirements.txt

export HF_HOME="$WS/hf"
python - <<'EOF'
from huggingface_hub import HfApi, login, snapshot_download, hf_hub_download
try:
    HfApi().whoami()
except Exception:
    login()  # prompts for the token
for repo in ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]:
    print("Downloading", repo)
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
print("Dataset:", hf_hub_download("meg-tong/sycophancy-eval", "answer.jsonl", repo_type="dataset"))
EOF

DATA=$(python -c "from huggingface_hub import hf_hub_download as d; print(d('meg-tong/sycophancy-eval','answer.jsonl',repo_type='dataset'))")
PYTHONPATH=. python -m v2.tests.test_pipeline --dataset-path "$DATA" > "$WS/setup_tests.log" 2>&1 \
    && echo "CPU tests passed (log: $WS/setup_tests.log)" \
    || { echo "CPU tests FAILED, see $WS/setup_tests.log"; exit 1; }
mkdir -p logs
echo "Setup complete. Next: sbatch v2/slurm/smoke.sbatch"
