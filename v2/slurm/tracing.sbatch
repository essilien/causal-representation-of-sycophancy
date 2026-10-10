#!/bin/bash
# Position-resolved causal tracing of the assertion (v2/run_tracing.py). No training; needs
# only the behavior stage. Array task = block chunk. Optional: COND (default
# assert_plausible), CORRUPTION (noise | resample), SEED, TRGROUPS (space-separated position
# groups, see positions.ALL_GROUPS) with VARIANT (run-name suffix). Submit from the repository root:
#   sbatch --mail-user="$SYCO_MAIL" --export=ALL,MODEL=llama v2/slurm/tracing.sbatch
#   sbatch --mail-user="$SYCO_MAIL" --export=ALL,MODEL=llama,COND=mention_plausible_1 v2/slurm/tracing.sbatch
#SBATCH --job-name=syco-tracing
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL   # recipient: --mail-user (SYCO_MAIL)
source v2/slurm/env.sh
python -m v2.run_tracing --model "$MODEL" --results-root "$SYCO_RESULTS" \
    --blocks all --chunk "$SLURM_ARRAY_TASK_ID/4" \
    --condition "${COND:-assert_plausible}" --corruption "${CORRUPTION:-noise}" --seed "${SEED:-0}" \
    ${TRGROUPS:+--groups $TRGROUPS} ${VARIANT:+--variant "$VARIANT"}
