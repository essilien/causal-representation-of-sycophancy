#!/bin/bash
# Follow-up experiments on where/when the assertion is read (E6/E7) and the DAS necessity
# test, for one model. Needs behavior (and, for the necessity test, main) to be finished.
#   tracing  : mention_plausible_1/2 and assert_irrelevant (noise, seed 0);
#              assert_plausible with noise seeds 1, 2 and resample corruption (seed 0)
#   knockout : mention_plausible_1/2 and assert_irrelevant
#   illusion : DIRECTION=reverse (necessity of the trained k=64 subspaces)
# Individual jobs only mail on failure; one final analysis job mails when everything is done.
# Usage: bash v2/slurm/submit_mechanism.sh llama
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL=${1:-llama}
P=${PARTITION:-gpu_a100_il}
mkdir -p logs
MAIL=()
[ -n "${SYCO_MAIL:-}" ] && MAIL=(--mail-user="$SYCO_MAIL")
[ -z "${SYCO_MAIL:-}" ] && echo "Note: SYCO_MAIL not set, no email notifications."
sub() { local env=$1; shift; sbatch --parsable -p "$P" ${MAIL[@]+"${MAIL[@]}"} --export=ALL,MODEL="$MODEL""$env" "$@"; }

IDS=()
for c in mention_plausible_1 mention_plausible_2 assert_irrelevant; do
    IDS+=("$(sub ",COND=$c" --mail-type=FAIL v2/slurm/tracing.sbatch)")
    IDS+=("$(sub ",COND=$c" --mail-type=FAIL v2/slurm/knockout.sbatch)")
done
for s in 1 2; do
    IDS+=("$(sub ",SEED=$s" --mail-type=FAIL v2/slurm/tracing.sbatch)")
done
IDS+=("$(sub ",CORRUPTION=resample" --mail-type=FAIL v2/slurm/tracing.sbatch)")
IDS+=("$(sub ",DIRECTION=reverse" --mail-type=FAIL v2/slurm/illusion.sbatch)")
DEP=$(IFS=:; echo "${IDS[*]}")
A=$(sub "" --dependency=afterany:"$DEP" v2/slurm/analyze.sbatch)
echo "$MODEL on $P: ${#IDS[@]} jobs (${IDS[*]}); final analysis $A (mails when done)"
