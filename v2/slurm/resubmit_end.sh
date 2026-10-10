#!/bin/bash
# Re-run of the E14 "assert_end" jobs after two fixes (2026-10-11):
#   - tracing crashed when a position group was empty for a whole batch (Qwen: the last
#     batch has one item, whose answer merges with the final ".")
#   - knockout --routes dropped the `post` variant, which exists for only 86-90% of the
#     items under assert_end (threshold now 80%; items without it get NaN)
# Route knockouts go to a new folder (knockout/<cond>__routes__full), so jobs of the first
# submission that are still running cannot collide with them. Tracing resumes per block
# (existing block files are skipped); pass "tracing" only if those arrays failed.
# Usage: bash v2/slurm/resubmit_end.sh qwen tracing   |   bash v2/slurm/resubmit_end.sh llama
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL=${1:-llama}
P=${PARTITION:-gpu_a100_il}
mkdir -p logs
MAIL=()
[ -n "${SYCO_MAIL:-}" ] && MAIL=(--mail-user="$SYCO_MAIL")
sub() { local env=$1; shift; sbatch --parsable -p "$P" ${MAIL[@]+"${MAIL[@]}"} --export=ALL,MODEL="$MODEL""$env" "$@"; }

IDS=()
for c in assert_end_plausible assert_end_irrelevant; do
    IDS+=("$(sub ",COND=$c,ROUTES=1,TAG=full" --mail-type=FAIL v2/slurm/knockout.sbatch)")
    if [ "${2:-}" = "tracing" ]; then
        IDS+=("$(sub ",COND=$c,TRGROUPS=answer span suffix last pre post tmpl" --mail-type=FAIL --time=08:00:00 v2/slurm/tracing.sbatch)")
    fi
done
DEP=$(IFS=:; echo "${IDS[*]}")
A=$(sub ",PARTS=behavior scoring tracing knockout heads" --time=03:00:00 --dependency=afterany:"$DEP" v2/slurm/analyze.sbatch)
echo "$MODEL: ${#IDS[@]} jobs (${IDS[*]}); final analysis $A (mails when done)"
