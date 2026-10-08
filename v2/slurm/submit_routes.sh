#!/bin/bash
# How does the assertion's framing strengthen the read of the answer? (tag vs. gate routes)
# and certainty markers before vs. after the answer, for one model. No DAS training.
#   1. behavior_add : score + cache config.CERTAINTY (waits for any other behavior_add job
#                     of this user via --dependency=singleton, so they never run together)
#   2. knockout --routes for assert_plausible, mention_plausible_1 and the four marker
#      conditions (hedge_pre, hedge_post, sure_pre, sure_post)
#   3. final analysis (the only completion mail)
# Usage: bash v2/slurm/submit_routes.sh llama
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL=${1:-llama}
P=${PARTITION:-gpu_a100_il}
mkdir -p logs
MAIL=()
[ -n "${SYCO_MAIL:-}" ] && MAIL=(--mail-user="$SYCO_MAIL")
[ -z "${SYCO_MAIL:-}" ] && echo "Note: SYCO_MAIL not set, no email notifications."
sub() { local env=$1; shift; sbatch --parsable -p "$P" ${MAIL[@]+"${MAIL[@]}"} --export=ALL,MODEL="$MODEL""$env" "$@"; }

B=$(sub "" --mail-type=FAIL --dependency=singleton v2/slurm/behavior_add.sbatch)
IDS=()
for c in assert_plausible mention_plausible_1 hedge_pre hedge_post sure_pre sure_post; do
    IDS+=("$(sub ",COND=$c,ROUTES=1" --mail-type=FAIL --dependency=afterok:"$B" v2/slurm/knockout.sbatch)")
done
DEP=$(IFS=:; echo "${IDS[*]}")
A=$(sub ",PARTS=behavior knockout" --dependency=afterany:"$DEP" v2/slurm/analyze.sbatch)
echo "$MODEL on $P: behavior_add $B, then ${#IDS[@]} knockout arrays (${IDS[*]}); final analysis $A (mails when done)"
