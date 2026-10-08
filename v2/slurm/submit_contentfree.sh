#!/bin/bash
# Content-free framing (config.CONTENT_FREE) and the answer-direction test, for one model.
# No DAS training: the existing main-run subspaces are only evaluated.
#   1. behavior_add  : score + cache the 5 content-free conditions
#   2. das_eval      : all 5, inserted into the neutral run and removed from their own run
#   3. knockout      : assert_empty_1 (minimal pair), assert_empty_2 (grammatical),
#                      mention_empty_1 (minimal pair of the mention)
#   4. answer_direction
#   5. final analysis (the only completion mail; other jobs mail on failure)
# Usage: bash v2/slurm/submit_contentfree.sh llama
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL=${1:-llama}
P=${PARTITION:-gpu_a100_il}
mkdir -p logs
MAIL=()
[ -n "${SYCO_MAIL:-}" ] && MAIL=(--mail-user="$SYCO_MAIL")
[ -z "${SYCO_MAIL:-}" ] && echo "Note: SYCO_MAIL not set, no email notifications."
sub() { local env=$1; shift; sbatch --parsable -p "$P" ${MAIL[@]+"${MAIL[@]}"} --export=ALL,MODEL="$MODEL""$env" "$@"; }

B=$(sub "" --mail-type=FAIL v2/slurm/behavior_add.sbatch)
IDS=()
IDS+=("$(sub ",CONDS=assert_empty_1 assert_empty_2 assert_empty_3 mention_empty_1 mention_empty_2" \
         --mail-type=FAIL --dependency=afterok:"$B" v2/slurm/das_eval.sbatch)")
for c in assert_empty_1 assert_empty_2 mention_empty_1; do
    IDS+=("$(sub ",COND=$c" --mail-type=FAIL --dependency=afterok:"$B" v2/slurm/knockout.sbatch)")
done
IDS+=("$(sub "" --mail-type=FAIL --dependency=afterok:"$B" v2/slurm/answer_direction.sbatch)")
DEP=$(IFS=:; echo "${IDS[*]}")
A=$(sub ",PARTS=behavior illusion knockout answer_direction" --dependency=afterany:"$DEP" v2/slurm/analyze.sbatch)
echo "$MODEL on $P: behavior_add $B, then ${#IDS[@]} jobs (${IDS[*]}); final analysis $A (mails when done)"
