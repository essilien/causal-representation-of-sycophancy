#!/bin/bash
# E13 (attention heads) and E14 (further assertion templates) for one model. No DAS training.
#   1. behavior_add : score + cache the E14 conditions (config.ASSERT_VARIANTS); waits for any
#                     other behavior_add job of this user (--dependency=singleton)
#   2. E14, for assert_end / assert_alt x plausible / irrelevant: noise tracing (with the
#      pre / post / tmpl groups), standard knockout, route knockout
#   3. main template: tracing of the pre / post / tmpl groups (VARIANT pp) for assert_plausible
#      and assert_irrelevant; knockout of the whole span with per-candidate log-probs (TAG cand)
#      for assert_plausible, assert_irrelevant and mention_plausible_1
#   4. E13 head recording + head knockout
#   5. final analysis (the only completion mail)
# Usage: bash v2/slurm/submit_heads_templates.sh llama
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
AFTER=(--mail-type=FAIL --dependency=afterok:"$B")
TR_GROUPS="answer span suffix last pre post tmpl"
IDS=()
for t in assert_end assert_alt; do
    for c in plausible irrelevant; do
        IDS+=("$(sub ",COND=${t}_$c,TRGROUPS=$TR_GROUPS" "${AFTER[@]}" --time=08:00:00 v2/slurm/tracing.sbatch)")
        IDS+=("$(sub ",COND=${t}_$c" "${AFTER[@]}" v2/slurm/knockout.sbatch)")
        IDS+=("$(sub ",COND=${t}_$c,ROUTES=1" "${AFTER[@]}" v2/slurm/knockout.sbatch)")
    done
done
for c in assert_plausible assert_irrelevant; do
    IDS+=("$(sub ",COND=$c,TRGROUPS=pre post tmpl,VARIANT=pp" "${AFTER[@]}" v2/slurm/tracing.sbatch)")
done
for c in assert_plausible assert_irrelevant mention_plausible_1; do
    IDS+=("$(sub ",COND=$c,KEYS=span,TAG=cand" "${AFTER[@]}" v2/slurm/knockout.sbatch)")
done
IDS+=("$(sub "" "${AFTER[@]}" v2/slurm/heads.sbatch)")
DEP=$(IFS=:; echo "${IDS[*]}")
A=$(sub ",PARTS=behavior scoring tracing knockout heads" --time=03:00:00 --dependency=afterany:"$DEP" v2/slurm/analyze.sbatch)
echo "$MODEL on $P: behavior_add $B, then ${#IDS[@]} jobs (${IDS[*]}); final analysis $A (mails when done)"
