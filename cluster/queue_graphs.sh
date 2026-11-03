#!/bin/bash
# Submit every remaining setup-v2 graph run at once (choices.md), as single jobs
# chained by dependencies: Perun allows only 3 active array jobs.
#   - 5 seeds of W2, C1, C2, C3 per task (their tuning is done)
#   - W3: smoke, then tuning per task, then 5 seeds
#   - W2 with audio (Step 23): tuning per task, then 5 seeds
# A job whose prerequisite fails stays as DependencyNeverSatisfied; scancel it.
# Run from the package root on the login node:
#     bash cluster/queue_graphs.sh             # all three tasks
#     TASKS="3_3" bash cluster/queue_graphs.sh
set -euo pipefail
S=cluster/train_graph.slurm
TASKS=${TASKS:-3_1 3_2 3_3}
SEEDS5="42 43 44 45 46"
mkdir -p logs

# sub NAME DEPENDENCY VAR=VALUE... : submit one job, print and return its id
sub() {
    local name=$1 dep=$2; shift 2
    local id
    id=$(env -u TUNE -u SMOKE -u HYPER -u SEEDS -u AUDIO -u APPROACH -u TASK -u EXTRA "$@" \
         sbatch --parsable ${dep:+--dependency=afterok:$dep} \
         -o "logs/cmp_graph_%j_${name}.out" -e "logs/cmp_graph_%j_${name}.err" "$S")
    echo "$id  $name${dep:+  (after $dep)}" >&2
    echo "$id"
}

smoke=$(sub W3_smoke "" APPROACH=W3 SMOKE=1)
for T in $TASKS; do
    for A in W2 C1 C2 C3; do
        sub "${A}_${T}_seeds" "" APPROACH=$A TASK=$T HYPER=tuned SEEDS="$SEEDS5" >/dev/null
    done
    tune=$(sub "W3_${T}_tune" "$smoke" APPROACH=W3 TASK=$T TUNE=20)
    sub "W3_${T}_seeds" "$tune" APPROACH=W3 TASK=$T HYPER=tuned SEEDS="$SEEDS5" >/dev/null
    tune=$(sub "W2audio_${T}_tune" "" APPROACH=W2 TASK=$T TUNE=20 AUDIO=1)
    sub "W2audio_${T}_seeds" "$tune" APPROACH=W2 TASK=$T HYPER=tuned SEEDS="$SEEDS5" AUDIO=1 >/dev/null
done
squeue -u "${USER:-$(whoami)}"
