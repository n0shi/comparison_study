#!/bin/bash
# Feed the remaining setup-v2 graph runs (choices.md) into the queue as slots
# free up. Perun allows 4 jobs per user, running and pending together, so they
# cannot all wait in the queue: this loop, on the login node, submits the next
# run that is ready whenever fewer than MAX of your jobs are queued.
#   - 5 seeds of W2, C1, C2, C3 per task (their tuning is done)
#   - W3: smoke, then tuning per task (after the smoke ends without a
#     traceback), then 5 seeds (after the tuning file exists)
#   - W2 with audio (Step 23): tuning per task, then 5 seeds
# Every submitted run is appended to $DONE ("<job id> <name>"); runs listed
# there are skipped, so the loop can be stopped and started again.
# Start it detached from the package root and follow its log:
#     nohup bash cluster/queue_graphs.sh > logs/queue_graphs.log 2>&1 &
#     tail -f logs/queue_graphs.log
# It exits when everything is submitted, or when nothing is queued and the
# runs left wait for an input that will not come (a failed smoke or tuning).
set -uo pipefail
S=cluster/train_graph.slurm
TASKS=${TASKS:-3_1 3_2 3_3}
MAX=${MAX:-4}                 # Perun: 4 jobs per user
POLL=${POLL:-180}             # seconds between checks
SEEDS5="42 43 44 45 46"
U=${USER:-$(whoami)}
DONE=logs/queue_graphs.done
mkdir -p logs && touch "$DONE"

items=()                      # "name|ready check|VAR=VALUE ..."
add() { items+=("$1|$2|$3"); }
add W3_smoke "none" "APPROACH=W3 SMOKE=1"
for T in $TASKS; do
    for A in W2 C1 C2 C3; do add "${A}_${T}_seeds" "none" "APPROACH=$A TASK=$T HYPER=tuned"; done
    add "W3_${T}_tune" "smoke" "APPROACH=W3 TASK=$T TUNE=20"
    add "W2audio_${T}_tune" "none" "APPROACH=W2 TASK=$T TUNE=20 AUDIO=1"
done
for T in $TASKS; do
    add "W3_${T}_seeds" "tuned W3 $T" "APPROACH=W3 TASK=$T HYPER=tuned"
    add "W2audio_${T}_seeds" "tuned W2 $T _audio" "APPROACH=W2 TASK=$T HYPER=tuned AUDIO=1"
done

queued() { squeue -h -u "$U" | wc -l; }
submitted() { grep -q " $1\$" "$DONE"; }
smoke_ok() {                  # the W3 smoke left the queue and its log has no failure
    local id
    id=$(grep ' W3_smoke$' "$DONE" | tail -1 | cut -d' ' -f1)
    [[ -n "$id" ]] && ! squeue -h -j "$id" 2>/dev/null | grep -q . &&
        [[ -f "logs/cmp_graph_${id}_W3_smoke.err" ]] &&
        ! grep -qE 'Traceback|CANCELLED|oom-kill|Killed' "logs/cmp_graph_${id}_W3_smoke.err"
}
ready() {
    case $1 in
        none) return 0 ;;
        smoke) smoke_ok ;;
        tuned) [[ -f "runs/graphs_v2/$2/tuning_${3}_e5${4:-}.json" ]] ;;
    esac
}
sub() {                       # sub NAME VAR=VALUE ... -> job id
    local name=$1; shift
    local seeds=()
    [[ " $* " == *" HYPER=tuned "* ]] && seeds=(SEEDS="$SEEDS5")
    env -u TUNE -u SMOKE -u HYPER -u SEEDS -u AUDIO -u APPROACH -u TASK -u EXTRA "$@" "${seeds[@]}" \
        sbatch --parsable -o "logs/cmp_graph_%j_${name}.out" -e "logs/cmp_graph_%j_${name}.err" "$S"
}

while :; do
    left=() progress=0
    for it in "${items[@]}"; do
        IFS='|' read -r name need vars <<< "$it"
        submitted "$name" || left+=("$name")
    done
    if (( ${#left[@]} == 0 )); then echo "$(date '+%F %T') everything submitted"; exit 0; fi
    for it in "${items[@]}"; do
        IFS='|' read -r name need vars <<< "$it"
        submitted "$name" && continue
        (( $(queued) < MAX )) || break
        ready $need || continue
        if id=$(sub "$name" $vars); then
            echo "$id $name" >> "$DONE"; progress=1
            echo "$(date '+%F %T') submitted $id $name"
        else
            echo "$(date '+%F %T') sbatch refused $name; retrying later"; break
        fi
    done
    if (( progress == 0 )) && (( $(queued) == 0 )); then
        echo "$(date '+%F %T') nothing queued and these wait for inputs that will not come:"
        printf '  %s\n' "${left[@]}"
        exit 1
    fi
    sleep "$POLL"
done
