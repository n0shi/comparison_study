#!/bin/bash
# Feed the remaining setup-v2 graph runs (choices.md) into the queue as slots
# free up. Perun allows 4 jobs per user, running and pending together, so they
# cannot all wait in the queue: this loop, on the login node, submits the next
# run that is ready whenever fewer than MAX of your jobs are queued.
#   - 5 seeds of W2, C1, C2, C3 per task (their tuning is done)
#   - W3: smoke, then tuning per task (after the smoke ends without a
#     traceback), then 5 seeds (after the tuning file exists)
#   - W2 with audio (Step 23): tuning per task, then 5 seeds
#   - W4 (learned edges, Step 13): smoke, then tuning per task, then 5 seeds
#   - B1, B1N (kNN links, Step 16): smoke, then tuning per task, then 5 seeds
# What is finished is read from disk (the tuning file, every seed's
# manifest.json), so only unfinished runs are submitted, and a job skips the
# seeds it already has (train.py --skip-done). A run is not submitted while its
# last job is still queued, and at most TRIES times (logs/queue_graphs.done
# lists every submission, "<job id> <name>"). Safe to stop and start again.
# Start it detached from the package root and follow its log:
#     nohup bash cluster/queue_graphs.sh > logs/queue_graphs.log 2>&1 &
#     tail -f logs/queue_graphs.log
# DRY=1 only prints what is finished and what would be submitted.
# CONV=mean or CONV=wmean queues only the robustness check (choices.md): smoke,
# tuning and 5 seeds of C1, C3, W2 per task with that message function, in
# runs/graphs_v2_<conv> (submissions listed in logs/queue_graphs_<conv>.done).
set -uo pipefail
S=cluster/train_graph.slurm
CONV=${CONV:-}
R=runs/graphs_v2${CONV:+_$CONV}
TASKS=${TASKS:-3_1 3_2 3_3}
MAX=${MAX:-4}                 # Perun: 4 jobs per user
POLL=${POLL:-180}             # seconds between checks
TRIES=${TRIES:-2}             # submissions per run before giving up on it
SEED_LIST=(42 43 44 45 46)
U=${USER:-$(whoami)}
DONE=logs/queue_graphs${CONV:+_$CONV}.done
mkdir -p logs && touch "$DONE"

items=()                      # "name|ready check|VAR=VALUE ..."
add() { items+=("$1|$2|$3"); }
if [[ -n $CONV ]]; then
    C=CONV=$CONV
    add C3_smoke "none" "APPROACH=C3 SMOKE=1 $C"
    for T in $TASKS; do
        for A in C1 C3 W2; do add "${A}_${T}_tune" "smoke C3" "APPROACH=$A TASK=$T TUNE=20 $C"; done
    done
    for T in $TASKS; do
        for A in C1 C3 W2; do add "${A}_${T}_seeds" "tuned $A $T" "APPROACH=$A TASK=$T HYPER=tuned $C"; done
    done
else
    add W3_smoke "none" "APPROACH=W3 SMOKE=1"
    add W4_smoke "none" "APPROACH=W4 SMOKE=1"
    add B1_smoke "none" "APPROACH=B1 SMOKE=1"
    add B1N_smoke "none" "APPROACH=B1N SMOKE=1"
    for T in $TASKS; do
        for A in W2 C1 C2 C3; do add "${A}_${T}_seeds" "none" "APPROACH=$A TASK=$T HYPER=tuned"; done
        add "W3_${T}_tune" "smoke W3" "APPROACH=W3 TASK=$T TUNE=20"
        add "W4_${T}_tune" "smoke W4" "APPROACH=W4 TASK=$T TUNE=20"
        add "B1_${T}_tune" "smoke B1" "APPROACH=B1 TASK=$T TUNE=20"
        add "B1N_${T}_tune" "smoke B1N" "APPROACH=B1N TASK=$T TUNE=20"
        add "W2audio_${T}_tune" "none" "APPROACH=W2 TASK=$T TUNE=20 AUDIO=1"
    done
    for T in $TASKS; do
        add "W3_${T}_seeds" "tuned W3 $T" "APPROACH=W3 TASK=$T HYPER=tuned"
        add "W2audio_${T}_seeds" "tuned W2 $T _audio" "APPROACH=W2 TASK=$T HYPER=tuned AUDIO=1"
        add "W4_${T}_seeds" "tuned W4 $T" "APPROACH=W4 TASK=$T HYPER=tuned"
        add "B1_${T}_seeds" "tuned B1 $T" "APPROACH=B1 TASK=$T HYPER=tuned"
        add "B1N_${T}_seeds" "tuned B1N $T" "APPROACH=B1N TASK=$T HYPER=tuned"
    done
fi

var() { local k=$1; shift; for kv; do [[ $kv == "$k="* ]] && echo "${kv#*=}"; done; }
finished() {                  # finished NAME VAR=VALUE ... : the run's outputs are on disk
    local name=$1; shift
    local A T aud s
    A=$(var APPROACH "$@"); T=$(var TASK "$@"); aud=$([[ -n $(var AUDIO "$@") ]] && echo _audio)
    case $name in
        *_smoke) [[ -f $R/$A/${A}_3_1_seed42_f1_e5_smoke/manifest.json ]] ;;
        *_tune) [[ -f $R/$A/tuning_${T}_e5$aud.json ]] ;;
        *_seeds) for s in "${SEED_LIST[@]}"; do
                     [[ -f $R/$A/${A}_${T}_seed${s}_tuned_e5$aud/manifest.json ]] || return 1
                 done ;;
    esac
}
queued() { squeue -h -u "$U" | wc -l; }
last_id() { grep " $1\$" "$DONE" | tail -1 | cut -d' ' -f1; }
tries() { grep -c " $1\$" "$DONE"; }
in_queue() { local id; id=$(last_id "$1"); [[ -n $id ]] && squeue -h -j "$id" 2>/dev/null | grep -q .; }
ready() {
    case $1 in
        none) return 0 ;;
        smoke) finished "${2}_smoke" APPROACH="$2" ;;
        tuned) [[ -f "$R/$2/tuning_${3}_e5${4:-}.json" ]] ;;
    esac
}
sub() {                       # sub NAME VAR=VALUE ... -> job id
    local name=$1; shift
    local seeds=()
    [[ " $* " == *" HYPER=tuned "* ]] && seeds=(SEEDS="${SEED_LIST[*]}")
    env -u TUNE -u SMOKE -u HYPER -u SEEDS -u AUDIO -u APPROACH -u TASK -u EXTRA -u CONV "$@" "${seeds[@]}" \
        sbatch --parsable -o "logs/cmp_graph_%j_${name}${CONV:+_$CONV}.out" -e "logs/cmp_graph_%j_${name}${CONV:+_$CONV}.err" "$S"
}

if [[ -n ${DRY:-} ]]; then
    for it in "${items[@]}"; do
        IFS='|' read -r name need vars <<< "$it"
        if finished "$name" $vars; then st="finished"
        elif in_queue "$name"; then st="in queue (job $(last_id "$name"))"
        elif (( $(tries "$name") >= TRIES )); then st="FAILED $TRIES times, not resubmitted"
        elif ready $need; then st="to submit"
        else st="waits for its input"; fi
        printf '  %-22s %s\n' "$name" "$st"
    done
    exit 0
fi

idle=0
while :; do
    todo=() progress=0
    for it in "${items[@]}"; do
        IFS='|' read -r name need vars <<< "$it"
        finished "$name" $vars || todo+=("$it")
    done
    if (( ${#todo[@]} == 0 )); then echo "$(date '+%F %T') everything finished"; exit 0; fi
    for it in "${todo[@]}"; do
        IFS='|' read -r name need vars <<< "$it"
        in_queue "$name" && continue
        (( $(tries "$name") < TRIES )) || continue
        (( $(queued) < MAX )) || break
        ready $need || continue
        finished "$name" $vars && continue     # it may have finished since the scan above
        if id=$(sub "$name" $vars); then
            echo "$id $name" >> "$DONE"; progress=1
            echo "$(date '+%F %T') submitted $id $name (try $(tries "$name"))"
        else
            echo "$(date '+%F %T') sbatch refused $name; retrying later"; break
        fi
    done
    if (( progress == 0 )) && (( $(queued) == 0 )); then idle=$((idle + 1)); else idle=0; fi
    if (( idle >= 2 )); then                  # twice in a row: a job may have just ended
        echo "$(date '+%F %T') nothing queued; these are unfinished and cannot be submitted:"
        for it in "${todo[@]}"; do
            IFS='|' read -r name need vars <<< "$it"
            echo "  $name (tries $(tries "$name"))"
        done
        exit 1
    fi
    sleep "$POLL"
done
