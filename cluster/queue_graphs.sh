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
# V3=1 queues setup v3 (choices.md): tuning with 3 seeds per trial, then 5 seeds,
# for C1 C3 W2 W3 C2 W4 W1 W5 W6 B1 B1N in runs/graphs_v3; with CONV=mean / wmean as well,
# only C1 (the noise floor: the same network as GATv2's C1, in runs/graphs_v3_<conv>).
# Tuning jobs then get 8 hours. V3_APPR="C1" limits the v3 queue to those approaches,
# and names the new ones: V3=1 V3_APPR="B3 B4 B4F" (smoke, tuning, 5 seeds each).
# KSWEEP=B1 (with V3=1) queues the k sweep of choices.md instead: 5 seeds per task with
# KPARAM (default nbr_k) fixed to each value of KS (default "1 3 5 10 20") and the other
# settings as tuned in runs/graphs_v3, in runs/graphs_v3_k. For B3: KSWEEP=B3 KPARAM=keep_k
# KS="3 5 10 20 40".
set -uo pipefail
S=cluster/train_graph.slurm
CONV=${CONV:-}
V3=${V3:-}
KSWEEP=${KSWEEP:-}
KPARAM=${KPARAM:-nbr_k}
KS=${KS:-1 3 5 10 20}
if [[ -n $KSWEEP ]]; then
    [[ -n $V3 && -z $CONV ]] || { echo "KSWEEP needs V3=1 and no CONV" >&2; exit 1; }
    R=runs/graphs_v3_k
elif [[ -n $V3 ]]; then R=runs/graphs_v3${CONV:+_$CONV}; else R=runs/graphs_v2${CONV:+_$CONV}; fi
TASKS=${TASKS:-3_1 3_2 3_3}
MAX=${MAX:-4}                 # Perun: 4 jobs per user
POLL=${POLL:-180}             # seconds between checks
TRIES=${TRIES:-2}             # submissions per run before giving up on it
SEED_LIST=(42 43 44 45 46)
U=${USER:-$(whoami)}
DONE=logs/queue_graphs${V3:+_v3}${CONV:+_$CONV}${KSWEEP:+_k}.done
mkdir -p logs && touch "$DONE"

items=()                      # "name|ready check|VAR=VALUE ..."
add() { items+=("$1|$2|$3"); }
if [[ -n $KSWEEP ]]; then
    mkdir -p "$R/$KSWEEP"
    for T in $TASKS; do                        # the sweep reads setup v3's tuning of the approach
        src=runs/graphs_v3/$KSWEEP/tuning_${T}_e5.json
        [[ -f $src ]] || { echo "missing $src (tune $KSWEEP first)" >&2; exit 1; }
        cp "$src" "$R/$KSWEEP/"
    done
    for K in $KS; do
        for T in $TASKS; do
            add "${KSWEEP}_${T}_${KPARAM}${K}_seeds" "tuned $KSWEEP $T"                 "APPROACH=$KSWEEP TASK=$T HYPER=tuned TUNE_SEEDS=3 FIX=$KPARAM=$K RUNS_DIR=$R"
        done
    done
elif [[ -n $V3 ]]; then
    if [[ -n $CONV ]]; then V3_APPR="C1"; else V3_APPR=${V3_APPR:-C1 C3 W2 W3 C2 W4 W1 W5 W6 B1 B1N}; fi
    # order = priority when a slot frees: smoke runs first (a crash shows early),
    # then the 5 seeds of every approach already tuned, then new tunings
    for A in $V3_APPR; do                      # new approaches: a smoke run before their tuning
        case $A in W1|W5|W6|B3|B4|B4F) add "${A}_smoke" "none" "APPROACH=$A SMOKE=1 TUNE_SEEDS=3" ;; esac
    done
    for A in $V3_APPR; do
        for T in $TASKS; do
            add "${A}_${T}_seeds" "tuned $A $T" "APPROACH=$A TASK=$T HYPER=tuned TUNE_SEEDS=3 ${CONV:+CONV=$CONV}"
        done
    done
    for A in $V3_APPR; do
        case $A in W1|W5|W6|B3|B4|B4F) need="smoke $A" ;; *) need="none" ;; esac
        for T in $TASKS; do add "${A}_${T}_tune" "$need" "APPROACH=$A TASK=$T TUNE=20 TUNE_SEEDS=3 ${CONV:+CONV=$CONV}"; done
    done
elif [[ -n $CONV ]]; then
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
    local A T aud s fx
    A=$(var APPROACH "$@"); T=$(var TASK "$@"); aud=$([[ -n $(var AUDIO "$@") ]] && echo _audio)
    fx=$(var FIX "$@"); fx=${fx:+_${fx/=/}}       # train.py's run name: ..._tuned_e5_nbr_k5
    case $name in
        *_smoke) [[ -f $R/$A/${A}_3_1_seed42_f1_e5_smoke/manifest.json ]] ;;
        *_tune) [[ -f $R/$A/tuning_${T}_e5$aud.json ]] ;;
        *_seeds) for s in "${SEED_LIST[@]}"; do
                     [[ -f $R/$A/${A}_${T}_seed${s}_tuned_e5$aud$fx/manifest.json ]] || return 1
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
    local time=()
    [[ -n $V3 && " $* " == *" TUNE="* ]] && time=(--time=8:00:00)
    local log="logs/cmp_graph_%j_${name}${V3:+_v3}${CONV:+_$CONV}"
    env -u TUNE -u SMOKE -u HYPER -u SEEDS -u AUDIO -u APPROACH -u TASK -u EXTRA -u CONV -u TUNE_SEEDS -u FIX -u RUNS_DIR "$@" "${seeds[@]}" \
        sbatch --parsable "${time[@]}" -o "$log.out" -e "$log.err" "$S"
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
