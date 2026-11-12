#!/bin/bash
# Seed replica of C1 (choices.md, noise floor): the same network and the same
# setup-v3 tuning as runs/graphs_v3/C1, trained with seeds 47-51 instead of
# 42-46, in runs/graphs_v3_rep. One job per task (3 of Perun's 4 slots).
# Run from the package root:
#     bash cluster/rep_c1.sh
# When the three jobs are done:
#     nohup bash cluster/rep_c1.sh report > results/rep_c1.txt 2>&1 &
set -euo pipefail
P=${PY:-/mnt/home/rupyke723/miniconda3/envs/exist_vt/bin/python}
SRC=runs/graphs_v3/C1
DST=runs/graphs_v3_rep/C1
if [[ ${1:-} == report ]]; then
    for T in 3_1 3_2 3_3; do
        n=$(ls -d $DST/C1_${T}_seed*_tuned_e5/manifest.json 2>/dev/null | wc -l)
        [[ $n == 5 ]] || { echo "task $T: $n of 5 seed runs finished in $DST" >&2; exit 1; }
    done
    "$P" src/graph_report.py --config config.perun.yaml --runs runs/graphs_v3_rep --out results/graphs_v3_rep --n-boot 0
    "$P" src/sharp_check.py --config config.perun.yaml --results results/graphs_v3 \
        --replicas results/graphs_v3_rep results/graphs_v3_mean results/graphs_v3_wmean --only-replicas
    exit 0
fi
mkdir -p "$DST" logs
for T in 3_1 3_2 3_3; do
    [[ -f $SRC/tuning_${T}_e5.json ]] || { echo "missing $SRC/tuning_${T}_e5.json" >&2; exit 1; }
    cp "$SRC/tuning_${T}_e5.json" "$DST/"
    APPROACH=C1 TASK=$T HYPER=tuned TUNE_SEEDS=3 SEEDS="47 48 49 50 51" EXTRA="--runs-dir runs/graphs_v3_rep" \
        sbatch -o "logs/cmp_graph_%j_C1_${T}_rep.out" -e "logs/cmp_graph_%j_C1_${T}_rep.err" cluster/train_graph.slurm
done
