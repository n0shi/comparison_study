#!/bin/bash
# k sweep (choices.md): seed ensembles of runs/graphs_v3_k (graph_report.py, no
# bootstrap), then the curve against k (k_sweep_report.py).
# Run detached from the package root:
#     nohup bash cluster/k_sweep_report.sh > results/k_sweep_B1.txt 2>&1 &
#     APPROACH=B3 PARAM=keep_k nohup bash cluster/k_sweep_report.sh > results/k_sweep_B3.txt 2>&1 &
set -euo pipefail
P=${PY:-/mnt/home/rupyke723/miniconda3/envs/exist_vt/bin/python}
"$P" src/graph_report.py --config config.perun.yaml --runs runs/graphs_v3_k --out results/graphs_v3_k --n-boot 0
"$P" src/k_sweep_report.py --config config.perun.yaml --approach "${APPROACH:-B1}" --param "${PARAM:-nbr_k}"
