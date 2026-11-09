#!/bin/bash
# Setup v3 noise floor (choices.md): seed ensembles of C1 under GATv2, mean and
# wmean (graph_report.py, no bootstrap), then C1 against C1 (conv_check.py).
# Run detached from the package root:
#     nohup bash cluster/conv_check_v3.sh > results/conv_check_v3.txt 2>&1 &
set -euo pipefail
P=${PY:-/mnt/home/rupyke723/miniconda3/envs/exist_vt/bin/python}
for s in "" _mean _wmean; do
    echo "== graph_report runs/graphs_v3$s"
    "$P" src/graph_report.py --config config.perun.yaml --runs "runs/graphs_v3$s" --out "results/graphs_v3$s" --n-boot 0
done
echo "== conv_check --setup v3"
"$P" src/conv_check.py --config config.perun.yaml --n-boot 1000 --setup v3
