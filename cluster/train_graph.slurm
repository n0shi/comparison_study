#!/bin/bash
#SBATCH --job-name=cmp_graph
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
# Graph approaches inside a video (train.py). The array index picks the
# approach (APPROACHES below; no array: index 0), or APPROACH names it for a
# single job (Perun allows only 3 active array jobs; cluster/queue_graphs.sh
# submits single jobs). The rest comes from the environment:
#   TASK   3_1 | 3_2 | 3_3                      (default 3_1)
#   TUNE   number of trials: tune on the inner folds, no outer predictions
#   SEEDS  seeds to train one after the other    (default 42)
#   HYPER  default | tuned                       (default default)
#   FOLDS  outer folds                           (default 1 2 3 4 5)
#   TEXT   xlmr | bge_text | e5: text node encoder (default e5, setup v2; runs/graphs_v2)
#   AUDIO  1 adds the Whisper audio-window nodes (Step 23; default off)
# Submit from the package root:
#     SMOKE=1 sbatch cluster/train_graph.slurm                                  # W2, 3.1: 300 videos, 2 epochs
#     TUNE=20 TASK=3_1 sbatch --array=0-3 cluster/train_graph.slurm             # tune W2, C1, C2, C3 (~45 min each)
#     TUNE=20 TASK=3_1 sbatch --array=4 cluster/train_graph.slurm               # tune W3 (VLM edges)
#     HYPER=tuned SEEDS="42 43 44 45 46" TASK=3_1 sbatch --array=0-3 cluster/train_graph.slurm
APPROACHES=(W2 C1 C2 C3 W3 W4)
IDX=${SLURM_ARRAY_TASK_ID:-0}
APPROACH=${APPROACH:-${APPROACHES[$IDX]}}
TASK=${TASK:-3_1}; SEEDS=${SEEDS:-42}; HYPER=${HYPER:-default}; FOLDS=${FOLDS:-1 2 3 4 5}; TEXT=${TEXT:-e5}
PART1_ROOT=/mnt/project/perun2601420/vlm_quad/vlm_quad

export PATH="/mnt/home/rupyke723/miniconda3/envs/exist_vt/bin:$PATH"
export HF_HUB_OFFLINE=1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
COMMON=(--config config.perun.yaml --part1-root "$PART1_ROOT" --approach "$APPROACH" --task "$TASK" --folds $FOLDS --text-features "$TEXT" ${AUDIO:+--audio})

if [[ -n "${TUNE:-}" ]]; then
    echo "array task $IDX -> tune approach=$APPROACH task=$TASK trials=$TUNE text=$TEXT audio=${AUDIO:-0}"
    python src/train.py "${COMMON[@]}" --tune "$TUNE" --skip-done ${SMOKE:+--smoke} ${EXTRA:-}
else
    for SEED in $SEEDS; do
        echo "array task $IDX -> approach=$APPROACH task=$TASK seed=$SEED hyper=$HYPER text=$TEXT audio=${AUDIO:-0}"
        python src/train.py "${COMMON[@]}" --seed "$SEED" --hyper "$HYPER" --skip-done ${SMOKE:+--smoke} ${EXTRA:-}
    done
fi
