#!/bin/bash
#SBATCH --job-name=cmp_c0
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-29%4
# C0 = the paper's cross-attention model on our author-grouped folds.
# 2 modality sets x 3 tasks x 5 seeds = 30 array tasks, each runs 5 folds.
# Submit from the package root:
#     sbatch cluster/c0_train.slurm                       # all 30
#     sbatch --array=0 cluster/c0_train.slurm             # just quad / 3_1 / seed 42
#     sbatch --array=30-44%4 cluster/c0_train.slurm       # text-only control (vlm + trn), 15 runs
#     SMOKE=1 sbatch --array=0 --time=0:45:00 cluster/c0_train.slurm   # smoke test

# ---- the part-1 runs behind the paper's quad model ----------------------------
# 3_1: 164307 (dirichlet_alpha 25; its alpha-15 twin 164238 scored lower in CV).
# 3_2: 184649 (2 fusion layers, Dirichlet 8, smoothing 0.05, as the paper
#      describes; its twin 184730 has 1 layer, no Dirichlet, smoothing 0.15).
#      It also uses mixup (alpha 0.2, p 0.4), which the paper does not mention.
# 3_3: 202433, the only quad run for this task.
# Their config_resolved.json files are copied verbatim into configs/c0/.
PART1_ROOT=/mnt/project/perun2601420/vlm_quad/vlm_quad
declare -A PART1_CONFIG=(
    [3_1]=configs/c0/task3_1.json
    [3_2]=configs/c0/task3_2.json
    [3_3]=configs/c0/task3_3.json
)
# -------------------------------------------------------------------------------

MODS=(quad trio text); TASKS=(3_1 3_2 3_3); SEEDS=(42 43 44 45 46)
i=${SLURM_ARRAY_TASK_ID}
MOD=${MODS[$(( i / 15 ))]}
TASK=${TASKS[$(( (i / 5) % 3 ))]}
SEED=${SEEDS[$(( i % 5 ))]}

export PATH="/mnt/home/rupyke723/miniconda3/envs/exist_vt/bin:$PATH"
export HF_HOME=/mnt/home/rupyke723/.cache/huggingface
export HF_HUB_OFFLINE=1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "array task $i -> modalities=$MOD task=$TASK seed=$SEED"

python src/c0_part1_folds.py --config config.perun.yaml \
    --part1-root "$PART1_ROOT" --part1-config "${PART1_CONFIG[$TASK]}" \
    --task "$TASK" --seed "$SEED" --modalities "$MOD" --runs-dir runs/C0 \
    ${SMOKE:+--smoke}
