#!/bin/bash
#SBATCH --job-name=cmp_features
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-4
# Step 6: frozen-encoder features for all 3198 videos, one encoder per array
# task, each into its own outputs/features/<stage>.h5.
# Submit from the package root:
#     SMOKE=1 sbatch --time=0:30:00 cluster/features.slurm   # 8 videos -> outputs/features_smoke
#     sbatch cluster/features.slurm                          # all five encoders
#     sbatch --array=2 cluster/features.slurm                # just whisper
# Re-submitting resumes (videos already in the h5 are skipped). For a clean
# re-run delete that stage's h5 first.
STAGES=(xlmr videomae whisper bge siglip)
STAGE=${STAGES[${SLURM_ARRAY_TASK_ID}]}

export PATH="/mnt/home/rupyke723/miniconda3/envs/exist_vt/bin:$PATH"
export HF_HOME=/mnt/home/rupyke723/.cache/huggingface
export HF_HUB_OFFLINE=1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "array task ${SLURM_ARRAY_TASK_ID} -> stage=$STAGE"

if [[ -n "${SMOKE:-}" ]]; then
    python src/s06_features.py --config config.perun.yaml --stage "$STAGE" \
        --limit 8 --out-dir outputs/features_smoke
else
    python src/s06_features.py --config config.perun.yaml --stage "$STAGE"
fi
