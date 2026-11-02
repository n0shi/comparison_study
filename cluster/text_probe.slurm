#!/bin/bash
#SBATCH --job-name=cmp_probe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
# Linear probes: how much each input carries on its own, per text encoder.
# Needs the feature files (s06_features.py); bge_text.h5 / e5.h5 are used if
# present, so re-run after `sbatch --array=5-6 cluster/features.slurm`.
# Submit from the package root:
#     sbatch cluster/text_probe.slurm
#     EXTRA="--only w3:links bge:desc" sbatch cluster/text_probe.slurm
export PATH="/mnt/home/rupyke723/miniconda3/envs/exist_vt/bin:$PATH"
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python src/text_probe.py --config config.perun.yaml --jobs "$SLURM_CPUS_PER_TASK" ${EXTRA:-}
