#!/bin/bash
#SBATCH --job-name=cmp_vlm_frames
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=4:00:00
# Stage 1 of the description regeneration: sample frames for all 3198 videos.
# Submit from the package root:  sbatch cluster/vlm_frames.slurm
# Re-submitting resumes (videos already in frames.h5 are skipped).
set -euo pipefail
PY=/mnt/project/perun2601420/envs/exist_vllm011/bin/python
# The venvs are built on exist_vt's conda Python, whose sqlite3 (imported by
# vLLM and others) links ICU, which needs a newer libstdc++ than /lib64 has.
export LD_PRELOAD=/mnt/home/rupyke723/miniconda3/envs/exist_vt/lib/libstdc++.so.6
cd "$SLURM_SUBMIT_DIR"
"$PY" src/vlm_frames.py --config config.perun.yaml --workers "$SLURM_CPUS_PER_TASK"
