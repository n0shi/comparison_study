#!/bin/bash
#SBATCH --job-name=cmp_asr_parakeet
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
# ASR bake-off, Parakeet side: the 30 videos of outputs/asr/bakeoff_ids.txt.
# Submit from the package root:  sbatch cluster/asr_parakeet.slurm
set -euo pipefail
PY=/mnt/project/perun2601420/envs/exist_nemo/bin/python
# same libstdc++ issue as the other venvs built on exist_vt's Python
export LD_PRELOAD=/mnt/home/rupyke723/miniconda3/envs/exist_vt/lib/libstdc++.so.6
export HF_HOME=/mnt/home/rupyke723/.cache/huggingface
export HF_HUB_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"$PY" src/asr_bakeoff.py parakeet --config config.perun.yaml
