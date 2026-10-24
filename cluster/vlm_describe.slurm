#!/bin/bash
#SBATCH --job-name=cmp_vlm_describe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
# Stage 2 of the description regeneration: Qwen3-VL-30B through vLLM.
# Submit from the package root:
#     SMOKE=1 sbatch --time=1:00:00 cluster/vlm_describe.slurm   # frames_smoke.h5 -> descriptions_smoke.jsonl
#     sbatch cluster/vlm_describe.slurm                          # everything; re-submitting resumes
set -euo pipefail
PY=/mnt/project/perun2601420/envs/exist_vllm011/bin/python
# The venvs are built on exist_vt's conda Python, whose sqlite3 (imported by
# vLLM and others) links ICU, which needs a newer libstdc++ than /lib64 has.
export LD_PRELOAD=/mnt/home/rupyke723/miniconda3/envs/exist_vt/lib/libstdc++.so.6
export HF_HOME=/mnt/home/rupyke723/.cache/huggingface
export HF_HUB_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

if [[ -n "${SMOKE:-}" ]]; then
    "$PY" src/vlm_describe.py --config config.perun.yaml \
        --frames outputs/vlm/frames_smoke.h5 \
        --out "${SMOKE_OUT:-outputs/vlm/descriptions_smoke.jsonl}" ${EXTRA:-}
else
    "$PY" src/vlm_describe.py --config config.perun.yaml
fi
