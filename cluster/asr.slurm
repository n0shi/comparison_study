#!/bin/bash
#SBATCH --job-name=cmp_asr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --array=0-3
# Speech with word timestamps for all 3198 videos, in 4 shards.
# Submit from the package root:
#     LIMIT=5 sbatch --array=0 --time=0:30:00 cluster/asr.slurm   # smoke: 5 videos
#     sbatch cluster/asr.slurm                                      # everything
#     N_SHARDS=1 IDS=outputs/asr/review_ids.txt OUT=outputs/asr/review_whisper.jsonl \
#         sbatch --array=0 --time=1:00:00 cluster/asr.slurm          # a fixed sample of ids
# Re-submitting resumes: videos already in a shard's file are skipped.

set -euo pipefail  # first, so a failed library lookup stops the job
PY=/mnt/project/perun2601420/envs/exist_asr/bin/python
# The venvs are built on exist_vt's conda Python, whose sqlite3 (imported by
# vLLM and others) links ICU, which needs a newer libstdc++ than /lib64 has.
export LD_PRELOAD=/mnt/home/rupyke723/miniconda3/envs/exist_vt/lib/libstdc++.so.6
N_SHARDS=${N_SHARDS:-4}

export HF_HOME=/mnt/home/rupyke723/.cache/huggingface
export HF_HUB_OFFLINE=1
# faster-whisper (CTranslate2) looks for cuDNN / cuBLAS on the system path; the
# ones that match come with torch's pip wheels.
# (nvidia.* are namespace packages: no __file__, so use __path__)
NV_LIBS="$("$PY" -c 'import nvidia.cublas.lib as a, nvidia.cudnn.lib as b; print(a.__path__[0] + ":" + b.__path__[0])')"
export LD_LIBRARY_PATH="$NV_LIBS:${LD_LIBRARY_PATH:-}"

cd "$SLURM_SUBMIT_DIR"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

"$PY" src/s04_asr.py --config config.perun.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --n-shards "$N_SHARDS" ${LIMIT:+--limit "$LIMIT"} \
    ${IDS:+--ids-file "$IDS"} ${OUT:+--out "$OUT"}
