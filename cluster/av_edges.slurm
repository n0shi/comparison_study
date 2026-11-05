#!/bin/bash
#SBATCH --job-name=cmp_av
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
# W3 smoke with audio-visual models (src/av_edges.py): the model watches each
# whole video with its sound and scores the moments per field, as W3's prompt v4.
# 20 train videos (W3's smoke sample), compared with W3's links (Qwen3-VL, one
# frame per moment) on the same videos. Writes only outputs/w3/edges_smoke_av_*.
# Setup: cluster/README_perun.md, step 10. Submit from the package root:
#     MODEL=omni    sbatch cluster/av_edges.slurm
#     MODEL=salmonn sbatch cluster/av_edges.slurm
set -euo pipefail
MODEL=${MODEL:-omni}
ENVS=/mnt/project/perun2601420/envs
case "$MODEL" in
    omni)    PY=$ENVS/exist_vllm011/bin/python; ARGS="" ;;
    salmonn) PY=$ENVS/exist_salmonn/bin/python
             ARGS="--salmonn-repo /mnt/project/perun2601420/ext/video-SALMONN-2" ;;
    *) echo "MODEL must be omni or salmonn, got $MODEL"; exit 1 ;;
esac
export LD_PRELOAD=/mnt/home/rupyke723/miniconda3/envs/exist_vt/lib/libstdc++.so.6
export HF_HOME=/mnt/home/rupyke723/.cache/huggingface
export HF_HUB_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

OUT=outputs/w3/edges_smoke_av_${MODEL}.jsonl
"$PY" -u src/av_edges.py --config config.perun.yaml --model "$MODEL" --sample 20 --out "$OUT" \
    --compare outputs/w3/edges.jsonl $ARGS ${EXTRA:-}
"$PY" src/w3_vlm_edges.py --config config.perun.yaml --review "$OUT" --compare outputs/w3/edges.jsonl
