#!/bin/bash
#SBATCH --job-name=cmp_w3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
# W3: Qwen3-VL-30B links each description field to the moments (frames and
# speech) that support it. Own job name and own output folder (outputs/w3/);
# reads only finished inputs (v3 JSONs, moments, H5 clips), so it does not
# touch the C0 or feature jobs.
# Submit from the package root:
#     SMOKE=1 sbatch --time=1:00:00 cluster/w3_edges.slurm   # 20 videos, prompt v4 -> outputs/w3/edges_smoke_v4.jsonl (compared with v3)
#     SMOKE=1 PROMPT=v1 sbatch ...                           # the first prompt -> outputs/w3/edges_smoke.jsonl
#     sbatch cluster/w3_edges.slurm                          # every video; re-submitting resumes
set -euo pipefail
PY=/mnt/project/perun2601420/envs/exist_vllm011/bin/python
# The venvs are built on exist_vt's conda Python, whose sqlite3 (imported by
# vLLM and others) links ICU, which needs a newer libstdc++ than /lib64 has.
export LD_PRELOAD=/mnt/home/rupyke723/miniconda3/envs/exist_vt/lib/libstdc++.so.6
export HF_HOME=/mnt/home/rupyke723/.cache/huggingface
export HF_HUB_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

PROMPT=${PROMPT:-v4}
declare -A PREV=([v2]=outputs/w3/edges_smoke.jsonl [v3]=outputs/w3/edges_smoke_v2.jsonl [v4]=outputs/w3/edges_smoke_v3.jsonl)
if [[ -n "${SMOKE:-}" ]]; then
    # v1's smoke is outputs/w3/edges_smoke.jsonl; later versions get their own file
    OUT=outputs/w3/edges_smoke_${PROMPT}.jsonl
    [[ "$PROMPT" == v1 ]] && OUT=outputs/w3/edges_smoke.jsonl
    "$PY" src/w3_vlm_edges.py --config config.perun.yaml --prompt "$PROMPT" --sample 20         --out "$OUT" ${EXTRA:-}
    "$PY" src/w3_vlm_edges.py --config config.perun.yaml --summary "$OUT"
    CMP=""
    P="${PREV[$PROMPT]:-}"
    [[ -n "$P" && -f "$P" ]] && CMP="--compare $P"
    "$PY" src/w3_vlm_edges.py --config config.perun.yaml --review "$OUT" $CMP
else
    "$PY" src/w3_vlm_edges.py --config config.perun.yaml --prompt "$PROMPT" ${EXTRA:-}
fi
