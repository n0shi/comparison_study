#!/bin/bash
#SBATCH --job-name=cmp_check_env
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
# Submit from the package root:  sbatch cluster/check_env.slurm
# Checks every environment, then runs the evaluation selftest (CPU, ~5 min).

export HF_HOME=/mnt/home/rupyke723/.cache/huggingface
export HF_HUB_OFFLINE=1
VT=/mnt/home/rupyke723/miniconda3/envs/exist_vt/bin/python
VENVS=/mnt/project/perun2601420/envs

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

status=0
for spec in "$VT:eval,data,features,graph" "$VENVS/exist_asr/bin/python:asr" "$VENVS/exist_vllm011/bin/python:vllm"; do
    py="${spec%%:*}"; stages="${spec##*:}"
    echo; echo "################ $py ($stages)"
    if [[ ! -x "$py" ]]; then
        echo "MISSING interpreter $py"; status=1; continue
    fi
    "$py" src/check_env.py --config config.perun.yaml --stage "$stages" || status=1
done

echo; echo "################ evaluation selftest (exist_vt)"
"$VT" src/evaluate.py --config config.perun.yaml selftest || status=1
exit $status
