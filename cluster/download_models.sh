#!/bin/bash
# Download every model the comparison needs into the shared HF cache.
# Run on the LOGIN node (compute nodes run with HF_HUB_OFFLINE=1):
#     bash cluster/download_models.sh            # all
#     bash cluster/download_models.sh qwen       # one group
# Already-cached files are skipped, so re-running is safe.
set -euo pipefail

export PATH="/mnt/home/rupyke723/miniconda3/envs/exist_vt/bin:$PATH"
export HF_HOME=/mnt/home/rupyke723/.cache/huggingface
unset HF_HUB_OFFLINE

GROUP="${1:-all}"

# Console scripts in exist_vt (pip, huggingface-cli) have shebangs pointing at
# the dead /mnt/data/home prefix, so everything goes through `python -c`.
dl() {  # repo_id
    echo "=== $1"
    python -c "from huggingface_hub import snapshot_download as s; print(s('$1'))" || { echo "FAILED: $1"; exit 1; }
}

if [[ "$GROUP" == all || "$GROUP" == text ]]; then
    dl BAAI/bge-m3                                # video vectors (Step 6)
    # xlm-roberta-large and videomae-base are already cached (ENVIRONMENT.md)
fi
if [[ "$GROUP" == all || "$GROUP" == vision ]]; then
    dl google/siglip2-so400m-patch14-384          # keyframes, grounding edges (Steps 6, 9)
fi
if [[ "$GROUP" == all || "$GROUP" == audio ]]; then
    dl Systran/faster-whisper-large-v3            # ASR with word timestamps (Step 5)
    dl openai/whisper-large-v3                    # encoder features for audio nodes (Step 23)
    dl jonatasgrosman/wav2vec2-large-xlsr-53-spanish   # WhisperX alignment, ES
    # WhisperX's English aligner (torchaudio WAV2VEC2_ASR_BASE_960H) is fetched
    # from download.pytorch.org into the torch hub cache. torchaudio lives in
    # the ASR venv, not in exist_vt.
    ASR_PY=/mnt/project/perun2601420/envs/exist_asr/bin/python
    [[ -x "$ASR_PY" ]] || { echo "FAILED: create the exist_asr venv first (README step 5)"; exit 1; }
    # urllib in the venv does not find the system CA bundle (CERTIFICATE_VERIFY_FAILED);
    # point it at certifi's bundle, which the HF downloads already use.
    export SSL_CERT_FILE="$("$ASR_PY" -c 'import certifi; print(certifi.where())')"
    "$ASR_PY" -c "import torchaudio; torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model(); print('torchaudio WAV2VEC2_ASR_BASE_960H cached')"         || { echo "FAILED: torchaudio English aligner"; exit 1; }
fi
if [[ "$GROUP" == all || "$GROUP" == qwen ]]; then
    dl Qwen/Qwen3-VL-30B-A3B-Instruct             # W3 / B2 judgments (~60 GB)
fi

echo "done. Cache size:"
du -sh "$HF_HOME"
# GraphPFN (B5) is not on the HF hub under a known id: follow
# https://github.com/yandex-research/graphpfn README for its checkpoint.
