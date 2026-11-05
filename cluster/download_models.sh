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
    dl intfloat/multilingual-e5-large             # alternative text node features (comparison)
    dl intfloat/multilingual-e5-small             # cheap text encoder (encoder check, choices.md)
    dl minishlab/potion-multilingual-128M         # Model2Vec static embeddings (encoder check)
    # xlm-roberta-large and videomae-base are already cached (ENVIRONMENT.md)
fi
if [[ "$GROUP" == all || "$GROUP" == vision ]]; then
    dl google/siglip2-so400m-patch14-384          # keyframes, grounding edges (Steps 6, 9)
fi
if [[ "$GROUP" == all || "$GROUP" == audio ]]; then
    dl Systran/faster-whisper-large-v3            # ASR with word timestamps (Step 5)
    dl openai/whisper-large-v3                    # encoder features for audio nodes (Step 23)
    dl jonatasgrosman/wav2vec2-large-xlsr-53-spanish   # WhisperX alignment, ES
    # WhisperX aligns English and Spanish with torchaudio pipelines, fetched
    # from download.pytorch.org into the torch hub cache (compute nodes cannot
    # fetch them: SSL verification fails there). torchaudio lives in the ASR
    # venv, and urllib in it needs certifi's CA bundle.
    ASR_PY=/mnt/project/perun2601420/envs/exist_asr/bin/python
    [[ -x "$ASR_PY" ]] || { echo "FAILED: create the exist_asr venv first (README step 5)"; exit 1; }
    export SSL_CERT_FILE="$("$ASR_PY" -c 'import certifi; print(certifi.where())')"
    # WhisperX splits sentences with NLTK punkt during alignment; ~/nltk_data
    # is on NLTK's default search path, so compute nodes find it.
    "$ASR_PY" -c "import nltk; [nltk.download(p, download_dir='$HOME/nltk_data', quiet=True) for p in ('punkt', 'punkt_tab')]; print('nltk punkt cached')"         || { echo "FAILED: nltk punkt"; exit 1; }
    for P in WAV2VEC2_ASR_BASE_960H VOXPOPULI_ASR_BASE_10K_ES; do
        "$ASR_PY" -c "import torchaudio; torchaudio.pipelines.$P.get_model(); print('torchaudio $P cached')"             || { echo "FAILED: torchaudio $P"; exit 1; }
    done
fi
if [[ "$GROUP" == all || "$GROUP" == qwen ]]; then
    dl Qwen/Qwen3-VL-30B-A3B-Instruct             # W3 / B2 judgments (~60 GB)
fi

if [[ "$GROUP" == all || "$GROUP" == av ]]; then
    dl Qwen/Qwen3-Omni-30B-A3B-Instruct           # audio-visual W3 smoke (av_edges.py, ~65 GB)
    dl tsinghua-ee/video-SALMONN2_plus_7B_full    # audio-visual W3 smoke (~17 GB)
    # the authors' inference code (Apache-2.0); av_edges.py imports it by path
    EXT=/mnt/project/perun2601420/ext
    mkdir -p "$EXT"
    [[ -d "$EXT/video-SALMONN-2" ]] || git clone --depth 1 https://github.com/bytedance/video-SALMONN-2 "$EXT/video-SALMONN-2"
    (cd "$EXT/video-SALMONN-2" && echo "video-SALMONN-2 at $(git rev-parse --short HEAD)")
fi

echo "done. Cache size:"
du -sh "$HF_HOME"
# GraphPFN (B5) is not on the HF hub under a known id: follow
# https://github.com/yandex-research/graphpfn README for its checkpoint.
