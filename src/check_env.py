"""Environment check for the cluster: packages, GPU, data paths, cached models.

Prints a table of OK / MISSING and exits non-zero if anything required for
the given stage is missing, so a broken environment is caught by a 1-minute
job instead of a multi-hour one.

Usage:
    python src/check_env.py --config config.yaml --stage all
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path

from common import load_config, setup_stdout

PACKAGES = {
    # stage -> importable modules
    "eval": ["numpy", "pandas", "scipy", "sklearn", "h5py", "pyevall", "yaml"],
    "data": ["av", "imagehash", "PIL"],
    "asr": ["torch", "torchaudio", "faster_whisper", "whisperx", "silero_vad"],
    "features": ["torch", "transformers", "sentence_transformers", "faiss"],
    "graph": ["torch", "torch_geometric", "lightgbm"],
    "vllm": ["vllm"],
}

MODELS = {
    "features": ["FacebookAI/xlm-roberta-large", "MCG-NJU/videomae-base", "BAAI/bge-m3",
                 "google/siglip2-so400m-patch14-384"],
    "asr": ["Systran/faster-whisper-large-v3", "openai/whisper-large-v3",
            "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"],
    "vllm": ["Qwen/Qwen3-VL-30B-A3B-Instruct"],
}

PATH_KEYS = ["train_json", "test_json", "gold_dir", "train_h5", "test_h5",
             "train_videos_dir", "test_videos_dir"]


def check_import(mod: str) -> str:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "ok")
    except Exception as e:
        return f"MISSING ({type(e).__name__}: {str(e)[:60]})"


def cached_models() -> set[str]:
    try:
        from huggingface_hub import scan_cache_dir
        return {r.repo_id for r in scan_cache_dir().repos}
    except Exception as e:
        print(f"  cannot scan HF cache: {e}")
        return set()


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", default="eval",
                    help=f"comma list of {sorted(PACKAGES)} or 'all'")
    a = ap.parse_args()
    stages = sorted(PACKAGES) if a.stage == "all" else a.stage.split(",")
    bad = []

    print(f"python {platform.python_version()} on {platform.node()}")
    print(f"HF_HOME={os.environ.get('HF_HOME')} HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")

    print("\n[packages]")
    for st in stages:
        for mod in PACKAGES[st]:
            v = check_import(mod)
            print(f"  {st:9s} {mod:22s} {v}")
            if v.startswith("MISSING"):
                bad.append(f"package {mod} ({st})")

    if "torch" in sys.modules or any("torch" in PACKAGES[s] for s in stages):
        try:
            import torch
            print(f"\n[gpu] torch {torch.__version__}, cuda {torch.version.cuda}, "
                  f"available={torch.cuda.is_available()}")
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  {i}: {p.name} {p.total_memory / 2**30:.0f} GiB")
        except Exception as e:
            print(f"[gpu] torch unavailable: {e}")

    print("\n[data paths]")
    cfg = load_config(a.config)
    for k in PATH_KEYS:
        p = cfg.path(k)
        ok = p.exists()
        extra = f"{len(list(p.glob('*.mp4')))} mp4" if ok and p.is_dir() and "videos" in k else ""
        print(f"  {k:18s} {'OK     ' if ok else 'MISSING'} {p} {extra}")
        if not ok:
            bad.append(f"path {k}")
    for k in ("videos.csv", "folds.csv"):
        ok = (cfg.path("out_dir") / k).is_file()
        print(f"  outputs/{k:10s} {'OK' if ok else 'MISSING (copy from local or rerun s01-s03)'}")
        if not ok:
            bad.append(f"outputs/{k}")

    want = [m for st in stages for m in MODELS.get(st, [])]
    if want:
        print("\n[HF cache]")
        have = cached_models()
        for m in want:
            ok = m in have
            print(f"  {'OK     ' if ok else 'MISSING'} {m}")
            if not ok:
                bad.append(f"model {m}")

    print("\n" + ("ALL OK" if not bad else "PROBLEMS:\n  " + "\n  ".join(bad)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
