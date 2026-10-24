"""Sample the VLM frames of every video once -> outputs/vlm/frames.h5.

The part-1 descriptions came from three sources (two models, two prompt
versions, two sampler versions; data_note.md), so they are regenerated with
one sampler, one model and one prompt. This is stage 1: frames, on CPU.

The sampler is part 1's own (src/vlm/frame_sampler.py, copied verbatim),
called exactly as its controller called it, including the adaptive frame cap
by transcript length, counted on the transcript stage 2 sends
(`vlm_transcript` of s05_moments.py: our ASR when reliable, else the
organisers' cleaned text). Frames are stored as the JPEG bytes that are sent to
the model, so stage 2 and the later W3 prompts see identical images.

h5 layout: /<id_EXIST>/jpeg  (vlen uint8, one JPEG per frame)
           attrs: timestamps, sources, max_frames, id_Tiktok, lang, split

Usage:
    python src/vlm_frames.py --config config.perun.yaml --workers 32
    python src/vlm_frames.py --config config.perun.yaml --limit 8 --out outputs/vlm/frames_smoke.h5
"""
from __future__ import annotations

import argparse
import base64
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from common import load_config, load_moments, require, setup_stdout

# part 1's ModelConfig defaults (vlm_provider/config.py), as the controller used them
SAMPLER = dict(min_frames=6, uniform_interval_sec=2.0, use_scene_detect=True,
               scene_detect_threshold=27.0, dedupe_hamming_threshold=4, max_frame_side=560)
MAX_FRAMES = 14
JPEG_QUALITY = 85


def adaptive_max_frames(transcript: str) -> int:
    """vlm_provider/controler.py::_adaptive_max_frames."""
    words = len((transcript or "").split())
    if words > 300:
        return min(MAX_FRAMES, 6)
    if words > 150:
        return min(MAX_FRAMES, 8)
    return MAX_FRAMES


def sample_one(job: tuple) -> tuple:
    id_exist, path, transcript = job
    from vlm.frame_sampler import frame_to_base64_jpeg, sample_frames
    cap = adaptive_max_frames(transcript)
    try:
        frames = sample_frames(video_path=path, max_frames=cap, **SAMPLER)
    except Exception as e:  # recorded, then the run fails loudly at the end
        return id_exist, None, f"{type(e).__name__}: {e}", cap
    jpegs = [base64.b64decode(frame_to_base64_jpeg(f.image, JPEG_QUALITY)) for f in frames]
    meta = {"timestamps": [round(f.timestamp, 2) for f in frames],
            "sources": [f.source for f in frames]}
    return id_exist, (jpegs, meta), None, cap


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None, help="default: outputs/vlm/frames.h5")
    ap.add_argument("--limit", type=int, default=0, help="first N train videos (4 EN + 4 ES style smoke)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--moments", default=None, help="default: outputs/moments/moments.jsonl")
    a = ap.parse_args()
    cfg = load_config(a.config)

    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str, "id_Tiktok": str})
    if a.limit:  # balanced smoke sample: half EN, half ES, train only
        tr = vids[vids.split == "train"]
        vids = pd.concat([tr[tr.lang == "en"].head(a.limit // 2), tr[tr.lang == "es"].head(a.limit - a.limit // 2)])
    moments = load_moments(cfg, a.moments)
    require(set(vids.id_EXIST) <= set(moments), "every video has its moments",
            expected=len(vids), found=len(set(vids.id_EXIST) & set(moments)))
    # the cap counts the words the model will see
    text = {k: m["vlm_transcript"] for k, m in moments.items()}
    dirs = {"train": cfg.path("train_videos_dir"), "test": cfg.path("test_videos_dir")}

    out = Path(a.out) if a.out else cfg.path("out_dir") / "vlm" / "frames.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "a") as h:
        done = set(h.keys())
    jobs = []
    for r in vids.itertuples():
        if r.id_EXIST in done:
            continue
        p = dirs[r.split] / f"{r.id_Tiktok}.mp4"
        require(p.is_file(), "video file exists", expected=str(p), found="missing")
        jobs.append((r.id_EXIST, str(p), text[r.id_EXIST]))
    meta_by_id = vids.set_index("id_EXIST")
    print(f"{len(vids)} videos, {len(done)} already sampled, {len(jobs)} to go -> {out}")

    failed, t0 = {}, time.time()
    vlen = h5py.vlen_dtype(np.dtype("uint8"))
    with Pool(a.workers) as pool, h5py.File(out, "a") as h:
        for n, (vid, res, err, cap) in enumerate(pool.imap_unordered(sample_one, jobs, chunksize=4), 1):
            if res is None or not res[0]:
                failed[vid] = err or "no frames"
                continue
            jpegs, meta = res
            g = h.create_group(vid)
            ds = g.create_dataset("jpeg", (len(jpegs),), dtype=vlen)
            for i, b in enumerate(jpegs):
                ds[i] = np.frombuffer(b, dtype=np.uint8)
            g.attrs["timestamps"] = np.asarray(meta["timestamps"], dtype=float)
            g.attrs["sources"] = np.asarray(meta["sources"], dtype="S8")
            g.attrs["max_frames"] = cap
            m = meta_by_id.loc[vid]
            g.attrs["id_Tiktok"], g.attrs["lang"], g.attrs["split"] = m.id_Tiktok, m.lang, m.split
            if n % 100 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n}/{len(jobs)}  {el / n:.2f} s/video", flush=True)

    with h5py.File(out, "r") as h:
        n_frames = np.array([h[k]["jpeg"].shape[0] for k in h])
        print(f"sampled videos: {len(n_frames)}; frames per video: min {n_frames.min()}, "
              f"median {np.median(n_frames):.0f}, max {n_frames.max()}")
    require(not failed, "every video yielded frames", expected=0,
            found=f"{len(failed)} failed: {list(failed.items())[:5]}")


if __name__ == "__main__":
    main()
