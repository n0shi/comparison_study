"""Cut every video into 4 moments and attach the text each one carries.

Per video (-> outputs/moments/moments.jsonl):

  windows   4 time windows, one H5 clip in each. Clip k starts at
            k * (T - span) / 3 s (span = 64 source frames); the boundary
            between windows k and k+1 starts halfway between the clip centres
            and moves to the nearest pause in the speech within SNAP_S, so a
            phrase is not cut in two. It never crosses a clip centre.
  reliable  our ASR is trusted when it has >= MIN_WORDS timed words and
            >= MIN_SPEECH of the video is speech (VAD). Agreed after the
            40-video review; about 3 in 4 videos with speech pass.
  speech    per window, the ASR words whose midpoint falls in it (empty for
            unreliable ASR: there is then no timed text, only the organisers').
  asr_text  all kept ASR text (reliable videos only).
  org_text  the organisers' `text`, cleaned (src/text_clean.py) and kept whole:
            the review showed that what it adds beyond the speech is mostly an
            LLM's account of the video, not on-screen text, so it is one untimed
            document rather than split into parts.
  vlm_transcript  what the VLM prompt's transcript slot gets: asr_text when
            reliable, org_text otherwise (the paper's input).
  coverage  diagnostic only: share of the ASR tokens that also occur in
            org_text (low = the organisers' text is not a transcript).

Usage:
    python src/s05_moments.py --config config.perun.yaml --asr "outputs/asr/asr_*of4.jsonl"
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config, load_records, require, setup_stdout
from text_clean import clean_text

N_WIN = 4
CLIP_FRAMES = 64      # 16 frames x stride 4 per H5 clip
SNAP_S = 1.0          # max shift of a window boundary towards a pause
MIN_PAUSE_S = 0.15    # a gap between words at least this long counts as a pause
MIN_WORDS = 5
MIN_SPEECH = 0.20


def norm_tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", s)


def clip_layout(duration: float, fps: float) -> tuple[list, list]:
    span = min(CLIP_FRAMES / fps, duration) if fps and fps > 0 else duration / N_WIN
    starts = [k * (duration - span) / (N_WIN - 1) for k in range(N_WIN)]
    return [[s, s + span] for s in starts], [s + span / 2 for s in starts]


def pauses(words: list, vad: list) -> list[float]:
    """Candidate cut points: middles of gaps between words and of VAD silences."""
    pts = []
    timed = [(s, e) for _, s, e, *_ in words if s is not None and e is not None]
    for (_, e0), (s1, _) in zip(timed, timed[1:]):
        if s1 - e0 >= MIN_PAUSE_S:
            pts.append((e0 + s1) / 2)
    for (_, e0), (s1, _) in zip(vad, vad[1:]):
        pts.append((e0 + s1) / 2)
    return sorted(pts)


def moment_windows(duration: float, fps: float, words: list, vad: list) -> tuple[list, list, list]:
    clips, centres = clip_layout(duration, fps)
    pts = pauses(words, vad)
    bounds, snapped = [0.0], []
    for k in range(N_WIN - 1):
        b = (centres[k] + centres[k + 1]) / 2
        lo, hi = max(b - SNAP_S, centres[k]), min(b + SNAP_S, centres[k + 1])
        cand = [p for p in pts if lo < p < hi]
        if cand:
            b2 = min(cand, key=lambda p: abs(p - b))
            snapped.append(round(b2 - b, 3))
            b = b2
        else:
            snapped.append(None)
        bounds.append(b)
    bounds.append(duration)
    wins = [[round(bounds[k], 3), round(bounds[k + 1], 3)] for k in range(N_WIN)]
    return wins, [[round(s, 3), round(e, 3)] for s, e in clips], snapped


def video_moments(rec: dict, asr: dict, duration: float, fps: float) -> dict:
    org = clean_text(rec.get("text"))[0]
    words = [w for w in (asr.get("words") or []) if w[0]]
    timed = [w for w in words if w[1] is not None and w[2] is not None]
    reliable = len(timed) >= MIN_WORDS and (asr.get("speech_ratio") or 0) >= MIN_SPEECH
    wins, clips, snapped = moment_windows(duration, fps, timed if reliable else [],
                                          asr.get("vad") or [])
    speech, n_words = [""] * N_WIN, [0] * N_WIN
    if reliable:
        buckets = [[] for _ in range(N_WIN)]
        for w, s, e, *_ in timed:
            m = (s + e) / 2
            k = next((i for i, (a, b) in enumerate(wins) if a <= m < b), N_WIN - 1)
            buckets[k].append(w.strip())
        speech = [" ".join(b) for b in buckets]
        n_words = [len(b) for b in buckets]
    asr_text = " ".join(s["text"].strip() for s in asr.get("segments", [])) if reliable else ""
    atok, otok = norm_tokens(" ".join(w[0] for w in words)), Counter(norm_tokens(org))
    coverage = round(sum(t in otok for t in atok) / len(atok), 3) if atok else None
    return {
        "reliable": reliable,
        "why_unreliable": None if reliable else
        ("no speech" if not timed else f"{len(timed)} words, speech {asr.get('speech_ratio')}"),
        "lang_used": asr.get("lang_used"),
        "windows": wins, "clips": clips, "boundary_shift": snapped,
        "speech": speech, "speech_n_words": n_words,
        "asr_text": asr_text, "org_text": org,
        "vlm_transcript": asr_text if reliable else org,
        "coverage": coverage,
    }


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--asr", nargs="+", required=True, help="ASR jsonl files or globs (s04_asr.py)")
    ap.add_argument("--out", default=None, help="default: outputs/moments/moments.jsonl")
    ap.add_argument("--allow-missing", action="store_true",
                    help="do not require every video (review subsets)")
    a = ap.parse_args()
    cfg = load_config(a.config)

    files = sorted({f for p in a.asr for f in glob.glob(p)})
    require(files, "ASR files found", expected=a.asr, found="none")
    asr = {}
    for f in files:
        for line in open(f, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                require(d["id_EXIST"] not in asr, "each video in one ASR file only",
                        expected="unique", found=d["id_EXIST"])
                asr[d["id_EXIST"]] = d
    recs = {str(r["id_EXIST"]): r for k in ("train_json", "test_json") for r in load_records(cfg.path(k))}
    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST")
    missing = sorted(set(vids.index) - set(asr))
    require(a.allow_missing or not missing, "ASR covers every video", expected=len(vids),
            found=f"{len(asr)} ({len(missing)} missing, e.g. {missing[:5]})")

    out = Path(a.out) if a.out else cfg.path("out_dir") / "moments" / "moments.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out, "w", encoding="utf-8") as f:
        for k in sorted(asr):
            v = vids.loc[k]
            m = video_moments(recs[k], asr[k], float(v.duration_s), float(v.fps))
            f.write(json.dumps({"id_EXIST": k, "split": v.split, "lang": v.lang,
                                "duration_s": float(v.duration_s), **m}, ensure_ascii=False) + "\n")
            rows.append({"split": v.split, "reliable": m["reliable"],
                         "filled": sum(n > 0 for n in m["speech_n_words"]),
                         "snapped": sum(s is not None for s in m["boundary_shift"]),
                         "coverage": m["coverage"]})
    r = pd.DataFrame(rows)
    print(f"{len(r)} videos -> {out}")
    print("reliable ASR by split:\n" + r.groupby("split").reliable.agg(["sum", "count"]).to_string())
    rel = r[r.reliable]
    print(f"reliable videos: windows with speech {rel.filled.mean():.2f} of {N_WIN} on average; "
          f"all {N_WIN} filled in {(rel.filled == N_WIN).mean():.0%}; "
          f"boundaries moved to a pause: {rel.snapped.sum() / (len(rel) * (N_WIN - 1)):.0%}")
    print(f"coverage of the speech by the organisers' text: median {rel.coverage.median():.2f}, "
          f"< 0.3 in {(rel.coverage < 0.3).mean():.0%} of reliable videos")


if __name__ == "__main__":
    main()
