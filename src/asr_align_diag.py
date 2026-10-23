"""Diagnostic used in the 40-video ASR review: align the organisers' sentences to ASR words.

Not a pipeline step. The review showed that the organisers' unmatched text is
mostly an LLM's account of the video rather than on-screen text, so the
pipeline keeps that text whole (src/s05_moments.py). Kept to reproduce the review.

Inputs: the cleaned organisers' `text` (src/text_clean.py) and our ASR words
(s04_asr.py). The organisers' text stays the content everywhere; ASR is only
the clock and the speech detector:

  1. split the cleaned text into sentences (punctuation / line breaks; runs
     longer than MAX_TOKENS are cut into chunks so they can be placed in time);
  2. align the sentence tokens to the ASR token stream (monotone longest
     matching blocks on normalised tokens);
  3. a sentence whose tokens are found in the speech (match >= MIN_MATCH) is
     `speech`, timed by its matched words; otherwise it is `overlay`
     (on-screen text, or text the organisers' LLM step added);
  4. moments: 4 windows around the 4 H5 clips (clip k starts at
     k * (T - span) / 3 s, span = 64 source frames; boundaries halfway between
     clip centres), and each speech sentence goes to the window it overlaps most.

Output: outputs/align/review_align.jsonl, one line per video.

Usage:
    python src/asr_align_diag.py --config config.yaml --asr outputs/asr/asr_*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config, load_records, require, setup_stdout
from text_clean import clean_text

MAX_TOKENS = 30      # longer sentences are cut into CHUNK-token pieces
CHUNK = 15
MIN_MATCH = 0.5      # share of a sentence's tokens found in the speech
CLIP_FRAMES = 64     # 16 frames x stride 4 per H5 clip
N_CLIPS = 4

_SENT_RE = re.compile(r"(?<=[.!?…])\s+|\n+|(?=[¿¡])")


def norm_tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", s)


def split_sentences(text: str) -> list[str]:
    out = []
    for part in _SENT_RE.split(text or ""):
        part = part.strip()
        if not part or not norm_tokens(part):
            continue
        words = part.split()
        if len(words) <= MAX_TOKENS:
            out.append(part)
        else:
            out += [" ".join(words[i:i + CHUNK]) for i in range(0, len(words), CHUNK)]
    return out


def asr_tokens(words: list) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Flatten ASR words into normalised tokens with times (missing times interpolated)."""
    toks, t0, t1 = [], [], []
    for w, s, e, *_ in words:
        for t in norm_tokens(w or ""):
            toks.append(t)
            t0.append(np.nan if s is None else s)
            t1.append(np.nan if e is None else e)
    t0, t1 = np.array(t0, float), np.array(t1, float)
    for arr in (t0, t1):
        ok = ~np.isnan(arr)
        if ok.any() and not ok.all():
            idx = np.arange(len(arr))
            arr[~ok] = np.interp(idx[~ok], idx[ok], arr[ok])
    return toks, t0, t1


def windows(duration: float, fps: float) -> tuple[list, list]:
    """H5 clip spans and the 4 moment windows around them."""
    span = min(CLIP_FRAMES / fps, duration) if fps and fps > 0 else duration / N_CLIPS
    starts = [k * (duration - span) / (N_CLIPS - 1) for k in range(N_CLIPS)]
    clips = [[round(s, 3), round(s + span, 3)] for s in starts]
    centres = [s + span / 2 for s in starts]
    bounds = [0.0] + [(centres[k] + centres[k + 1]) / 2 for k in range(N_CLIPS - 1)] + [duration]
    wins = [[round(bounds[k], 3), round(bounds[k + 1], 3)] for k in range(N_CLIPS)]
    return clips, wins


def align_video(text: str, asr: dict, duration: float, fps: float) -> dict:
    sents = split_sentences(text)
    stoks = [norm_tokens(s) for s in sents]
    flat = [t for ts in stoks for t in ts]
    owner = [i for i, ts in enumerate(stoks) for _ in ts]
    atoks, at0, at1 = asr_tokens(asr.get("words") or [])
    match = np.full(len(flat), -1)
    if flat and atoks:
        sm = SequenceMatcher(None, flat, atoks, autojunk=False)
        for a, b, n in sm.get_matching_blocks():
            match[a:a + n] = np.arange(b, b + n)
    clips, wins = windows(duration, fps)
    out_s = []
    for i, s in enumerate(sents):
        m = match[[j for j, o in enumerate(owner) if o == i]]
        hit = m[m >= 0]
        frac = len(hit) / max(len(stoks[i]), 1)
        rec = {"i": i, "text": s, "n_tokens": len(stoks[i]), "match": round(frac, 3)}
        if frac >= MIN_MATCH and len(hit) >= min(2, len(stoks[i])):
            t0, t1 = float(np.nanmin(at0[hit])), float(np.nanmax(at1[hit]))
            ov = [max(0.0, min(t1, w1) - max(t0, w0)) for w0, w1 in wins]
            rec.update(kind="speech", t0=round(t0, 3), t1=round(t1, 3),
                       window=int(np.argmax(ov)) if max(ov) > 0 else
                       int(np.argmin([abs((t0 + t1) / 2 - (w0 + w1) / 2) for w0, w1 in wins])))
        else:
            rec.update(kind="overlay", t0=None, t1=None, window=None)
        out_s.append(rec)
    used = set(match[match >= 0].tolist())
    return {
        "sentences": out_s, "clips": clips, "windows": wins,
        "n_speech": sum(r["kind"] == "speech" for r in out_s),
        "n_overlay": sum(r["kind"] == "overlay" for r in out_s),
        # share of what was said that the organisers' text contains
        "asr_coverage": round(len(used) / len(atoks), 3) if atoks else None,
    }


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--asr", nargs="+", required=True, help="ASR jsonl file(s) or globs")
    ap.add_argument("--out", default=None, help="default: outputs/align/review_align.jsonl")
    a = ap.parse_args()
    cfg = load_config(a.config)

    files = sorted({f for p in a.asr for f in glob.glob(p)})
    require(files, "ASR files found", expected=a.asr, found="none")
    asr = {}
    for f in files:
        for line in open(f, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                asr[d["id_EXIST"]] = d
    recs = {str(r["id_EXIST"]): r for k in ("train_json", "test_json") for r in load_records(cfg.path(k))}
    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST")
    out = Path(a.out) if a.out else cfg.path("out_dir") / "align" / "align.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = []
    with open(out, "w", encoding="utf-8") as f:
        for k in sorted(asr):
            v = vids.loc[k]
            res = {"id_EXIST": k, "lang": v.lang,
                   **align_video(clean_text(recs[k].get("text"))[0], asr[k],
                                 float(v.duration_s), float(v.fps))}
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            stats.append((res["n_speech"], res["n_overlay"], res["asr_coverage"]))
    s = pd.DataFrame(stats, columns=["speech", "overlay", "coverage"])
    print(f"{len(s)} videos -> {out}")
    print(f"sentences: speech {s.speech.sum()}, overlay {s.overlay.sum()}; "
          f"videos with no speech sentence: {(s.speech == 0).sum()}; "
          f"median ASR coverage by organisers' text: {s.coverage.median()}")


if __name__ == "__main__":
    main()
