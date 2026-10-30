"""Do W2's field-to-clip edges mean anything? SigLIP 2's view of the 4 clips.

W3 found that the 4 frames of a video rarely differ (data_note 16): the VLM
links 91 % of fields to all 4. W2 links each field to its top-2 clips by
SigLIP 2 similarity, which is arbitrary if the 4 similarities are nearly
equal. This measures, from the cached features (s06_features.py):

  1. clip homogeneity: cosine between the 4 clips of one video, against
     clips of two different videos
  2. per field: spread of its 4 clip similarities (max - min) and the margin
     between the 2nd and 3rd clip (what top-2 decides on), in units of the
     field's own-vs-other-video gap (the size of a real visual match)
  3. agreement with W3 where the VLM was selective (1-3 frames scored 2):
     is SigLIP's top clip one of the VLM's, against chance (chosen / 4)?
  4. position: which clips SigLIP's top-2 picks (moments 1-4)

Usage:
    python src/w2_grounding_check.py --config config.perun.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from common import load_config, setup_stdout

FIELDS = ["TARGET", "MESSAGE", "SITUATION", "PEOPLE"]


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--features", default=None, help="default: outputs/features")
    ap.add_argument("--w3", default=None, help="default: outputs/w3/edges.jsonl")
    a = ap.parse_args()
    cfg = load_config(a.config)
    feat = Path(a.features) if a.features else cfg.path("out_dir") / "features"
    with h5py.File(feat / "siglip.h5", "r") as h:
        ids = sorted(h.keys())
        F = np.stack([h[k]["fields"][()] for k in ids]).astype(np.float32)   # (n, 4 fields, d)
        C = np.stack([h[k]["clips"][()] for k in ids]).astype(np.float32)    # (n, 4 clips, d)
    n = len(ids)
    rng = np.random.default_rng(0)

    # 1. homogeneity
    cc = np.einsum("ncd,ned->nce", C, C)
    within = cc[:, np.triu_indices(4, 1)[0], np.triu_indices(4, 1)[1]]       # (n, 6)
    perm = rng.permutation(n)
    between = np.einsum("ncd,ncd->nc", C, C[perm])                           # clip k of two videos
    print(f"1. clip cosine, same video: median {np.median(within):.3f} (min pair per video, median "
          f"{np.median(within.min(1)):.3f}); different videos: median {np.median(between):.3f}")

    # 2. field-to-clip similarities
    S = np.einsum("nfd,ncd->nfc", F, C)                                      # (n, 4, 4)
    other = np.einsum("nfd,ncd->nfc", F, C[perm]).mean(2)                    # field vs another video's clips
    gap = float(np.median(S.mean(2) - other))                                # size of a real match
    srt = np.sort(S, axis=2)[:, :, ::-1]
    spread = srt[:, :, 0] - srt[:, :, 3]
    margin = srt[:, :, 1] - srt[:, :, 2]
    print(f"2. field-clip similarity: own video {S.mean():.4f}, another video {other.mean():.4f}; "
          f"real-match gap (median) {gap:.4f}")
    print(f"   spread of a field's 4 clips (max - min): median {np.median(spread):.4f} = "
          f"{np.median(spread) / gap:.2f} x gap")
    print(f"   2nd-3rd margin (what top-2 decides on): median {np.median(margin):.4f} = "
          f"{np.median(margin) / gap:.2f} x gap; below 0.1 x gap in {np.mean(margin < 0.1 * gap):.0%} "
          f"of fields, below 0.25 x gap in {np.mean(margin < 0.25 * gap):.0%}")
    for i, f in enumerate(FIELDS):
        print(f"     {f:9s} spread {np.median(spread[:, i]):.4f}  margin {np.median(margin[:, i]):.4f}")

    # 4. position of SigLIP's top-2
    top2 = np.argsort(-S, axis=2)[:, :, :2]
    print(f"4. SigLIP top-2 by position (moments 1-4): {np.bincount(top2.ravel(), minlength=4).tolist()}")

    # 3. agreement with W3 where the VLM was selective
    w3 = Path(a.w3) if a.w3 else cfg.out("w3/edges.jsonl")
    if not w3.is_file():
        print(f"3. W3 file {w3} not found, skipped")
        return
    row = {k: i for i, k in enumerate(ids)}
    hit = chance = cnt = 0
    hit_none = []
    for line in open(w3, encoding="utf-8"):
        r = json.loads(line)
        if r["id_EXIST"] not in row or not r.get("edges"):
            continue
        i = row[r["id_EXIST"]]
        for fi, f in enumerate(FIELDS):
            ch = r["edges"][f]["frames"]
            if 0 < len(ch) < 4:
                cnt += 1
                hit += (int(np.argmax(S[i, fi])) + 1) in ch
                chance += len(ch) / 4
            elif not ch:
                hit_none.append(S[i, fi].mean() - other[i, fi])
    if cnt:
        print(f"3. fields where W3 chose 1-3 frames: {cnt}; SigLIP's top clip is one of them in "
              f"{hit / cnt:.0%} (chance {chance / cnt:.0%})")
    if hit_none:
        print(f"   fields W3 found in no frame: {len(hit_none)}; their own-vs-other similarity gap "
              f"{np.median(hit_none):.4f} (all fields: {gap:.4f})")


if __name__ == "__main__":
    main()
