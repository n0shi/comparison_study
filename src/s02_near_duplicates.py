"""Near-duplicate groups across train+test -> outputs/near_duplicates.csv + dup_groups.csv.

Visual: 64-bit pHash of the middle frame of each of the 4 H5 clips; a pair is
a near-duplicate when >= min_matching_clips clips are within max_hamming.
Text: identical normalised transcripts with >= 8 words (re-uploads, shared
sounds/lyrics). Both kinds are reported separately; groups are the connected
components of the union, and are what folds must keep together.

Usage:
    python src/s02_near_duplicates.py --config config.yaml
"""
from __future__ import annotations

import argparse
import io
import json
import re

import h5py
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from common import load_config, require, setup_stdout


def clip_hashes(h5_path, ids) -> np.ndarray:
    """(n_videos, 4) uint64 pHashes, middle frame of each clip."""
    out = np.zeros((len(ids), 4), dtype=np.uint64)
    with h5py.File(h5_path, "r") as f:
        g = f["clips"]
        for i, vid in enumerate(ids):
            clips = g[vid]
            require(clips.shape[0] == 4, f"H5 clip count for {vid}", expected=4, found=clips.shape)
            for k in range(4):
                frame = clips[k, clips.shape[1] // 2]
                img = Image.open(io.BytesIO(bytes(frame)))
                h = imagehash.phash(img).hash.flatten()
                out[i, k] = np.packbits(h).view(">u8")[0]
    return out


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    b = x.view(np.uint8).reshape(*x.shape, 8)
    return np.unpackbits(b, axis=-1).sum(-1)


def visual_pairs(hashes: np.ndarray, max_ham: int, min_clips: int):
    n = len(hashes)
    pairs = []
    for i in range(n - 1):
        # Clip k of video i vs any clip of video j: re-uploads may be trimmed.
        d = popcount64(hashes[i][None, :, None] ^ hashes[i + 1:][:, None, :])  # (m,4,4)
        matched = (d.min(axis=2) <= max_ham).sum(axis=1)
        for j in np.nonzero(matched >= min_clips)[0]:
            pairs.append((i, i + 1 + j, int(matched[j]), int(d[j].min(axis=1).mean())))
    return pairs


def norm_text(t: str) -> str:
    return re.sub(r"\W+", " ", (t or "").lower()).strip()


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    nd = cfg["near_duplicates"]

    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str, "id_Tiktok": str})
    from common import load_records
    text = {str(r["id_EXIST"]): r.get("text") or ""
            for k in ("train_json", "test_json") for r in load_records(cfg.path(k))}

    hashes = []
    for split, key in (("train", "train_h5"), ("test", "test_h5")):
        ids = vids.loc[vids.split == split, "id_Tiktok"].tolist()
        hashes.append(clip_hashes(cfg.path(key), ids))
        print(f"hashed {split}: {len(ids)}")
    hashes = np.concatenate(hashes)
    order = pd.concat([vids[vids.split == "train"], vids[vids.split == "test"]]).reset_index(drop=True)

    rows = []
    for i, j, m, d in visual_pairs(hashes, nd["max_hamming"], nd["min_matching_clips"]):
        rows.append({"a": order.id_EXIST[i], "b": order.id_EXIST[j], "kind": "visual",
                     "clips_matched": m, "mean_hamming": d})
    order["norm_text"] = order.id_EXIST.map(lambda i: norm_text(text[i]))
    long_txt = order[order.norm_text.str.split().str.len() >= 8]
    for _, grp in long_txt.groupby("norm_text"):
        ids = grp.id_EXIST.tolist()
        for a in range(len(ids) - 1):
            for b in range(a + 1, len(ids)):
                rows.append({"a": ids[a], "b": ids[b], "kind": "text"})
    pairs = pd.DataFrame(rows, columns=["a", "b", "kind", "clips_matched", "mean_hamming"])

    meta = order.set_index("id_EXIST")
    for col in ("split", "author", "lang", "hard_3_1"):
        pairs[f"a_{col}"] = pairs.a.map(meta[col])
        pairs[f"b_{col}"] = pairs.b.map(meta[col])
    pairs.to_csv(cfg.out("near_duplicates.csv"), index=False, encoding="utf-8")

    # Connected components over duplicate pairs -> dup_group per video.
    idx = {v: k for k, v in enumerate(order.id_EXIST)}
    if len(pairs):
        r = pairs.a.map(idx).to_numpy()
        c = pairs.b.map(idx).to_numpy()
        adj = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(order), len(order)))
        _, comp = connected_components(adj, directed=False)
    else:
        comp = np.arange(len(order))
    order["dup_group"] = comp
    sizes = order.dup_group.map(order.dup_group.value_counts())
    order.loc[sizes == 1, "dup_group"] = -1
    order[["id_EXIST", "split", "dup_group"]].to_csv(cfg.out("dup_groups.csv"), index=False)

    both = pairs.a_split != pairs.b_split
    s = {
        "pairs": pairs.kind.value_counts().to_dict(),
        "videos_in_dup_groups": int((order.dup_group >= 0).sum()),
        "n_dup_groups": int(order.loc[order.dup_group >= 0, "dup_group"].nunique()),
        "max_group_size": int(sizes.max()),
        "train_test_pairs": pairs[both].kind.value_counts().to_dict(),
        "test_videos_with_train_duplicate": int(pd.concat([
            pairs.loc[both & (pairs.a_split == "test"), "a"],
            pairs.loc[both & (pairs.b_split == "test"), "b"]]).nunique()),
        "pairs_same_author": int((pairs.a_author == pairs.b_author).sum()),
        "train_train_pairs_label_disagree_3_1": int(
            ((pairs.a_split == "train") & (pairs.b_split == "train")
             & (pairs.a_hard_3_1.notna()) & (pairs.b_hard_3_1.notna())
             & (pairs.a_hard_3_1 != pairs.b_hard_3_1)).sum()),
    }
    with open(cfg.out("near_duplicates_stats.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
