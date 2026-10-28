"""Step 7: candidate neighbours for linking videos, per split and video vector.

For every split, the index holds that split's training videos only, and every
video (train and test) gets its 50 most similar index videos by cosine:

  outer k      index = outer_fold != k                         (5 splits)
  inner k.j    index = outer_fold != k and inner_for_outer_k != j  (20 splits)

Inner splits exist so that tuning on an inner fold never sees that fold's
videos as neighbours (the nested protocol of C0).

A video is never its own neighbour, and never gets a neighbour from its own
group: same author or same near-duplicate cluster (s03's `group`, merged
transitively). The test set shares an author with train for 1 of 674 videos,
so an author's other videos would be a CV-only gain.

Search is exact: one matrix product on L2-normalised vectors, the same
ranking as FAISS IndexFlatIP.

Video vectors (s06_features.py):
  bge_desc        BGE-M3 on TARGET + MESSAGE
  bge_transcript  BGE-M3 on the VLM transcript
  siglip          mean SigLIP 2 vector of the 4 clips' middle frames

Output outputs/candidates.h5:
  ids (n,)                       every video, train then test, sorted
  <split>/index_mask (n,)        1 where the video is in the split's index
  <split>/<source>/nbr (n, 50)   int32 row in `ids`, -1 where fewer exist
  <split>/<source>/sim (n, 50)   float16 cosine

Usage:
    python src/s07_candidates.py --config config.perun.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from common import load_config, require, setup_stdout

K = 50
SOURCES = {"bge_desc": ("bge", "desc"), "bge_transcript": ("bge", "transcript"),
           "siglip": ("siglip", "video")}


def load_vectors(feat_dir: Path, ids: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for name, (stage, key) in SOURCES.items():
        with h5py.File(feat_dir / f"{stage}.h5", "r") as h:
            missing = [i for i in ids if i not in h]
            require(not missing, f"{stage}.h5 has every video", expected=len(ids),
                    found=f"{len(missing)} missing, e.g. {missing[:5]}")
            x = np.stack([h[i][key][()] for i in ids]).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
        out[name] = x
    return out


def groups_for_all(vids: pd.DataFrame, folds: pd.DataFrame, dups: pd.DataFrame) -> np.ndarray:
    """One group id per video. Train videos keep s03's group; a test video takes
    the group of any train video it shares an author or a duplicate cluster with."""
    key_group = {}
    tr = folds.merge(dups[["id_EXIST", "dup_group"]], on="id_EXIST", how="left")
    for r in tr.itertuples():
        key_group[("a", r.author)] = r.group
        if r.dup_group >= 0:
            key_group[("d", r.dup_group)] = r.group
    g = {r.id_EXIST: r.group for r in tr.itertuples()}
    dg = dict(zip(dups.id_EXIST, dups.dup_group))
    out = []
    for r in vids.itertuples():
        if r.id_EXIST in g:
            out.append(g[r.id_EXIST])
            continue
        hit = key_group.get(("a", r.author))
        if hit is None and dg.get(r.id_EXIST, -1) >= 0:
            hit = key_group.get(("d", dg[r.id_EXIST]))
        out.append(hit if hit is not None else f"test:{r.id_EXIST}")
    return np.array(out, dtype=object)


def splits(folds: pd.DataFrame, ids: list[str]) -> dict[str, np.ndarray]:
    f = folds.set_index("id_EXIST").reindex(ids)
    is_train = f.outer_fold.notna().to_numpy()
    outer = f.outer_fold.fillna(-99).astype(int).to_numpy()
    out = {}
    for k in sorted(folds.outer_fold.unique()):
        if k == 0:
            continue  # fold 0 = train-only ties, always in the index
        out[f"outer{k}"] = is_train & (outer != k)
        inner = f[f"inner_for_outer_{k}"].fillna(-99).astype(int).to_numpy()
        for j in sorted(set(inner[is_train & (outer != k)]) - {-1, 0, -99}):
            out[f"inner{k}.{j}"] = is_train & (outer != k) & (inner != j)
    return out


def neighbours(x: np.ndarray, mask: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(mask)
    sim = x @ x[idx].T                                       # (n, n_index)
    same = groups[:, None] == groups[idx][None, :]           # own group, self included
    sim[same] = -np.inf
    k = min(K, len(idx))
    top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(-sim, top, 1).argsort(1, kind="stable")
    top = np.take_along_axis(top, order, 1)
    s = np.take_along_axis(sim, top, 1)
    nbr = np.where(np.isfinite(s), idx[top], -1).astype(np.int32)
    return nbr, np.where(np.isfinite(s), s, 0).astype(np.float16)


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--features", default=None, help="default: outputs/features")
    ap.add_argument("--out", default=None, help="default: outputs/candidates.h5")
    a = ap.parse_args()
    cfg = load_config(a.config)
    feat = Path(a.features) if a.features else cfg.path("out_dir") / "features"
    out = Path(a.out) if a.out else cfg.out("candidates.h5")

    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str}).sort_values(["split", "id_EXIST"],
                                                                                      ascending=[False, True])
    folds = pd.read_csv(cfg.out("folds.csv"), dtype={"id_EXIST": str})
    dups = pd.read_csv(cfg.out("dup_groups.csv"), dtype={"id_EXIST": str})
    ids = vids.id_EXIST.tolist()
    groups = groups_for_all(vids, folds, dups)
    n_test_linked = int(sum(not str(g).startswith("test:") for g, s in zip(groups, vids.split) if s == "test"))
    print(f"{len(ids)} videos; {n_test_linked} test videos share a group with train (excluded as neighbours)")

    X = load_vectors(feat, ids)
    sp = splits(folds, ids)
    print(f"{len(sp)} splits: {', '.join(sp)}")
    with h5py.File(out, "w") as h:
        h.create_dataset("ids", data=np.array(ids, dtype="S"))
        h.attrs.update(k=K, sources=list(SOURCES), created_by="comparison_study/src/s07_candidates.py")
        for name, mask in sp.items():
            g = h.create_group(name)
            g.create_dataset("index_mask", data=mask.astype(np.uint8))
            for src, x in X.items():
                nbr, sim = neighbours(x, mask, groups)
                # leak checks: neighbours come from the index, never from the own group
                ok = nbr >= 0
                require(mask[nbr[ok]].all(), f"{name}/{src}: neighbours are index videos",
                        expected="all", found=int((~mask[nbr[ok]]).sum()))
                rows = np.repeat(np.arange(len(ids)), K)[ok.ravel()]
                require((groups[rows] != groups[nbr[ok]]).all(), f"{name}/{src}: no own-group neighbour",
                        expected=0, found=int((groups[rows] == groups[nbr[ok]]).sum()))
                g.create_dataset(f"{src}/nbr", data=nbr)
                g.create_dataset(f"{src}/sim", data=sim)
            if name.startswith("outer"):
                print(f"  {name}: index {mask.sum()}, " + ", ".join(
                    f"{s} top1 {h[name][s]['sim'][:, 0].astype(np.float32).mean():.3f}"
                    f" top50 {h[name][s]['sim'][:, -1].astype(np.float32).mean():.3f}" for s in X))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
