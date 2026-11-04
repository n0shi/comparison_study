"""Step 16: B1, kNN links between videos, on top of the base graph (C3, choices.md).

Each video's own graph (graphs_within.build with C3's pairs) gains one node per
neighbour video: the union of the top nbr_k of each candidate list
(s07_candidates.py: bge_desc, bge_transcript, siglip). A neighbour found by
several sources is one node with one edge per source.

  node   nbr     the neighbour's video vector (mean of its 4 text-encoder
                 fields | mean of its 4 VideoMAE clips) and its soft label;
                 the model adds "label shown" (UniMP label masking, train.py)
  edges  (nbr, near_<source>, hub), attribute [cosine, rank / MAX_K]

The graph only stores the neighbour's row (data["nbr"].vid); the model looks
the vector and label up in its own buffers, so the many per-fold graph sets
share one copy of the vectors.

Pools (choices.md, B1 pool rule): a fold-k model trains and selects its epoch
with neighbours from inner split k.1 (outer fold k and inner fold 1 removed)
and predicts the outer fold and the test set from outer split k, as C4 does.
Every attach checks: the neighbour is in the split's index, is a training
video, is not the video itself, and (by s07) not of its author or duplicate
group.

Variants (train.py):
  B1   labels of neighbours hidden at random in training (rate tuned), all
       shown when predicting
  B1N  labels always hidden: what the neighbours' content alone is worth
"""
from __future__ import annotations

import copy
from pathlib import Path

import h5py
import numpy as np

from common import require

APPROACHES = ["B1", "B1N"]
SOURCES = ["bge_desc", "bge_transcript", "siglip"]
MAX_K = 20


def edge_types() -> list[tuple]:
    return [("nbr", f"near_{s}", "hub") for s in SOURCES]


class Between:
    """Candidate lists, video vectors and soft labels, aligned with graphs_within's ids."""

    def __init__(self, cand_path: Path, ft, ids: list[str], labels: dict[str, np.ndarray],
                 train_ids: list[str], n_classes: int):
        self.h = h5py.File(cand_path, "r")
        cids = [i.decode() for i in self.h["ids"][()]]
        require(sorted(cids) == sorted(ids), "candidates.h5 holds the graphs' videos",
                expected=len(ids), found=len(cids))
        self.ids = ids
        self.row = {k: i for i, k in enumerate(ids)}
        self.c2g = np.array([self.row[i] for i in cids])          # candidate row -> graph row
        self.g2c = {self.row[i]: j for j, i in enumerate(cids)}
        require(all(s in list(self.h.attrs["sources"]) for s in SOURCES), "candidate sources present",
                expected=SOURCES, found=list(self.h.attrs["sources"]))
        rows = np.array([ft.row[k] for k in ids])
        self.vec = np.concatenate([ft.field[rows].mean(1), ft.clip[rows].mean(1)], 1).astype(np.float32)
        self.lab = np.zeros((len(ids), n_classes), np.float32)
        self.is_train = np.zeros(len(ids), bool)
        for k in train_ids:
            self.lab[self.row[k]] = labels[k]
            self.is_train[self.row[k]] = True
        self._lists = {}

    def lists(self, split: str):
        if split not in self._lists:
            g = self.h[split]
            mask = g["index_mask"][()].astype(bool)
            idx_mask = np.zeros(len(self.ids), bool)
            idx_mask[self.c2g[mask]] = True
            per = {}
            for s in SOURCES:
                nbr = g[s]["nbr"][()][:, :MAX_K]
                sim = g[s]["sim"][()][:, :MAX_K].astype(np.float32)
                per[s] = (np.where(nbr >= 0, self.c2g[np.maximum(nbr, 0)], -1), sim)
            self._lists[split] = (idx_mask, per)
        return self._lists[split]

    def attach(self, graphs: dict, keys: list[str], split: str, nbr_k: int,
               forbidden: set[str]) -> dict:
        """Shallow copies of graphs[k] for k in keys with neighbour nodes from split's index.
        forbidden: ids that must never be a neighbour here (the fold's outer / selection videos)."""
        import torch
        require(1 <= nbr_k <= MAX_K, "nbr_k within the stored lists", expected=f"1..{MAX_K}", found=nbr_k)
        idx_mask, per = self.lists(split)
        bad = np.array([self.row[i] for i in forbidden], dtype=np.int64)
        require(not idx_mask[bad].any(), f"{split}: forbidden videos are not in the index",
                expected=0, found=int(idx_mask[bad].sum()))
        require(self.is_train[idx_mask].all(), f"{split}: the index holds training videos only",
                expected="all", found=int((~self.is_train[idx_mask]).sum()))
        out = {}
        for k in keys:
            r = self.row[k]
            c = self.g2c[r]
            local, edges = {}, {}
            for s in SOURCES:
                nbr, sim = per[s][0][c, :nbr_k], per[s][1][c, :nbr_k]
                ok = nbr >= 0
                src, attr = [], []
                for rank in np.flatnonzero(ok):
                    v = int(nbr[rank])
                    src.append(local.setdefault(v, len(local)))
                    attr.append([float(sim[rank]), rank / MAX_K])
                edges[s] = (src, attr)
            vids = np.array(list(local), dtype=np.int64)
            require(not len(vids) or (idx_mask[vids].all() and (vids != r).all()),
                    f"{k}/{split}: neighbours are index videos other than itself", expected="all",
                    found=f"{int((~idx_mask[vids]).sum())} outside, self {int((vids == r).sum())}")
            g = copy.copy(graphs[k])
            g["nbr"].vid = torch.from_numpy(vids)
            g["nbr"].num_nodes = len(vids)
            for s, (src, attr) in edges.items():
                et = ("nbr", f"near_{s}", "hub")
                g[et].edge_index = torch.tensor([src, [0] * len(src)], dtype=torch.long).reshape(2, -1)
                g[et].edge_attr = torch.tensor(attr, dtype=torch.float32).reshape(-1, 2)
            out[k] = g
        return out
