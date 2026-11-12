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

Step 18: B3, learned links. The neighbour nodes are the union of the top POOL
(50, all that s07 stores) of each candidate list, with B1's edges; the model's
scorer (train.py) keeps each video's keep_k best. The hub carries the video's
own row (data["hub"].vid) for the scorer.

Step 19: B4, theme hyperedges. Per split, k-means (n_clusters, tuned) on the
BGE-M3 vectors of one description field of the split's index (training)
videos; a cluster is a hyperedge. Its members reach a video through one node
per family (star expansion, the members summarised):

  node   nbr     the cluster's centroid; the mean soft label of its members
                 outside the video's own group (author / duplicates, as s07),
                 so a training video never sees its own label; `has` = 0 when
                 no such member exists
  edges  (nbr, theme_<FIELD>, hub), attribute [cosine to the centroid,
         members / index size]

A video joins the nearest centroid of each family, whether or not it is in the
index. B4 uses TARGET alone (the plan's), B4F all four fields.
"""
from __future__ import annotations

import copy
from pathlib import Path

import h5py
import numpy as np

from common import require

APPROACHES = ["B1", "B1N", "B3", "B4", "B4F"]
POOL = 50                            # B3: candidates per source (s07's K)
FIELDS = ["TARGET", "MESSAGE", "SITUATION", "PEOPLE"]       # order of the h5 "fields" rows
FAMILIES = {"B4": ["TARGET"], "B4F": FIELDS}
KMEANS_SEED = 0
SOURCES = ["bge_desc", "bge_transcript", "siglip"]
MAX_K = 20


def edge_types(approach: str | None = None) -> list[tuple]:
    if approach in FAMILIES:
        return [("nbr", f"theme_{f}", "hub") for f in FAMILIES[approach]]
    return [("nbr", f"near_{s}", "hub") for s in SOURCES]


class Between:
    """Candidate lists, video vectors and soft labels, aligned with graphs_within's ids."""

    def __init__(self, cand_path: Path, ft, ids: list[str], labels: dict[str, np.ndarray],
                 train_ids: list[str], n_classes: int, groups: dict[str, object] | None = None):
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
        parts = [ft.field[rows].mean(1), ft.clip[rows].mean(1)]
        self.vec = np.concatenate(parts, 1).astype(np.float32)
        # B3's scorer: each part L2-normalised, so neither encoder's scale decides the cosine
        self.unit = np.concatenate([p / np.linalg.norm(p, axis=1, keepdims=True).clip(min=1e-8) for p in parts],
                                   1).astype(np.float32)
        self.theme_vec = ft._bge_f[rows].astype(np.float32)      # (n, 4 fields, 1024), B4
        self.theme_vec /= np.linalg.norm(self.theme_vec, axis=2, keepdims=True).clip(min=1e-8)
        self.group = None if groups is None else np.array([groups[k] for k in ids], dtype=object)
        self._themes = {}
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
                nbr = g[s]["nbr"][()]
                sim = g[s]["sim"][()].astype(np.float32)
                per[s] = (np.where(nbr >= 0, self.c2g[np.maximum(nbr, 0)], -1), sim)
            self._lists[split] = (idx_mask, per)
        return self._lists[split]

    def _index(self, split: str, forbidden: set[str]) -> np.ndarray:
        idx_mask = self.lists(split)[0]
        bad = np.array([self.row[i] for i in forbidden], dtype=np.int64)
        require(not idx_mask[bad].any(), f"{split}: forbidden videos are not in the index",
                expected=0, found=int(idx_mask[bad].sum()))
        require(self.is_train[idx_mask].all(), f"{split}: the index holds training videos only",
                expected="all", found=int((~self.is_train[idx_mask]).sum()))
        return idx_mask

    def themes(self, split: str, family: str, n_clusters: int):
        """k-means of the split's index videos on one field: (centroids, cluster of every
        video, cosine of every video to its centroid, index rows)."""
        key = (split, family, n_clusters)
        if key not in self._themes:
            from sklearn.cluster import KMeans
            idx = np.flatnonzero(self.lists(split)[0])
            x = self.theme_vec[:, FIELDS.index(family)]
            require(n_clusters <= len(idx), f"{split}: no more clusters than index videos",
                    expected=f"<= {len(idx)}", found=n_clusters)
            km = KMeans(n_clusters=n_clusters, n_init=3, random_state=KMEANS_SEED).fit(x[idx])
            cen = km.cluster_centers_.astype(np.float32)
            cl = km.predict(x)
            cos = (x * cen[cl]).sum(1) / np.linalg.norm(cen[cl], axis=1).clip(min=1e-8)
            self._themes[key] = (cen, cl, cos.astype(np.float32), idx)
        return self._themes[key]

    def attach_themes(self, graphs: dict, keys: list[str], split: str, n_clusters: int,
                      families: list[str], forbidden: set[str]) -> dict:
        """Shallow copies of graphs[k] for k in keys with one theme node per family."""
        import torch
        require(self.group is not None, "groups given (author / duplicate groups)", expected="dict", found=None)
        idx_mask = self._index(split, forbidden)
        per = {}
        for f in families:
            cen, cl, cos, idx = self.themes(split, f, n_clusters)
            members = {}                                   # cluster -> index rows
            for r in idx:
                members.setdefault(int(cl[r]), []).append(int(r))
            per[f] = (cen, cl, cos, {c: np.array(m) for c, m in members.items()})
        out = {}
        for k in keys:
            r = self.row[k]
            xs, labs, has, attrs = [], [], [], []
            for f in families:
                cen, cl, cos, members = per[f]
                m = members.get(int(cl[r]), np.zeros(0, np.int64))
                m = m[self.group[m] != self.group[r]]      # never the video itself or its group
                require(idx_mask[m].all() and (m != r).all(), f"{k}/{split}: theme members are index videos",
                        expected="all", found=int((~idx_mask[m]).sum()))
                xs.append(cen[cl[r]])
                labs.append(self.lab[m].mean(0) if len(m) else np.zeros(self.lab.shape[1], np.float32))
                has.append(float(len(m) > 0))
                attrs.append([float(cos[r]), len(m) / int(idx_mask.sum())])
            g = copy.copy(graphs[k])
            g["nbr"].x = torch.from_numpy(np.stack(xs))
            g["nbr"].lab = torch.from_numpy(np.stack(labs).astype(np.float32))
            g["nbr"].has = torch.tensor(has, dtype=torch.float32)
            g["nbr"].num_nodes = len(families)
            for j, f in enumerate(families):
                et = ("nbr", f"theme_{f}", "hub")
                g[et].edge_index = torch.tensor([[j], [0]], dtype=torch.long)
                g[et].edge_attr = torch.tensor([attrs[j]], dtype=torch.float32)
            out[k] = g
        return out

    def attach(self, graphs: dict, keys: list[str], split: str, nbr_k: int,
               forbidden: set[str], max_k: int = MAX_K) -> dict:
        """Shallow copies of graphs[k] for k in keys with neighbour nodes from split's index.
        forbidden: ids that must never be a neighbour here (the fold's outer / selection videos).
        max_k: the largest nbr_k allowed and the rank's scale (B1: MAX_K; B3: POOL)."""
        import torch
        require(1 <= nbr_k <= max_k, "nbr_k within the stored lists", expected=f"1..{max_k}", found=nbr_k)
        idx_mask = self._index(split, forbidden)
        per = self.lists(split)[1]
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
                    attr.append([float(sim[rank]), rank / max_k])
                edges[s] = (src, attr)
            vids = np.array(list(local), dtype=np.int64)
            require(not len(vids) or (idx_mask[vids].all() and (vids != r).all()),
                    f"{k}/{split}: neighbours are index videos other than itself", expected="all",
                    found=f"{int((~idx_mask[vids]).sum())} outside, self {int((vids == r).sum())}")
            g = copy.copy(graphs[k])
            g["nbr"].vid = torch.from_numpy(vids)
            g["nbr"].num_nodes = len(vids)
            g["hub"].vid = torch.tensor([r], dtype=torch.long)
            for s, (src, attr) in edges.items():
                et = ("nbr", f"near_{s}", "hub")
                g[et].edge_index = torch.tensor([src, [0] * len(src)], dtype=torch.long).reshape(2, -1)
                g[et].edge_attr = torch.tensor(attr, dtype=torch.float32).reshape(-1, 2)
            out[k] = g
        return out
