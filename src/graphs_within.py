"""Steps 9-10: graphs inside one video, W2 and the controls C1-C3.

Nodes (features from s06_features.py, frozen):
  field   4   XLM-R of TARGET, MESSAGE, SITUATION, PEOPLE; pos = which field;
              extra = how visible the field is in the video (below)
  speech  0-4 XLM-R of each moment's speech, only moments with speech; pos = moment
  clip    4   VideoMAE of each H5 clip; pos = moment
  org     1   XLM-R of the organisers' cleaned text
  hub     1   learned (the model's own vector), linked to every node

W2 relations (stored in both directions; edge attribute = cosine for the two
similarity relations, 1 otherwise):
  next          consecutive speech nodes, consecutive clips
  overlaps      speech of moment k - clip of moment k
  field         every pair of fields
  member        every node - hub
  grounds       field - clip, SigLIP 2 (field's first sentence vs the clip's middle frame)
  paraphrases   field - speech, BGE-M3

grounds and paraphrases do not take the top 2 of 4 (the plan's first rule):
the 4 clips of a video have cosine 0.91 to each other (0.58 across videos),
and the 2nd-3rd similarity margin is below a tenth of a real match for 56 %
of fields (w2_grounding_check.py, data_note 16), so a top 2 would be chosen
by noise. A field links to every moment within DELTA_FRAC of a real match
(the median own-vs-other-video gap) of its best one, and to none if even its
best is no closer than a random other video's best (95th percentile): the
rule W3's VLM applies, so W2 and W3 differ only in who decides. Thresholds
come from training videos' features only, no labels.

Controls on the same nodes, with the same relation types:
  C1  no edges at all (each node encoded alone, then pooled)
  C2  random edges, the same number per relation and video as W2
      (seeded by video id, so every run sees the same graph)
  C3  every pair of nodes for each relation type

Every approach shares the field visibility feature: (best SigLIP similarity
to the video's clips - median best similarity to a random video's clips) /
real-match gap. It uses inputs only, so test videos have it too.

Usage (statistics of every approach -> results/graph_stats/):
    python src/graphs_within.py --config config.perun.yaml --stats
"""
from __future__ import annotations

import argparse
import time
import zlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from common import load_config, require, setup_stdout

N_WIN = 4
DELTA_FRAC = 0.25      # a moment ties with the best one if within this share of a real match
TAU_PCT = 95           # "shown at all": best similarity above this percentile of random videos' best
APPROACHES = ["W2", "C1", "C2", "C3"]
# undirected relation types: (node type a, name, node type b); same-type ones are stored
# once in both directions, cross-type ones as (a, name, b) and (b, rev_name, a)
UNDIRECTED = [("speech", "next", "speech"), ("clip", "next", "clip"), ("speech", "overlaps", "clip"),
              ("field", "field", "field"), ("field", "grounds", "clip"), ("field", "paraphrases", "speech"),
              *[(t, "member", "hub") for t in ("field", "speech", "clip", "org")]]
SIM_RELS = {"grounds", "paraphrases"}


def edge_types() -> list[tuple]:
    """Every directed edge type a graph carries (empty ones included, so batches line up)."""
    out = []
    for a, r, b in UNDIRECTED:
        out.append((a, r, b))
        if a != b:
            out.append((b, f"rev_{r}", a))
    return out


NODE_TYPES = ["field", "speech", "clip", "org", "hub"]
TEXT_FEATURES = ["xlmr", "bge_text", "e5"]


class Features:
    """All node features and similarity matrices, from the s06 h5 files."""

    def __init__(self, feat_dir: Path, ids: list[str], text: str = "xlmr"):
        def load(stage, key):
            with h5py.File(feat_dir / f"{stage}.h5", "r") as h:
                missing = [i for i in ids if i not in h]
                require(not missing, f"{stage}.h5 has every video", expected=len(ids),
                        found=f"{len(missing)} missing, e.g. {missing[:3]}")
                return np.stack([h[i][key][()] for i in ids]).astype(np.float32)
        self.ids = ids
        self.row = {k: i for i, k in enumerate(ids)}
        # text node features: xlmr (the plan's), or bge_text / e5 (s06 stages in the same layout)
        require(text in TEXT_FEATURES, "known text encoder", expected=TEXT_FEATURES, found=text)
        self.text = text
        self.field = load(text, "fields")
        self.speech = load(text, "speech")
        self.mask = load(text, "speech_mask").astype(bool)
        self.org = load(text, "org")[:, 0]
        self.clip = load("videomae", "clips")
        self._sig_f, self._sig_c = load("siglip", "fields"), load("siglip", "clips")
        self._bge_f, self._bge_s = load("bge", "fields"), load("bge", "speech")
        self.sig = np.einsum("nfd,ncd->nfc", self._sig_f, self._sig_c)     # (n, 4 fields, 4 clips)
        self.bge = np.einsum("nfd,nsd->nfs", self._bge_f, self._bge_s)     # (n, 4 fields, 4 moments)


def thresholds(ft: Features, train_rows: np.ndarray, seed: int = 0) -> dict:
    """Tie margin and "shown at all" threshold per similarity, from training videos only."""
    rng = np.random.default_rng(seed)
    out = {}
    # each field against its own video's moments and against a random other training video's
    perm = rng.permutation(train_rows)
    every = np.ones((len(train_rows), N_WIN), bool)
    out["grounds"] = _calibrate(ft.sig[train_rows], _cross(ft, "sig", train_rows, perm), every, every)
    has = train_rows[ft.mask[train_rows].any(1)]
    perm = rng.permutation(has)
    out["paraphrases"] = _calibrate(ft.bge[has], _cross(ft, "bge", has, perm), ft.mask[has], ft.mask[perm])
    return out


def _cross(ft: Features, kind: str, rows: np.ndarray, other: np.ndarray) -> np.ndarray:
    if kind == "bge":
        return np.einsum("nfd,nsd->nfs", ft._bge_f[rows], ft._bge_s[other])
    return np.einsum("nfd,ncd->nfc", ft._sig_f[rows], ft._sig_c[other])


def _calibrate(own: np.ndarray, cross: np.ndarray, own_mask: np.ndarray, cross_mask: np.ndarray) -> dict:
    """gap = median(field's mean own similarity - mean similarity to another video);
    tau = 95th percentile of a field's best similarity to another video."""
    om = np.where(own_mask[:, None, :], own, np.nan)
    cm = np.where(cross_mask[:, None, :], cross, np.nan)
    gap = float(np.nanmedian(np.nanmean(om, 2) - np.nanmean(cm, 2)))
    best_other = np.nanmax(cm, 2)
    require(gap > 0, "own video is more similar than another video (features are meaningful)",
            expected="gap > 0", found=gap)
    return {"gap": gap, "delta": DELTA_FRAC * gap, "tau": float(np.nanpercentile(best_other, TAU_PCT)),
            "median_best_other": float(np.nanmedian(best_other))}


def ties_to_best(sim: np.ndarray, present: np.ndarray, cal: dict) -> list[tuple[int, int]]:
    """(field, moment) pairs: moments within delta of the field's best, if the best clears tau."""
    out = []
    for f in range(sim.shape[0]):
        s = np.where(present, sim[f], -np.inf)
        best = s.max()
        if not np.isfinite(best) or best < cal["tau"]:
            continue
        out += [(f, m) for m in range(sim.shape[1]) if present[m] and s[m] >= best - cal["delta"]]
    return out


def build(ft: Features, k: str, approach: str, cal: dict, seed: int = 0) -> dict:
    """One video's graph: x / pos per node type, directed edges (2, E) and attributes (E, 1)."""
    i = ft.row[k]
    present = ft.mask[i]
    moments = np.flatnonzero(present)                      # speech node j <-> moment moments[j]
    node_of = {m: j for j, m in enumerate(moments)}
    vis = (ft.sig[i].max(1) - cal["grounds"]["median_best_other"]) / cal["grounds"]["gap"]
    x = {"field": np.concatenate([ft.field[i], vis[:, None].astype(np.float32)], 1),
         "speech": ft.speech[i][moments], "clip": ft.clip[i], "org": ft.org[i][None],
         "hub": np.zeros((1, 0), np.float32)}
    pos = {"field": np.arange(4), "speech": moments, "clip": np.arange(N_WIN),
           "org": np.zeros(1, int), "hub": np.zeros(1, int)}
    n = {t: len(v) for t, v in x.items()}
    sim = {"grounds": ft.sig[i], "paraphrases": ft.bge[i][:, moments] if len(moments) else np.zeros((4, 0))}

    def all_pairs(a, b):
        if a == b:
            return [(u, v) for u in range(n[a]) for v in range(u + 1, n[a])]
        return [(u, v) for u in range(n[a]) for v in range(n[b])]

    w2 = {("speech", "next", "speech"): [(j, j + 1) for j in range(n["speech"] - 1)],
          ("clip", "next", "clip"): [(c, c + 1) for c in range(N_WIN - 1)],
          ("speech", "overlaps", "clip"): [(node_of[m], m) for m in moments],
          ("field", "field", "field"): all_pairs("field", "field"),
          ("field", "grounds", "clip"): ties_to_best(ft.sig[i], np.ones(N_WIN, bool), cal["grounds"]),
          ("field", "paraphrases", "speech"): [(f, node_of[m]) for f, m in
                                               ties_to_best(ft.bge[i], present, cal["paraphrases"])],
          **{(t, "member", "hub"): [(u, 0) for u in range(n[t])] for t in ("field", "speech", "clip", "org")}}
    if approach == "W2":
        und = w2
    elif approach == "C1":
        und = {r: [] for r in w2}
    elif approach == "C3":
        und = {r: all_pairs(r[0], r[2]) for r in w2}
    elif approach == "C2":
        rng = np.random.default_rng([zlib.crc32(k.encode()), seed])
        und = {}
        for r, e in w2.items():
            pool = all_pairs(r[0], r[2])
            pick = rng.choice(len(pool), size=len(e), replace=False) if e else []
            und[r] = [pool[j] for j in sorted(pick)]
    else:
        raise ValueError(approach)

    edges = {}
    for (a, r, b), pairs in und.items():
        attr = np.array([sim[r][u, v] if r in SIM_RELS else 1.0 for u, v in pairs], np.float32).reshape(-1, 1)
        ei = np.array(pairs, dtype=np.int64).reshape(-1, 2).T
        if a == b:
            edges[(a, r, b)] = (np.concatenate([ei, ei[::-1]], 1), np.concatenate([attr, attr]))
        else:
            edges[(a, r, b)] = (ei, attr)
            edges[(b, f"rev_{r}", a)] = (ei[::-1].copy(), attr.copy())
    require(set(edges) == set(edge_types()), f"{k}/{approach}: every edge type present",
            expected=len(edge_types()), found=len(edges))
    return {"id": k, "x": x, "pos": pos, "edges": edges, "n_speech_expected": int(present.sum())}


def to_stats_graph(g: dict):
    from graph_stats import VideoGraph
    return VideoGraph(g["id"], {t: len(v) for t, v in g["x"].items()},
                      {et: ei for et, (ei, _) in g["edges"].items()},
                      expected={"speech": g["n_speech_expected"], "field": 4, "clip": N_WIN, "org": 1, "hub": 1})


def load_inputs(cfg, feat_dir: Path | None = None, text: str = "xlmr") -> tuple[Features, dict, list[str]]:
    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str}).sort_values("id_EXIST")
    ids = vids.id_EXIST.tolist()
    ft = Features(feat_dir or cfg.path("out_dir") / "features", ids, text)
    train_rows = np.flatnonzero((vids.split == "train").to_numpy())
    cal = thresholds(ft, train_rows)
    return ft, cal, ids


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--features", default=None, help="default: outputs/features")
    ap.add_argument("--stats", action="store_true", help="graph statistics of every approach")
    ap.add_argument("--out", default="results/graph_stats")
    a = ap.parse_args()
    cfg = load_config(a.config)
    t0 = time.time()
    ft, cal, ids = load_inputs(cfg, Path(a.features) if a.features else None)
    for r, c in cal.items():
        print(f"{r}: real-match gap {c['gap']:.4f}, tie margin {c['delta']:.4f}, shown-at-all threshold "
              f"{c['tau']:.4f} (random video's best, median {c['median_best_other']:.4f})")
    print(f"features loaded in {time.time() - t0:.0f} s")
    if a.stats:
        from graph_stats import write_stats
        for ap_ in APPROACHES:
            graphs = [to_stats_graph(build(ft, k, ap_, cal)) for k in ids]
            row = write_stats(ap_, "all", Path(a.out), graphs=graphs)
            per_rel = {c.replace("e_", ""): round(v, 2) for c, v in row.items()
                       if c.startswith("e_") and c.endswith("_median")}
            print(f"  {ap_}: edges/video median {row.get('edges_median')}, errors {row['errors']}, "
                  f"isolated without hub (median) {row.get('isolated_without_hub_median')}; {per_rel}")


if __name__ == "__main__":
    main()
