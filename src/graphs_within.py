"""Steps 9-10, 12-13: graphs inside one video, W2, W3, W4 and the controls C1-C3.

Nodes (features from s06_features.py, frozen):
  field   4   text encoder (e5 in setup v2, XLM-R in v1) of TARGET, MESSAGE,
              SITUATION, PEOPLE; pos = which field;
              extra = how visible the field is in the video (below)
  speech  0-4 text encoder of each moment's speech, only moments with speech; pos = moment
  clip    4   VideoMAE of each H5 clip; pos = moment
  org     1   text encoder of the organisers' cleaned text
  hub     1   learned (the model's own vector), linked to every node
  audio   0-4 optional (--audio, Step 23): Whisper encoder mean over each moment's
              window, only windows with audio; pos = moment

W2 relations (stored in both directions; edge attribute = cosine for the two
similarity relations, 1 otherwise):
  next          consecutive speech nodes, consecutive clips
  overlaps      speech of moment k - clip of moment k
  field         every pair of fields
  member        every node - hub
  grounds       field - clip, SigLIP 2 (field's first sentence vs the clip's middle frame)
  paraphrases   field - speech, BGE-M3
  with --audio: next (consecutive audio windows), overlaps (audio - clip of
  the same moment), heard_in (speech - audio of the same moment), member

grounds and paraphrases do not take the top 2 of 4 (the plan's first rule):
the 4 clips of a video have cosine 0.91 to each other (0.58 across videos),
and the 2nd-3rd similarity margin is below a tenth of a real match for 56 %
of fields (w2_grounding_check.py, data_note 16), so a top 2 would be chosen
by noise. A field links to every moment within DELTA_FRAC of a real match
(the median own-vs-other-video gap) of its best one, and to none if even its
best is no closer than a random other video's best (95th percentile): the
rule W3's VLM applies, so W2 and W3 differ only in who decides. Thresholds
come from training videos' features only, no labels.

W3 (Step 12): W2 with grounds and paraphrases chosen by Qwen3-VL instead of by
similarity (w3_vlm_edges.py, prompt v4, outputs/w3/edges.jsonl): a field links
to a moment's clip / speech when the VLM scored that frame / speech 2 ("shows
or says it specifically"). Every other relation, the node features and the
edge attribute (the same cosine as W2) stay, so W2 and W3 differ only in who
picks the pairs.

Controls on the same nodes, with the same relation types:
  C1  no edges at all (each node encoded alone, then pooled)
  C2  random edges, the same number per relation and video as W2
      (seeded by video id, so every run sees the same graph)
  C3  every pair of nodes for each relation type

W4 (Step 13), learned edges: built here exactly as C3 (every pair = the
candidates); the model's scorer (train.py) rates each candidate pair of a
non-member relation and keeps the k best per receiving node during training,
so W4 differs from C3 only in that pruning. member edges to the hub stay.

Every approach shares the field visibility feature: (best SigLIP similarity
to the video's clips - median best similarity to a random video's clips) /
real-match gap. It uses inputs only, so test videos have it too.

Usage (statistics of every approach -> results/graph_stats/):
    python src/graphs_within.py --config config.perun.yaml --stats
"""
from __future__ import annotations

import argparse
import json
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
APPROACHES = ["W2", "C1", "C2", "C3", "W3", "W4"]     # new ones last; B1, B1N: graphs_between.py
# undirected relation types: (node type a, name, node type b); same-type ones are stored
# once in both directions, cross-type ones as (a, name, b) and (b, rev_name, a)
UNDIRECTED = [("speech", "next", "speech"), ("clip", "next", "clip"), ("speech", "overlaps", "clip"),
              ("field", "field", "field"), ("field", "grounds", "clip"), ("field", "paraphrases", "speech"),
              *[(t, "member", "hub") for t in ("field", "speech", "clip", "org")]]
AUDIO_UNDIRECTED = [("audio", "next", "audio"), ("audio", "overlaps", "clip"), ("speech", "heard_in", "audio"),
                    ("audio", "member", "hub")]
SIM_RELS = {"grounds", "paraphrases"}


def edge_types(audio: bool = False) -> list[tuple]:
    """Every directed edge type a graph carries (empty ones included, so batches line up).
    Audio types come last, so a graph without audio keeps the original order."""
    out = []
    for a, r, b in UNDIRECTED + (AUDIO_UNDIRECTED if audio else []):
        out.append((a, r, b))
        if a != b:
            out.append((b, f"rev_{r}", a))
    return out


NODE_TYPES = ["field", "speech", "clip", "org", "hub"]
TEXT_FEATURES = ["xlmr", "bge_text", "e5"]


def node_types(audio: bool = False) -> list[str]:
    return NODE_TYPES + (["audio"] if audio else [])


class Features:
    """All node features and similarity matrices, from the s06 h5 files."""

    def __init__(self, feat_dir: Path, ids: list[str], text: str = "xlmr", audio: bool = False):
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
        self.audio = audio
        if audio:
            self.wav = load("whisper", "audio")
            self.wav_mask = load("whisper", "audio_mask").astype(bool)
        self._sig_f, self._sig_c = load("siglip", "fields"), load("siglip", "clips")
        self._bge_f, self._bge_s = load("bge", "fields"), load("bge", "speech")
        self.sig = np.einsum("nfd,ncd->nfc", self._sig_f, self._sig_c)     # (n, 4 fields, 4 clips)
        self.w3 = None
        self.bge = np.einsum("nfd,nsd->nfs", self._bge_f, self._bge_s)     # (n, 4 fields, 4 moments)


FIELDS = ["TARGET", "MESSAGE", "SITUATION", "PEOPLE"]


def load_w3(ft: Features, path: Path) -> None:
    """W3's chosen moments per video: ft.w3[k] = (frames, speech), each a list per field
    of 0-based moments scored 2. Every video must have an answer whose speech flags
    match the moments the graph uses."""
    require(path.is_file(), f"{path} exists (run w3_vlm_edges.py)", expected="file", found="missing")
    rows = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("edges"):
            rows[r["id_EXIST"]] = r
    missing = [k for k in ft.ids if k not in rows]
    require(not missing, "W3 has an answer for every video", expected=len(ft.ids),
            found=f"{len(missing)} missing, e.g. {missing[:3]}")
    bad = [k for k in ft.ids if list(map(bool, rows[k]["has_speech"])) != list(ft.mask[ft.row[k]])]
    require(not bad, "W3's speech flags equal the speech nodes'", expected=0, found=f"{len(bad)}, e.g. {bad[:3]}")
    ft.w3 = {}
    for k in ft.ids:
        e = rows[k]["edges"]
        ft.w3[k] = ([[m - 1 for m in e[f]["frames"]] for f in FIELDS],
                    [[m - 1 for m in e[f]["speech"]] for f in FIELDS])


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
    """One video's graph: x / pos per node type, directed edges (2, E) and attributes (E, 1).
    Audio nodes and relations are added when ft was loaded with audio=True."""
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
    if ft.audio:
        heard = np.flatnonzero(ft.wav_mask[i])             # audio node j <-> window heard[j]
        a_of = {m: j for j, m in enumerate(heard)}
        x["audio"], pos["audio"], n["audio"] = ft.wav[i][heard], heard, len(heard)
        w2.update({("audio", "next", "audio"): [(j, j + 1) for j in range(n["audio"] - 1)],
                   ("audio", "overlaps", "clip"): [(a_of[m], m) for m in heard],
                   ("speech", "heard_in", "audio"): [(node_of[m], a_of[m]) for m in moments if m in a_of],
                   ("audio", "member", "hub"): [(u, 0) for u in range(n["audio"])]})
    if approach == "W2":
        und = w2
    elif approach == "W3":
        require(ft.w3 is not None, "W3 edges loaded (load_w3)", expected="loaded", found=None)
        frames, speech = ft.w3[k]
        und = {**w2, ("field", "grounds", "clip"): [(f, m) for f in range(4) for m in frames[f]],
               ("field", "paraphrases", "speech"): [(f, node_of[m]) for f in range(4) for m in speech[f]
                                                    if m in node_of]}
    elif approach == "C1":
        und = {r: [] for r in w2}
    elif approach in ("C3", "W4"):            # W4: C3's pairs are its candidates
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
    require(set(edges) == set(edge_types(ft.audio)), f"{k}/{approach}: every edge type present",
            expected=len(edge_types(ft.audio)), found=len(edges))
    return {"id": k, "x": x, "pos": pos, "edges": edges, "n_speech_expected": int(present.sum()),
            "n_audio_expected": int(ft.wav_mask[i].sum()) if ft.audio else None}


def to_stats_graph(g: dict):
    from graph_stats import VideoGraph
    return VideoGraph(g["id"], {t: len(v) for t, v in g["x"].items()},
                      {et: ei for et, (ei, _) in g["edges"].items()},
                      expected={"speech": g["n_speech_expected"], "field": 4, "clip": N_WIN, "org": 1, "hub": 1,
                                **({"audio": g["n_audio_expected"]} if g["n_audio_expected"] is not None else {})})


def load_inputs(cfg, feat_dir: Path | None = None, text: str = "xlmr",
                audio: bool = False) -> tuple[Features, dict, list[str]]:
    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str}).sort_values("id_EXIST")
    ids = vids.id_EXIST.tolist()
    ft = Features(feat_dir or cfg.path("out_dir") / "features", ids, text, audio)
    train_rows = np.flatnonzero((vids.split == "train").to_numpy())
    cal = thresholds(ft, train_rows)
    return ft, cal, ids


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--features", default=None, help="default: outputs/features")
    ap.add_argument("--stats", action="store_true", help="graph statistics of every approach")
    ap.add_argument("--audio", action="store_true", help="add the Whisper audio nodes (Step 23)")
    ap.add_argument("--w3", default=None, help="W3 edges (default outputs/w3/edges.jsonl)")
    ap.add_argument("--out", default="results/graph_stats")
    a = ap.parse_args()
    cfg = load_config(a.config)
    t0 = time.time()
    ft, cal, ids = load_inputs(cfg, Path(a.features) if a.features else None, audio=a.audio)
    load_w3(ft, Path(a.w3) if a.w3 else cfg.out("w3/edges.jsonl"))
    for r, c in cal.items():
        print(f"{r}: real-match gap {c['gap']:.4f}, tie margin {c['delta']:.4f}, shown-at-all threshold "
              f"{c['tau']:.4f} (random video's best, median {c['median_best_other']:.4f})")
    print(f"features loaded in {time.time() - t0:.0f} s")
    if a.stats:
        from graph_stats import write_stats
        for ap_ in APPROACHES:
            graphs = [to_stats_graph(build(ft, k, ap_, cal)) for k in ids]
            row = write_stats(ap_ + ("_audio" if a.audio else ""), "all", Path(a.out), graphs=graphs)
            per_rel = {c.replace("e_", ""): round(v, 2) for c, v in row.items()
                       if c.startswith("e_") and c.endswith("_median")}
            print(f"  {ap_}: edges/video median {row.get('edges_median')}, errors {row['errors']}, "
                  f"isolated without hub (median) {row.get('isolated_without_hub_median')}; {per_rel}")


if __name__ == "__main__":
    main()
