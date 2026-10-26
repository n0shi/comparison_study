"""Step 8: structure, leak checks and cost of every built graph.

Three kinds of graph, three kinds of statistics:

  inside a video (W1-W6, C1-C3)   video_stats(): one row per video
      nodes and undirected edges per type, density per relation, isolated
      nodes with and without the hub's member edges (every node touches the
      hub, so only the second count can show an isolated node), components
      without the hub, node types that receive no messages, and errors that
      must be 0: self-loops, duplicate edges, out-of-range indices, edges
      without their reverse, node counts that differ from the inputs
      (an absent speech moment must have no node; its absence is a flag).
      Hypergraphs (W5) also report hyperedge sizes and uncovered nodes.

  between videos (B1-B5, C4)      link_stats(): one row per query video
      neighbours, similarity by role (train / val / test queries), overlap of
      sources, hubness (how often one training video is a neighbour), leak
      checks, and label structure: adjusted homophily (Platonov et al. 2023)
      per task and category, soft-label agreement against shuffled
      neighbours, and the link between a video's annotator disagreement and
      its neighbours'. Label statistics use training videos only: queries and
      neighbours both in the split's index, so validation labels never inform
      a choice.

  cost                            CostLog / read_costs()
      wall-clock, GPU-hours, VLM calls and tokens per approach, split and
      stage (build / train). Shared inputs (features, ASR, descriptions) are
      logged once as approach "shared", not charged to each approach.

write_stats() keeps everything: the per-video and per-query tables in
results/graph_stats/<approach>/<split>_*.csv for deeper analysis, and one
summary row per approach and split in results/graph_stats/graph_stats.csv.

Usage:
    python src/graph_stats.py selftest
    python src/graph_stats.py candidates --config config.perun.yaml   # Step 7 lists as kNN graphs
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from common import TASKS, require, setup_stdout

HUB = "hub"
OUT = Path("results/graph_stats")


# ------------------------------------------------------------------ graphs
@dataclass
class VideoGraph:
    """One video's graph, independent of the framework.

    nodes      {node type: count}
    edges      {(src type, relation, dst type): int array (2, E)}, directed;
               an undirected edge is stored in both directions, the reverse
               relation named `rev_<relation>` or the same name
    hyperedges list of hyperedges, each a list of (node type, index)
    expected   {node type: count} implied by the inputs, e.g. speech nodes =
               moments with speech; checked against `nodes`
    """
    vid: str
    nodes: dict
    edges: dict = field(default_factory=dict)
    hyperedges: list | None = None
    expected: dict | None = None


def from_hetero(data, vid: str, expected: dict | None = None) -> VideoGraph:
    """A PyTorch Geometric HeteroData (duck-typed: no torch import here)."""
    nodes = {nt: int(data[nt].num_nodes or 0) for nt in data.node_types}
    edges = {et: np.asarray(data[et].edge_index.cpu().numpy() if hasattr(data[et].edge_index, "cpu")
                            else data[et].edge_index) for et in data.edge_types}
    hyper = getattr(data, "hyperedges", None)
    return VideoGraph(vid, nodes, edges, hyper, expected)


def _base(rel: str) -> str:
    return rel[4:] if rel.startswith("rev_") else rel


def _pair_key(s: str, r: str, d: str) -> tuple:
    """One key for a relation and its reverse: (base relation, sorted types)."""
    a, b = sorted((s, d))
    return _base(r), a, b


def video_stats(g: VideoGraph) -> dict:
    row = {"id": g.vid, "nodes": sum(g.nodes.values())}
    for nt, n in sorted(g.nodes.items()):
        row[f"n_{nt}"] = n
    offset, o = {}, 0
    for nt in sorted(g.nodes):
        offset[nt], o = o, o + g.nodes[nt]
    n_all = o

    err = dict(self_loops=0, duplicates=0, out_of_range=0, no_reverse=0)
    directed = set()               # (global u, global v, base relation)
    pairs: dict[tuple, set] = {}   # relation key -> undirected pairs
    incoming = {nt: 0 for nt in g.nodes}
    for (s, r, d), ei in g.edges.items():
        ei = np.asarray(ei).reshape(2, -1)
        bad = (ei[0] < 0) | (ei[0] >= g.nodes.get(s, 0)) | (ei[1] < 0) | (ei[1] >= g.nodes.get(d, 0))
        err["out_of_range"] += int(bad.sum())
        ei = ei[:, ~bad]
        incoming[d] = incoming.get(d, 0) + ei.shape[1]
        key = _pair_key(s, r, d)
        for u, v in zip(ei[0] + offset[s], ei[1] + offset[d]):
            u, v = int(u), int(v)
            if u == v:
                err["self_loops"] += 1
                continue
            t = (u, v, key)
            if t in directed:
                err["duplicates"] += 1
            directed.add(t)
            pairs.setdefault(key, set()).add((min(u, v), max(u, v)))
    err["no_reverse"] = sum((v, u, k) not in directed for u, v, k in directed)

    for key, ps in sorted(pairs.items()):
        rel, a, b = key
        na, nb = g.nodes.get(a, 0), g.nodes.get(b, 0)
        possible = na * (na - 1) // 2 if a == b else na * nb
        name = f"{rel}:{a}-{b}"
        row[f"e_{name}"] = len(ps)
        row[f"density_{name}"] = round(len(ps) / possible, 4) if possible else np.nan
    row["edges"] = sum(len(p) for p in pairs.values())

    hub_ids = set(range(offset.get(HUB, 0), offset.get(HUB, 0) + g.nodes.get(HUB, 0)))
    deg, deg_nohub = np.zeros(n_all, int), np.zeros(n_all, int)
    parent = list(range(n_all))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for ps in pairs.values():
        for u, v in ps:
            deg[u] += 1
            deg[v] += 1
            if u not in hub_ids and v not in hub_ids:
                deg_nohub[u] += 1
                deg_nohub[v] += 1
                parent[find(u)] = find(v)
    rest = [i for i in range(n_all) if i not in hub_ids]
    row["isolated"] = int((deg[rest] == 0).sum()) if rest else 0
    row["isolated_without_hub"] = int((deg_nohub[rest] == 0).sum()) if rest else 0
    row["components_without_hub"] = len({find(i) for i in rest})
    row["types_without_incoming"] = "|".join(sorted(nt for nt, n in g.nodes.items()
                                                    if n > 0 and incoming.get(nt, 0) == 0))

    if g.expected is not None:
        err["node_count_mismatch"] = sum(int(g.nodes.get(nt, 0) != n) for nt, n in g.expected.items())
    if g.hyperedges is not None:
        sizes = [len(h) for h in g.hyperedges]
        members = [(nt, int(i)) for h in g.hyperedges for nt, i in h]
        covered = set(members)
        row.update(hyperedges=len(sizes), hyperedge_size_mean=round(float(np.mean(sizes)), 3) if sizes else 0,
                   hyperedge_size_max=max(sizes, default=0),
                   uncovered_nodes=sum(1 for nt, n in g.nodes.items() for i in range(n)
                                       if nt != HUB and (nt, i) not in covered))
        err["empty_hyperedges"] = sum(s == 0 for s in sizes)
        err["hyper_out_of_range"] = sum(not (0 <= i < g.nodes.get(nt, 0)) for nt, i in members)
        err["hyper_duplicate_members"] = sum(len(h) - len(set(map(tuple, h))) for h in g.hyperedges)
    row.update({f"err_{k}": v for k, v in err.items()})
    return row


def summarise_videos(df: pd.DataFrame) -> dict:
    """Medians (and min / max) of the per-video numbers; error counts summed."""
    out = {"videos": len(df)}
    for c in df.columns:
        if c == "id" or df[c].dtype == object:
            continue
        if c.startswith("err_"):
            out[c] = int(df[c].sum())
        else:
            out[f"{c}_median"] = float(df[c].median())
            if c in ("nodes", "edges", "isolated_without_hub", "components_without_hub"):
                out[f"{c}_min"], out[f"{c}_max"] = float(df[c].min()), float(df[c].max())
    if "types_without_incoming" in df:
        t = df.types_without_incoming[df.types_without_incoming != ""]
        out["videos_with_type_without_incoming"] = len(t)
        out["types_without_incoming"] = "|".join(sorted({x for s in t for x in s.split("|")}))
    return out


# ------------------------------------------------------------------ labels
def label_table(soft: dict, hard: dict, task: str) -> dict:
    """{id: (soft vector, hard class index or None, set of on categories)} for one task."""
    classes = TASKS[task]["classes"]
    out = {}
    for vid, s in soft.items():
        vec = np.array([s.get(c, 0.0) for c in classes], dtype=float)
        h = hard.get(vid)
        if TASKS[task]["type"] == "multilabel":
            out[vid] = (vec, None, set(h) if isinstance(h, list) else None)
        else:
            out[vid] = (vec, classes.index(h) if h in classes else None, None)
    return out


def adjusted_homophily(src: np.ndarray, dst: np.ndarray) -> tuple[float, float]:
    """(edge homophily, adjusted homophily) of directed edges with class labels.

    Adjusted = (h - E) / (1 - E), E the agreement expected by chance given the
    class shares at the edges' sources and targets. With both directions of an
    undirected graph this is Platonov et al.'s formula (shares = degree-weighted)."""
    if len(src) == 0:
        return np.nan, np.nan
    h = float((src == dst).mean())
    ks = np.union1d(src, dst)
    e = float(sum((src == k).mean() * (dst == k).mean() for k in ks))
    return h, (h - e) / (1 - e) if e < 1 else np.nan


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(-1)


def label_structure(q: np.ndarray, n: np.ndarray, labels: dict, task: str,
                    rng: np.random.Generator, n_perm: int = 20) -> dict:
    """Label statistics of directed links q[i] -> n[i] (ids), for one task."""
    out = {}
    have = np.array([a in labels and b in labels for a, b in zip(q, n)], bool)
    q, n = q[have], n[have]
    out[f"{task}_links_labelled"] = int(len(q))
    if not len(q):
        return out
    if TASKS[task]["type"] == "multilabel":
        cats = TASKS[task]["classes"][1:]
        hq = [labels[a][2] for a in q]
        hn = [labels[b][2] for b in n]
        ok = np.array([a is not None and b is not None for a, b in zip(hq, hn)])
        for c in cats:
            s = np.array([c in a for a, o in zip(hq, ok) if o], int)
            d = np.array([c in b for b, o in zip(hn, ok) if o], int)
            out[f"{task}_adj_homophily_{c}"] = round(adjusted_homophily(s, d)[1], 4)
    else:
        hq = np.array([labels[a][1] if labels[a][1] is not None else -1 for a in q])
        hn = np.array([labels[b][1] if labels[b][1] is not None else -1 for b in n])
        ok = (hq >= 0) & (hn >= 0)
        h, adj = adjusted_homophily(hq[ok], hn[ok])
        out[f"{task}_edge_homophily"], out[f"{task}_adj_homophily"] = round(h, 4), round(adj, 4)
    sq = np.stack([labels[a][0] for a in q])
    sn = np.stack([labels[b][0] for b in n])
    agree = 1 - 0.5 * np.abs(sq - sn).sum(1).mean()
    rand = np.mean([1 - 0.5 * np.abs(sq - sn[rng.permutation(len(sn))]).sum(1).mean() for _ in range(n_perm)])
    out[f"{task}_soft_agreement"], out[f"{task}_soft_agreement_shuffled"] = round(agree, 4), round(rand, 4)
    # disagreement: a video's annotator entropy against its neighbours' mean entropy
    eq = pd.Series(_entropy(sq), index=q).groupby(level=0).first()
    en = pd.Series(_entropy(sn), index=q).groupby(level=0).mean()
    en = en.loc[eq.index]
    out[f"{task}_entropy_spearman"] = round(float(eq.corr(en, method="spearman")), 4) \
        if len(eq) > 2 and eq.nunique() > 1 and en.nunique() > 1 else np.nan
    return out


# ------------------------------------------------------------------ links
def link_stats(links: pd.DataFrame, index_ids: set, role: dict, groups: dict | None,
               labels: dict | None, seed: int = 0) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """links: query, neighbour, source, sim, rank (one row per source that found the pair).

    role: {id: 'train' | 'val' | 'test'} for this split; index_ids: its training
    videos. Returns (summary, per-query table, per-neighbour hubness table)."""
    L = links.copy()
    L["role"] = L["query"].map(role)
    summ = {"queries": L["query"].nunique(), "link_rows": len(L)}
    summ["err_self_links"] = int((L["query"] == L.neighbour).sum())
    summ["err_neighbour_not_in_index"] = int((~L.neighbour.isin(index_ids)).sum())
    if groups is not None:
        summ["err_same_group"] = int((L["query"].map(groups) == L.neighbour.map(groups)).sum())

    pairs = L.groupby(["query", "neighbour"]).agg(n_sources=("source", "nunique"), sim=("sim", "max"),
                                                  role=("role", "first")).reset_index()
    summ["pairs"] = len(pairs)
    summ["share_pairs_found_by_2plus_sources"] = round(float((pairs.n_sources >= 2).mean()), 4)
    per_q = pairs.groupby("query").agg(role=("role", "first"), neighbours=("neighbour", "size"),
                                       sim_max=("sim", "max"), sim_mean=("sim", "mean"),
                                       multi_source=("n_sources", lambda s: int((s >= 2).sum())))
    for r, g in per_q.groupby("role"):
        summ[f"neighbours_median_{r}"] = float(g.neighbours.median())
        summ[f"sim_top1_median_{r}"] = round(float(g.sim_max.median()), 4)
        summ[f"sim_mean_median_{r}"] = round(float(g.sim_mean.median()), 4)

    indeg = pairs.neighbour.value_counts().reindex(sorted(index_ids), fill_value=0)
    hub = indeg.rename("times_neighbour").rename_axis("id").reset_index()
    x = np.sort(indeg.to_numpy())
    gini = float((2 * np.arange(1, len(x) + 1) - len(x) - 1).dot(x) / (len(x) * x.sum())) if x.sum() else 0.0
    summ.update(hub_max=int(x.max(initial=0)), hub_gini=round(gini, 4),
                share_index_never_neighbour=round(float((x == 0).mean()), 4),
                share_links_to_top1pct=round(float(np.sort(x)[::-1][:max(1, len(x) // 100)].sum() / max(x.sum(), 1)), 4))

    if labels:
        tr = pairs[pairs["query"].isin(index_ids) & pairs.neighbour.isin(index_ids)]
        summ["label_pairs_train_only"] = len(tr)
        rng = np.random.default_rng(seed)
        for task, lab in labels.items():
            summ.update(label_structure(tr["query"].to_numpy(), tr.neighbour.to_numpy(), lab, task, rng))
    return summ, per_q.reset_index(), hub


# ------------------------------------------------------------------ cost
@contextmanager
def cost_log(path: Path, approach: str, split: str, stage: str, gpus: int = 0, **extra):
    """Times the block and appends one cost record. Set rec['vlm_calls'] /
    rec['vlm_tokens'] inside the block if the stage calls the VLM."""
    rec = {"approach": approach, "split": split, "stage": stage, "gpus": gpus, "vlm_calls": 0,
           "vlm_tokens": 0, **extra}
    t = time.time()
    try:
        yield rec
    finally:
        rec["wall_s"] = round(time.time() - t, 2)
        rec["gpu_h"] = round(rec["wall_s"] * gpus / 3600, 4)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec) + "\n")


def read_costs(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["approach", "split", "stage", "wall_s", "gpu_h", "vlm_calls", "vlm_tokens"])
    return pd.DataFrame([json.loads(x) for x in open(path, encoding="utf-8") if x.strip()])


# ------------------------------------------------------------------ output
def write_stats(approach: str, split: str, out: Path = OUT, graphs: list | None = None,
                links: tuple | None = None, cost_path: Path | None = None) -> dict:
    """Write the detailed tables and upsert one summary row (approach, split)."""
    d = out / approach
    d.mkdir(parents=True, exist_ok=True)
    row = {"approach": approach, "split": split}
    if graphs:
        per_video = pd.DataFrame([video_stats(g) for g in graphs])
        per_video.to_csv(d / f"{split}_per_video.csv", index=False)
        row.update(summarise_videos(per_video))
    if links is not None:
        summ, per_q, hub = links
        per_q.to_csv(d / f"{split}_per_query.csv", index=False)
        hub.to_csv(d / f"{split}_hubness.csv", index=False)
        row.update(summ)
    if cost_path is not None:
        c = read_costs(cost_path)
        c = c[(c.approach == approach) & (c.split.isin([split, "all"]))]
        for st, g in c.groupby("stage"):
            row[f"{st}_wall_h"] = round(g.wall_s.sum() / 3600, 4)
            row[f"{st}_gpu_h"] = round(g.gpu_h.sum(), 4)
            row[f"{st}_vlm_calls"] = int(g.vlm_calls.sum())
    errs = {k: v for k, v in row.items() if k.startswith("err_") and v}
    row["errors"] = sum(errs.values())
    summary = out / "graph_stats.csv"
    old = pd.read_csv(summary) if summary.is_file() else pd.DataFrame()
    if len(old):
        old = old[~((old.approach == approach) & (old.split == split))]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(summary, index=False)
    if errs:
        print(f"  ! {approach}/{split}: {errs}")
    return row


# ------------------------------------------------------------------ selftest
def _und(pairs) -> np.ndarray:
    a = np.array(pairs, int).T
    return np.concatenate([a, a[::-1]], 1)


def selftest() -> None:
    def eq(got, want, what):
        ok = (np.isnan(got) and np.isnan(want)) if isinstance(want, float) and np.isnan(want) \
            else abs(got - want) < 1e-9 if isinstance(want, float) else got == want
        require(ok, what, expected=want, found=got)

    # 1. a clean W2-like graph: 4 fields, 2 speech, 4 clips, hub
    nodes = {"field": 4, "speech": 2, "clip": 4, HUB: 1}
    member = {}
    for nt in ("field", "speech", "clip"):
        member[(nt, "member", HUB)] = np.array([range(nodes[nt]), [0] * nodes[nt]])
        member[(HUB, "rev_member", nt)] = np.array([[0] * nodes[nt], range(nodes[nt])])
    edges = {("speech", "next", "speech"): _und([(0, 1)]),
             ("clip", "next", "clip"): _und([(0, 1), (1, 2), (2, 3)]),
             ("field", "field", "field"): _und([(i, j) for i in range(4) for j in range(i + 1, 4)]),
             ("field", "grounds", "clip"): np.array([[0], [0]]),
             ("clip", "rev_grounds", "field"): np.array([[0], [0]]), **member}
    r = video_stats(VideoGraph("toy", nodes, edges, expected={"speech": 2}))
    eq(r["nodes"], 11, "toy nodes")
    eq(r["edges"], 1 + 3 + 6 + 1 + 10, "toy undirected edges")
    eq(r["density_field:field-field"], 1.0, "complete field relation")
    eq(r["density_next:clip-clip"], 0.5, "3 of 6 clip pairs")
    eq(r["density_grounds:clip-field"], 0.0625, "1 of 16 field-clip pairs")
    eq(r["isolated"], 0, "every node touches the hub")
    eq(r["isolated_without_hub"], 0, "no isolated node without the hub either")
    eq(r["components_without_hub"], 2, "speech pair + fields joined to clips")
    eq(r["types_without_incoming"], "", "every type receives messages")
    eq(sum(v for k, v in r.items() if k.startswith("err_")), 0, "clean graph has no errors")

    # 2. the same graph with one of each error, and C1 (no edges)
    bad = dict(edges)
    bad[("clip", "next", "clip")] = np.concatenate([edges[("clip", "next", "clip")], [[2], [2]], [[0], [1]],
                                                    [[0], [9]]], 1)
    bad[("field", "grounds", "clip")] = np.array([[0, 3], [0, 3]])     # (3, 3) has no reverse
    r = video_stats(VideoGraph("bad", nodes, bad, expected={"speech": 3}))
    for k, want in dict(self_loops=1, duplicates=1, out_of_range=1, no_reverse=1, node_count_mismatch=1).items():
        eq(r[f"err_{k}"], want, f"detects {k}")
    r = video_stats(VideoGraph("c1", nodes, {}))
    eq(r["edges"], 0, "C1 has no edges")
    eq(r["isolated"], 10, "C1: every non-hub node isolated")
    eq(r["types_without_incoming"], "clip|field|hub|speech", "C1: no type receives messages")

    # 3. hypergraph: 2 hyperedges, one clip uncovered, one bad member
    hg = VideoGraph("hg", {"clip": 2, "speech": 1, "field": 1}, {},
                    hyperedges=[[("clip", 0), ("speech", 0), ("field", 0)], [("field", 0), ("clip", 5)]])
    r = video_stats(hg)
    eq(r["hyperedges"], 2, "hyperedge count")
    eq(r["hyperedge_size_max"], 3, "largest hyperedge")
    eq(r["uncovered_nodes"], 1, "clip 1 is in no hyperedge")
    eq(r["err_hyper_out_of_range"], 1, "clip 5 does not exist")

    # 4. homophily: perfect, inverted, chance
    lab = np.array([0, 0, 1, 1])
    eq(adjusted_homophily(lab[[0, 1, 2, 3]], lab[[1, 0, 3, 2]])[1], 1.0, "same-label links -> 1")
    eq(adjusted_homophily(lab[[0, 1, 2, 3]], lab[[2, 3, 0, 1]])[1], -1.0, "opposite-label links -> -1")
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.3).astype(int)                       # imbalanced classes
    s, d = rng.integers(0, 2000, 20000), rng.integers(0, 2000, 20000)
    h, adj = adjusted_homophily(y[s], y[d])
    require(h > 0.5 and abs(adj) < 0.03, "random links: raw homophily high, adjusted ~0",
            expected="h > 0.5, |adj| < 0.03", found=(round(h, 3), round(adj, 3)))
    # Platonov's undirected formula on a random undirected graph
    ei = np.concatenate([np.stack([s, d]), np.stack([d, s])], 1)
    ei = ei[:, ei[0] != ei[1]]
    hp = (y[ei[0]] == y[ei[1]]).mean()
    deg = np.bincount(ei[0], minlength=2000)
    pk = np.array([deg[y == k].sum() for k in (0, 1)]) / ei.shape[1]
    eq(round(adjusted_homophily(y[ei[0]], y[ei[1]])[1], 9), round((hp - (pk ** 2).sum()) / (1 - (pk ** 2).sum()), 9),
       "matches Platonov et al. on an undirected graph")

    # 5. links: leak checks, hubness, train-only labels
    ids = [f"t{i}" for i in range(6)] + ["v0", "x0"]
    role = {**{f"t{i}": "train" for i in range(6)}, "v0": "val", "x0": "test"}
    index = {f"t{i}" for i in range(6)}
    rows = [(q, "t0", "a", 0.9, 0) for q in ("t1", "t2", "t3", "t4", "v0", "x0")] + \
           [("t0", "t1", "a", 0.8, 0), ("t1", "t0", "b", 0.7, 0), ("v0", "v0", "a", 1.0, 1),
            ("x0", "v0", "a", 0.5, 1), ("t5", "t4", "a", 0.6, 0)]
    L = pd.DataFrame(rows, columns=["query", "neighbour", "source", "sim", "rank"])
    groups = {i: i for i in ids} | {"t5": "g", "t4": "g"}
    soft = {f"t{i}": {"YES": float(i % 2), "NO": 1 - float(i % 2)} for i in range(6)} | \
           {"v0": {"YES": 1.0, "NO": 0.0}}
    hard = {k: ("YES" if v["YES"] > 0.5 else "NO") for k, v in soft.items()}
    summ, per_q, hub = link_stats(L, index, role, groups, {"3_1": label_table(soft, hard, "3_1")})
    eq(summ["err_self_links"], 1, "v0 -> v0 detected")
    eq(summ["err_neighbour_not_in_index"], 2, "links to the val video detected")
    eq(summ["err_same_group"], 2, "same-group link (and the self-link) detected")
    eq(summ["hub_max"], 6, "t0 is everyone's neighbour")
    eq(summ["share_pairs_found_by_2plus_sources"], 0.1, "t1->t0 found by two sources (1 of 10 pairs)")
    eq(summ["label_pairs_train_only"], 6, "label stats use train-to-train pairs only")
    print("SELFTEST PASSED: structure, errors, hypergraph, homophily (incl. Platonov), links, hubness")


# ------------------------------------------------------------------ candidates
def candidates_report(cfg, path: Path, out: Path, ks=(3, 5, 10, 50)) -> None:
    """The Step 7 lists as kNN graphs (B1 / C4 before any training)."""
    import h5py
    from common import load_gold
    folds = pd.read_csv(cfg.out("folds.csv"), dtype={"id_EXIST": str})
    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str})
    groups = dict(zip(folds.id_EXIST, folds.group))
    labels = {t: label_table(*load_gold(cfg, t), t) for t in TASKS}
    h = h5py.File(path, "r")
    ids = np.array([i.decode() for i in h["ids"][()]])
    test = set(vids.id_EXIST[vids.split == "test"])
    for split in [s for s in h if s.startswith("outer")]:
        mask = h[split]["index_mask"][()].astype(bool)
        index = set(ids[mask])
        role = {i: "test" if i in test else ("train" if m else "val") for i, m in zip(ids, mask)}
        for src in h.attrs["sources"]:
            nbr, sim = h[split][src]["nbr"][()], h[split][src]["sim"][()].astype(float)
            for k in ks:
                q = np.repeat(ids, k)
                L = pd.DataFrame({"query": q, "neighbour": ids[nbr[:, :k].ravel()], "source": src,
                                  "sim": sim[:, :k].ravel(), "rank": np.tile(np.arange(k), len(ids))})
                L = L[nbr[:, :k].ravel() >= 0]
                row = write_stats(f"cand_{src}_k{k}", split, out, links=link_stats(L, index, role, groups, labels))
                print(f"  {split} {src:15s} k={k:2d}  adj.homophily 3.1 {row.get('3_1_adj_homophily'):+.3f} "
                      f"3.2 {row.get('3_2_adj_homophily'):+.3f}  hub max {row['hub_max']}  "
                      f"top1 sim train {row.get('sim_top1_median_train', np.nan):.3f} "
                      f"test {row.get('sim_top1_median_test', np.nan):.3f}")


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    c = sp.add_parser("candidates")
    c.add_argument("--config", default="config.yaml")
    c.add_argument("--candidates", default=None, help="default: outputs/candidates.h5")
    c.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        from common import load_config
        cfg = load_config(a.config)
        candidates_report(cfg, Path(a.candidates) if a.candidates else cfg.out("candidates.h5"), Path(a.out))


if __name__ == "__main__":
    main()
