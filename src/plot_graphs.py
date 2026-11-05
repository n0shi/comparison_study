"""Draw one video's graph as each inside-video approach builds it, side by side:
C1 (no edges), C2 (random), W2 (similarity rules), W3 (VLM-chosen), C3 (every
pair) and W4 (what the trained scorer kept). Same node layout in every panel,
one colour per relation, so the panels differ only in their edges.

W4's edges come from a finished seed run: learned_edges_fold<f>.jsonl.gz, where
f is the video's own outer fold (the fold whose model never trained on it).
The hub's member edges (every node - hub) are the same in all panels and are
drawn faintly. B1 adds neighbour videos to the hub of C3: plot_b1_graph.py.

Without --video, the first outer-fold-1 video (folds.csv order) is taken that
has 3+ moments with speech and W3 edges that differ from W2's.

Usage (Perun, package root):
    python src/plot_graphs.py --config config.perun.yaml
    python src/plot_graphs.py --config config.perun.yaml --video 121369 --task 3_1
    -> results/graph_views/<video>.png, plus edge counts per relation printed
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

import graphs_within as gw
from common import TASKS, load_config, require, setup_stdout
from plot_b1_graph import FIELD_NAMES, TYPE_STYLE, layout

REL_STYLE = {           # relation -> (colour, label)
    "next": ("#555555", "next: consecutive moments"),
    "overlaps": ("#8C564B", "overlaps: speech - clip, same moment"),
    "field": ("#4C72B0", "field: field - field"),
    "grounds": ("#E07B39", "grounds: field - clip (SigLIP)"),
    "paraphrases": ("#2A9D8F", "paraphrases: field - speech (BGE)"),
}
PANELS = ["C1", "C2", "W2", "W3", "C3", "W4"]
CAPTION = {"C1": "C1: no edges (each node alone, then pooled)",
           "C2": "C2: random edges, as many per relation as W2",
           "W2": "W2: time order + similarity rules",
           "W3": "W3: W2, but the VLM picks grounds / paraphrases",
           "C3": "C3: every pair (typed), cosine as edge feature",
           "W4": "W4: the trained scorer keeps the top k per node"}


def undirected(edges: dict) -> list[tuple]:
    """[(src type, rel, dst type, u, v, weight)] once per pair, rev_ and mirrored copies dropped."""
    out, seen = [], set()
    for (a, r, b), (ei, attr) in edges.items():
        if r.startswith("rev_") or r == "member":
            continue
        for (u, v), w in zip(np.asarray(ei).T.tolist(), np.asarray(attr).reshape(-1).tolist()):
            key = (a, r, b, min(u, v), max(u, v)) if a == b else (a, r, b, u, v)
            if key not in seen:
                seen.add(key)
                out.append((a, r, b, u, v, w))
    return out


def w4_edges(path: Path, video: str) -> dict:
    """The kept edges of one video from a W4 dump, in build()'s (ei, attr) form;
    a reverse edge counts for its forward pair."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        row = next((r for r in map(json.loads, f) if r["id"] == video), None)
    require(row is not None, f"{video} in {path}", expected="a row", found="none")
    out = {}
    for key, es in row["edges"].items():
        a, r, b = key.split("|")
        if r.startswith("rev_"):
            a, r, b = b, r[4:], a
            es = [[v, u, s] for u, v, s in es]
        ei, at = out.get((a, r, b), ([[], []], []))
        for u, v, s in es:
            ei[0].append(u)
            ei[1].append(v)
            at.append(s)
        out[(a, r, b)] = (ei, at)
    return {k: (np.array(ei, dtype=np.int64).reshape(2, -1), np.array(at, np.float32)) for k, (ei, at) in out.items()}


def draw_panel(ax, g: dict, edges: dict, title: str, weighted: bool) -> int:
    pos = layout(g)
    pos[("org", 0)] = (-1.5, -3.6)
    for t in ("field", "speech", "clip", "org"):
        for u in range(len(g["x"][t])):
            (x0, y0), (x1, y1) = pos[(t, u)], pos[("hub", 0)]
            ax.plot([x0, x1], [y0, y1], color="#CCCCCC", lw=0.5, ls=":", zorder=0)
    es = undirected(edges)
    for a, r, b, u, v, w in es:
        (x0, y0), (x1, y1) = pos[(a, u)], pos[(b, v)]
        col = REL_STYLE[r][0]
        lw = 0.6 + 3.0 * max(w, 0) if weighted and r in ("grounds", "paraphrases") else 1.4
        if a == b and abs(v - u) > 1 and y0 == y1:          # same-row pairs further apart: arc
            xs = np.linspace(x0, x1, 30)
            ys = y0 + 0.25 * abs(v - u) * np.sin(np.linspace(0, np.pi, 30))
            ax.plot(xs, ys, color=col, lw=lw, alpha=0.8, zorder=1)
        else:
            ax.plot([x0, x1], [y0, y1], color=col, lw=lw, alpha=0.75, zorder=1)
    spoken = set(int(m) for m in g["pos"]["speech"])
    for (t, u), (x, y) in pos.items():
        col, short = TYPE_STYLE[t]
        lab = FIELD_NAMES[u][:4] if t == "field" else (t if t in ("hub", "org") else f"{short}{g['pos'][t][u] + 1}")
        ax.scatter([x], [y], s=520 if t != "hub" else 760, c=col, edgecolors="black", zorder=3)
        ax.text(x, y, lab, ha="center", va="center", fontsize=6.5, color="white", weight="bold", zorder=4)
    for m in range(len(g["x"]["clip"])):
        if m not in spoken:
            ax.scatter([-6 + 1.4 * m], [-2.4], s=520, facecolors="none", edgecolors="#55A868", linestyles="--")
            ax.text(-6 + 1.4 * m, -2.4, f"S{m + 1}\nnone", ha="center", va="center", fontsize=5.5, color="#55A868")
    ax.set_title(f"{title}\n{len(es)} edges (+ {sum(len(g['x'][t]) for t in ('field', 'speech', 'clip', 'org'))}"
                 f" to the hub)", fontsize=9)
    ax.set_xlim(-6.8, 0.8)
    ax.set_ylim(-4.2, 3.2)
    ax.axis("off")
    return len(es)


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--task", default="3_1", choices=list(TASKS), help="which W4 run's edges to show")
    ap.add_argument("--video", default=None)
    ap.add_argument("--seed", type=int, default=42, help="W4 seed run")
    ap.add_argument("--features", default=None)
    ap.add_argument("--w3", default=None)
    ap.add_argument("--runs", default="runs/graphs_v2")
    ap.add_argument("--out-dir", default="results/graph_views")
    a = ap.parse_args()

    cfg = load_config(a.config)
    ft, cal, ids = gw.load_inputs(cfg, Path(a.features) if a.features else None, "e5", False)
    gw.load_w3(ft, Path(a.w3) if a.w3 else cfg.out("w3/edges.jsonl"))
    folds = pd.read_csv(cfg.out("folds.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST")
    if a.video:
        require(a.video in folds.index, "--video is a training video", expected="in folds.csv", found=a.video)
        video = a.video
    else:
        video = None
        for v in folds.index[(folds.outer_fold == 1) & folds.val_eligible.astype(bool)]:
            if ft.mask[ft.row[v]].sum() < 3:
                continue
            e2, e3 = gw.build(ft, v, "W2", cal)["edges"], gw.build(ft, v, "W3", cal)["edges"]
            if any(e2[et][0].shape != e3[et][0].shape or not np.array_equal(e2[et][0], e3[et][0]) for et in e2):
                video = v
                break
        require(video is not None, "a fold-1 video with 3+ speech moments and W3 != W2", expected="one", found="none")
    fold = int(folds.outer_fold[video])
    w4_dir = Path(a.runs) / "W4" / f"W4_{a.task}_seed{a.seed}_tuned_e5"
    w4_path = w4_dir / f"learned_edges_fold{fold}.jsonl.gz"
    require(w4_path.is_file(), f"{w4_path} exists (finished W4 run)", expected="file", found="missing")
    k = json.load(open(w4_dir / "manifest.json"))["folds"][str(fold)]["hyper"]["k"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    counts = {}
    base = gw.build(ft, video, "C3", cal)
    for ax, appr in zip(axes.flat, PANELS):
        if appr == "W4":
            g, edges = base, w4_edges(w4_path, video)
            title = f"{CAPTION[appr]} (task {a.task}, fold {fold}, k = {k}; width = score)"
        else:
            g = gw.build(ft, video, appr, cal)
            edges, title = g["edges"], CAPTION[appr]
        counts[appr] = draw_panel(ax, g, edges, title, weighted=appr in ("W2", "W3", "C3", "W4"))
        rel = {}
        for _, r, _, *_ in undirected(edges):
            rel[r] = rel.get(r, 0) + 1
        print(f"  {appr:3s} {counts[appr]:3d} edges  " + "  ".join(f"{r} {n}" for r, n in sorted(rel.items())))
    handles = [Line2D([], [], color=c, lw=2.5, label=l) for c, l in REL_STYLE.values()]
    handles.append(Line2D([], [], color="#CCCCCC", lw=1, ls=":", label="member: every node - hub (all panels)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle(f"Video {video}: the same nodes, six ways to connect them | rows: fields (VLM description), "
                 f"clips (VideoMAE), speech (ASR per moment); org = organisers' text", fontsize=11)
    out = Path(a.out_dir) / f"{video}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"video {video} (outer fold {fold})\nwrote {out}")


if __name__ == "__main__":
    main()
