"""Draw one video's B1 graph as the model sees it in one fold: its own C3 graph
(fields, speech, clips, organisers' text, hub) plus the neighbour videos that
graphs_between.attach links to the hub, one coloured edge per source that found
the neighbour (line width = cosine). Each neighbour shows its soft label and
whether it shares the author (it never may: the leak checks of attach run here).

Without --video, the first outer-fold video (in folds.csv order) is taken that
has speech and 5-10 neighbour nodes, at least one of them found by more than
one source.

Usage (Perun, package root):
    python src/plot_b1_graph.py --config config.perun.yaml --task 3_1 --fold 1 --nbr-k 5
    python src/plot_b1_graph.py --config config.perun.yaml --task 3_1 --fold 1 --nbr-k 5 --pool inner
    -> results/b1_graph/<task>_fold<k>_<pool>_k<nbr_k>_<video>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import graphs_between as gb
import graphs_within as gw
from common import TASKS, load_config, load_json, require, setup_stdout

TYPE_STYLE = {"field": ("#4C72B0", "F"), "speech": ("#55A868", "S"), "clip": ("#C44E52", "K"),
              "org": ("#8172B2", "O"), "hub": ("#222222", "hub")}
SOURCE_COLOR = {"bge_desc": "#D62E8A", "bge_transcript": "#2A9D8F", "siglip": "#9467BD"}
FIELD_NAMES = ["TARGET", "MESSAGE", "SITUATION", "PEOPLE"]


def layout(g: dict) -> dict[tuple[str, int], tuple[float, float]]:
    """Fields on top, clips in the middle, speech below (x = moment), org below the hub."""
    p = {}
    for j in range(len(g["x"]["field"])):
        p[("field", j)] = (-6 + 1.4 * j, 2.4)
    for j, m in enumerate(g["pos"]["clip"]):
        p[("clip", j)] = (-6 + 1.4 * m, 0.0)
    for j, m in enumerate(g["pos"]["speech"]):
        p[("speech", j)] = (-6 + 1.4 * m, -2.4)
    p[("org", 0)] = (-1.5, -3.5)
    p[("hub", 0)] = (0.0, 0.0)
    return p


def draw(g: dict, nbrs: list[dict], title: str, out: Path) -> None:
    """g: graphs_within.build output. nbrs: [{"label", "text", "edges": {source: (cos, rank)}}]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pos = layout(g)
    ny = np.linspace(3.2, -3.2, max(len(nbrs), 1)) if len(nbrs) > 1 else [0.0]
    for j in range(len(nbrs)):
        pos[("nbr", j)] = (4.5, float(ny[j]))
    fig, ax = plt.subplots(figsize=(15, 8))

    # inside the video: C3 = every pair; similarity relations drawn by cosine, others thin grey
    seen = set()
    for (a, r, b), (ei, attr) in g["edges"].items():
        if r.startswith("rev_") or r == "member":
            continue
        for (u, v), w in zip(ei.T.tolist(), attr[:, 0].tolist()):
            key = (a, min(u, v), b, max(u, v)) if a == b else (a, u, b, v)
            if key in seen:
                continue
            seen.add(key)
            (x0, y0), (x1, y1) = pos[(a, u)], pos[(b, v)]
            sim = r in ("grounds", "paraphrases")
            ax.plot([x0, x1], [y0, y1], color="#E07B39" if sim else "#BBBBBB",
                    lw=0.4 + 2.5 * max(w, 0) if sim else 0.7, alpha=0.35 + (0.4 * max(w, 0) if sim else 0), zorder=1)
    for t, n_ in ((t, len(x)) for t, x in g["x"].items() if t != "hub"):
        for u in range(n_):
            (x0, y0), (x1, y1) = pos[(t, u)], pos[("hub", 0)]
            ax.plot([x0, x1], [y0, y1], color="#999999", lw=0.5, ls=":", zorder=0)

    # between videos: neighbour -> hub, one edge per source, bent so parallel edges stay visible
    for j, nb in enumerate(nbrs):
        (x0, y0), (x1, y1) = pos[("nbr", j)], pos[("hub", 0)]
        for s_i, (s, (cos, rank)) in enumerate(nb["edges"].items()):
            off = (s_i - (len(nb["edges"]) - 1) / 2) * 0.35
            xs = np.linspace(x0, x1, 30)
            ys = np.linspace(y0, y1, 30) + off * np.sin(np.linspace(0, np.pi, 30))
            ax.plot(xs, ys, color=SOURCE_COLOR[s], lw=1 + 5 * max(cos, 0) ** 2, alpha=0.85, zorder=2)
            ax.text(xs[8], ys[8] + 0.12, f"{cos:.2f}·#{rank + 1}", fontsize=6.5, color=SOURCE_COLOR[s])
        ax.scatter([x0], [y0], s=900, c="#F2C14E", edgecolors="black", zorder=3)
        ax.text(x0, y0, f"N{j + 1}", ha="center", va="center", fontsize=9, weight="bold", zorder=4)
        ax.text(x0 + 0.45, y0, nb["text"], va="center", fontsize=8, family="monospace")

    for (t, u), (x, y) in pos.items():
        if t == "nbr":
            continue
        col, short = TYPE_STYLE[t]
        lab = FIELD_NAMES[u][:4] if t == "field" else ("hub" if t == "hub" else
                                                         "org" if t == "org" else f"{short}{g['pos'][t][u] + 1}")
        ax.scatter([x], [y], s=1300 if t == "hub" else 800, c=col, edgecolors="black", zorder=3)
        ax.text(x, y, lab, ha="center", va="center", fontsize=8, color="white", weight="bold", zorder=4)

    ax.text(-6, 3.3, "fields (VLM description)", fontsize=9, color=TYPE_STYLE["field"][0])
    ax.text(-8.9, -0.6, "clips\n(VideoMAE,\nper moment)", fontsize=9, color=TYPE_STYLE["clip"][0])
    ax.text(-6, -3.25, "speech (per moment with speech)", fontsize=9, color=TYPE_STYLE["speech"][0])
    ax.text(4.0, 4.0, "neighbour videos (training only)", fontsize=9)
    handles = [Line2D([], [], color=c, lw=3, label=f"edge from list: {s}") for s, c in SOURCE_COLOR.items()]
    handles += [Line2D([], [], color="#E07B39", lw=2, label="inside: similarity edge (grounds / paraphrases)"),
                Line2D([], [], color="#BBBBBB", lw=1, label="inside: other C3 pair (next, overlaps, field, ...)"),
                Line2D([], [], color="#999999", lw=1, ls=":", label="inside: member (node - hub)")]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False, bbox_to_anchor=(0.0, -0.13), ncol=3)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-9.2, 10.5)
    ax.set_ylim(-4, 4.4)
    ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--task", default="3_1", choices=list(TASKS))
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--pool", choices=["outer", "inner"], default="outer",
                    help="outer: neighbours as at prediction (outer split k); inner: as in training (inner k.1)")
    ap.add_argument("--nbr-k", type=int, default=5)
    ap.add_argument("--video", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--out-dir", default="results/b1_graph")
    a = ap.parse_args()
    import torch  # noqa: F401  (attach builds torch tensors)
    from train import ES_INNER, to_hetero

    cfg = load_config(a.config)
    task, classes, k = a.task, TASKS[a.task]["classes"], a.fold
    ft, cal, ids = gw.load_inputs(cfg, Path(a.features) if a.features else None, "e5", False)
    folds = pd.read_csv(cfg.out("folds.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST")
    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST")
    train_ids = [i for i in ids if vids.loc[i, "split"] == "train"]
    test_ids = [i for i in ids if vids.loc[i, "split"] == "test"]
    soft = {str(r["id"]): np.array([r["value"].get(c, 0.0) for c in classes], np.float32)
            for r in load_json(cfg.gold_path(task, "soft"))}
    require(all(i in soft for i in train_ids), "every training video has a soft label", expected=0,
            found=sum(i not in soft for i in train_ids))
    nb = gb.Between(Path(a.candidates) if a.candidates else cfg.out("candidates.h5"), ft, ids,
                    {i: soft[i] for i in train_ids}, train_ids, len(classes))

    own = folds.outer_fold == k
    inner1 = {i for i in train_ids if not own[i] and folds[f"inner_for_outer_{k}"][i] == ES_INNER}
    outer_all = {i for i in train_ids if own[i]}
    if a.pool == "outer":
        split, forbidden = f"outer{k}", outer_all | set(test_ids)
        pool_ids = [i for i in train_ids if own[i] and folds.val_eligible[i]]
    else:
        split, forbidden = f"inner{k}.{ES_INNER}", outer_all | inner1 | set(test_ids)
        pool_ids = [i for i in train_ids if not own[i] and folds[f"inner_for_outer_{k}"][i] != ES_INNER]

    def attach(v):
        g = gw.build(ft, v, "C3", cal)
        h = nb.attach({v: to_hetero(g, None)}, [v], split, a.nbr_k, forbidden)[v]
        return g, h

    if a.video:
        require(a.video in pool_ids, f"--video is a fold-{k} {'outer' if a.pool == 'outer' else 'training'} video",
                expected="in the pool", found=a.video)
        video = a.video
        g, h = attach(video)
    else:
        video = None
        for v in pool_ids:
            if not ft.mask[ft.row[v]].sum() >= 2:
                continue
            g, h = attach(v)
            n_nb = int(h["nbr"].num_nodes)
            multi = n_nb < sum(h[et].edge_index.shape[1] for et in gb.edge_types())
            if 5 <= n_nb <= 10 and multi:
                video = v
                break
        require(video is not None, "a video with 5-10 neighbours, one found by several sources",
                expected="one", found="none (raise --nbr-k or pass --video)")

    vid_rows = h["nbr"].vid.tolist()
    nbrs = [{"edges": {}} for _ in vid_rows]
    for s in gb.SOURCES:
        e = h[("nbr", f"near_{s}", "hub")]
        for src, (cos, rank) in zip(e.edge_index[0].tolist(), e.edge_attr.tolist()):
            nbrs[src]["edges"][s] = (cos, int(round(rank * gb.MAX_K)))
    author = vids.loc[video, "author"]
    for j, r in enumerate(vid_rows):
        nid = ids[r]
        lab = " ".join(f"{c[:4]} {p:.2f}" for c, p in zip(classes, nb.lab[r]))
        same = vids.loc[nid, "author"] == author
        require(not same, f"neighbour {nid} has another author than {video}", expected="other", found=author)
        nbrs[j]["text"] = f"{nid}  [{lab}]  fold {folds.outer_fold[nid]}"
    own_lab = " ".join(f"{c} {p:.2f}" for c, p in zip(classes, soft[video])) if video in soft else "no label"
    title = (f"B1 graph of video {video} (task {task}, outer fold {k}, outer fold of the video: "
             f"{folds.outer_fold[video]}) | pool {split} ({'prediction' if a.pool == 'outer' else 'training'}), "
             f"nbr_k {a.nbr_k} per list -> {len(vid_rows)} neighbour nodes, "
             f"{sum(len(n['edges']) for n in nbrs)} edges\n"
             f"own soft label (the model never sees it): {own_lab}  |  "
             f"never a neighbour: itself, its author, outer fold {k}"
             + (", inner fold 1" if a.pool == "inner" else "") + ", the test set")
    out = Path(a.out_dir) / f"{task}_fold{k}_{a.pool}_k{a.nbr_k}_{video}.png"
    draw(g, nbrs, title, out)
    print(title)
    for j, n_ in enumerate(nbrs):
        print(f"  N{j + 1}: {n_['text']}  sources {', '.join(f'{s} cos {c:.3f} rank {r + 1}' for s, (c, r) in n_['edges'].items())}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
