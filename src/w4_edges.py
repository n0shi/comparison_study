"""Inspect the edges W4's trained scorer keeps (Task 5.3): learned_edges_fold{k}.jsonl.gz
of every finished W4 seed run (train.py).

Per task it prints
  - the k each fold chose (manifest) and the most edges any receiving node keeps
  - per relation: share of videos with at least one kept edge, mean kept edges per
    video, and the score quantiles (score = the scorer's weighted cosine, in [-1, 1])
  - agreement between seeds: mean Jaccard of a video's kept edge set, same fold,
    every pair of seeds (1 = the seeds keep the same edges)
and writes the per-relation table to results/graphs_v2/w4_edges.csv.

Usage:
    python src/w4_edges.py [--runs runs/graphs_v2/W4] [--out results/graphs_v2/w4_edges.csv]
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common import require, setup_stdout


def load(path: Path) -> dict[str, dict[str, list]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return {r["id"]: r["edges"] for r in map(json.loads, f)}


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/graphs_v2/W4")
    ap.add_argument("--out", default="results/graphs_v2/w4_edges.csv")
    a = ap.parse_args()
    root = Path(a.runs)
    dirs = sorted(d for d in root.glob("W4_*_seed*_tuned_e5") if (d / "manifest.json").is_file())
    require(len(dirs) > 0, "finished W4 seed runs", expected=">0", found=0)
    rows = []
    for task in sorted({json.load(open(d / "manifest.json"))["task"] for d in dirs}):
        runs = [d for d in dirs if json.load(open(d / "manifest.json"))["task"] == task]
        man = json.load(open(runs[0] / "manifest.json"))
        ks = {int(f): v["hyper"].get("k") for f, v in man["folds"].items()}
        print(f"\n=== task {task}: {len(runs)} seeds; k per fold {ks}")
        per_rel = defaultdict(lambda: {"videos": 0, "edges": 0, "scores": []})
        n_videos = 0
        max_in = []
        jac = []
        for fold in sorted(ks):
            seeds = [load(d / f"learned_edges_fold{fold}.jsonl.gz") for d in runs
                     if (d / f"learned_edges_fold{fold}.jsonl.gz").is_file()]
            require(len(seeds) == len(runs), f"task {task} fold {fold}: an edge file per seed",
                    expected=len(runs), found=len(seeds))
            for vid, edges in seeds[0].items():
                n_videos += 1
                indeg = defaultdict(int)
                for key, es in edges.items():
                    s = per_rel[key]
                    s["videos"] += 1
                    s["edges"] += len(es)
                    s["scores"] += [e[2] for e in es]
                    for _, v, _ in es:
                        indeg[(key.split("|")[2], v)] += 1
                max_in.append(max(indeg.values(), default=0))
                sets = [{(key, u, v) for key, es in sd.get(vid, {}).items() for u, v, _ in es} for sd in seeds]
                for x, y in itertools.combinations(sets, 2):
                    if x | y:
                        jac.append(len(x & y) / len(x | y))
        print(f"  most kept edges at one receiving node: max {max(max_in)}, "
              f"median {int(np.median(max_in))} (k bounds it)")
        print(f"  seed agreement (Jaccard of kept edges, same fold): mean {np.mean(jac):.3f}, "
              f"10th pct {np.percentile(jac, 10):.3f}")
        for key, s in sorted(per_rel.items(), key=lambda kv: -kv[1]["edges"]):
            q = np.percentile(s["scores"], [5, 50, 95])
            row = {"task": task, "relation": key, "share_videos": round(s["videos"] / n_videos, 3),
                   "edges_per_video": round(s["edges"] / n_videos, 2),
                   "score_p5": round(q[0], 3), "score_p50": round(q[1], 3), "score_p95": round(q[2], 3)}
            rows.append(row)
            print(f"  {key:40s} videos {row['share_videos']:.2f}  edges/video {row['edges_per_video']:5.2f}  "
                  f"score p5/p50/p95 {q[0]:+.3f} {q[1]:+.3f} {q[2]:+.3f}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
