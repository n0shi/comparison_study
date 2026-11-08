"""Message-function check (choices.md): bootstrap of C3 / W2 against C1 under GATv2,
wmean and mean, and of C1 against C1 (the same network, no edges: the noise floor).

Reads the seed ensembles graph_report.py wrote (results/graphs_v2*/ensemble) and
writes results/conv_check_compare.csv.

Usage:
    python src/conv_check.py --config config.yaml --n-boot 1000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

import evaluate as ev
from common import TASKS, load_config, setup_stdout
from graph_report import THRESHOLD

VARIANTS = {"gat": "graphs_v2", "wmean": "graphs_v2_wmean", "mean": "graphs_v2_mean"}


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = Path(a.results)
    pairs = []
    for c in VARIANTS:
        pairs += [(f"C3 {c}", f"C1 {c}"), (f"W2 {c}", f"C1 {c}")]
    pairs += [("C1 wmean", "C1 gat"), ("C1 mean", "C1 gat"), ("C1 mean", "C1 wmean")]   # same network
    pairs += [("C3 wmean", "C3 gat"), ("C3 mean", "C3 gat"), ("W2 wmean", "W2 gat"), ("W2 mean", "W2 gat")]
    rows = []
    for task in TASKS:
        gold = ev.load_gold_frame(cfg, task)
        pred = {f"{ap_} {c}": ev.load_pred(root / d / "ensemble" / f"{ap_}_{task}_tuned+e5_oof.csv", gold,
                                           THRESHOLD[task])
                for c, d in VARIANTS.items() for ap_ in ("C1", "C3", "W2")}
        for b, ref in pairs:
            t = time.time()
            res = ev.bootstrap_diff(gold, *pred[ref], *pred[b], a.n_boot, 0, "all")
            for k in ("icm_soft_norm", "auc"):
                lo, hi = res[k]["ci95"]
                rows.append({"task": task, "pair": f"{b} - {ref}", "metric": k, "diff": round(res[k]["diff"], 4),
                             "lo": round(lo, 4), "hi": round(hi, 4), "sig": "*" if lo > 0 or hi < 0 else ""})
            print(f"  {task} {b} - {ref}: {time.time() - t:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(root / "conv_check_compare.csv", index=False)
    df["cell"] = df["diff"].map("{:+.4f}".format) + df.sig
    for k in ("icm_soft_norm", "auc"):
        print(f"\n{k} (all videos; * = 95 % interval excludes 0)")
        print(df[df.metric == k].pivot(index="pair", columns="task", values="cell").to_string())
    print(f"\nwrote {root / 'conv_check_compare.csv'}")


if __name__ == "__main__":
    main()
