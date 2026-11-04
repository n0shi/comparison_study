"""Checks on finished out-of-fold predictions; nothing is trained (CPU, minutes).

  controls  C0 made exactly as uncertain as the graph, with no new information
            (blend towards the fold's training prior, or a temperature, matched
            to the graph's mean entropy; evaluate.py's controls). Graph minus
            control shows whether the graph locates disagreement or is merely
            less confident than C0 (data_note 13).
  blend     late blend with the kNN vote C4: p = (1 - w) p_model + w p_C4, the
            weight per outer fold chosen on the other four folds' OOF by soft
            cross-entropy (the criterion C4 itself was tuned with). For the graph
            this is the reference a between-video approach must beat
            (choices.md, decision rules); for C0 it is Step 28's "C0 plus kNN".
  temper    one temperature per outer fold, fitted on the other four folds' OOF
            by soft cross-entropy, for the graph and for C0: does calibration
            alone close the ICM-Soft-Norm gap to C0? A diagnostic for Step 27,
            never used to choose a structure.

Every comparison is a paired author bootstrap (evaluate.bootstrap_diff), B - A.

Inputs (local paths as written by graph_report.py and c4_knn_vote.py):
    results/graphs_v2/ensemble/{graph}_{task}_tuned+e5_oof.csv
    results/C0/ensemble/C0_quad_{task}_oof.csv
    runs/C4/C4_{task}/oof_outer.csv

Usage:
    python src/late_checks.py --config config.yaml
    python src/late_checks.py --config config.yaml --graph C3 --tasks 3_1 --n-boot 500
Output: results/late_checks/checks.csv, params.json, and the blended / tempered
predictions as CSVs (evaluate.py's prediction format).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate as ev
from common import TASKS, load_config, require, setup_stdout
from graph_report import THRESHOLD

W_GRID = np.linspace(0, 1, 41)                    # blend weight of C4
T_GRID = np.exp(np.linspace(-1.5, 1.5, 301))      # temperature, ~0.22 .. 4.5
METRICS = ["icm_soft_norm", "icm_norm", "auc", "ce"]


def soft_ce(p: np.ndarray, g: np.ndarray, kind: str) -> float:
    q = np.clip(p, ev.EPS, 1 - ev.EPS)
    if kind == "multilabel":
        return float(-(g * np.log(q) + (1 - g) * np.log(1 - q)).mean(1).mean())
    return float(-(g * np.log(q)).sum(1).mean())


def per_fold(fold: np.ndarray, gold: ev.Gold, kind: str, grid, make) -> tuple[np.ndarray, dict]:
    """For every outer fold k: the grid value with the lowest soft CE on the
    other four folds, applied to fold k. make(x, rows) -> predictions for rows."""
    out = np.zeros_like(gold.soft, dtype=float)
    chosen = {}
    for k in np.unique(fold):
        rest, own = fold != k, fold == k
        ces = [soft_ce(make(x, rest), gold.soft[rest], kind) for x in grid]
        x = grid[int(np.argmin(ces))]
        out[own] = make(x, own)
        chosen[int(k)] = round(float(x), 4)
    return out, chosen


def renorm(p: np.ndarray, kind: str) -> np.ndarray:
    return p if kind == "multilabel" else p / p.sum(1, keepdims=True)


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--graph", default="C3", help="graph approach (C3 = the base, choices.md)")
    ap.add_argument("--hyper", default="tuned+e5")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--graph-dir", default="results/graphs_v2/ensemble")
    ap.add_argument("--c0-dir", default="results/C0/ensemble")
    ap.add_argument("--c4-dir", default="runs/C4")
    ap.add_argument("--out", default="results/late_checks")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--subsets", nargs="+", default=["all", "contested"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(cfg.out("folds.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST")

    rows, params = [], {}
    for task in a.tasks:
        t0 = time.time()
        kind, classes, thr = TASKS[task]["type"], TASKS[task]["classes"], THRESHOLD[task]
        gold = ev.load_gold_frame(cfg, task)
        fold = folds.loc[gold.ids, "outer_fold"].to_numpy()
        paths = {"graph": Path(a.graph_dir) / f"{a.graph}_{task}_{a.hyper}_oof.csv",
                 "C0": Path(a.c0_dir) / f"C0_quad_{task}_oof.csv",
                 "C4": Path(a.c4_dir) / f"C4_{task}" / "oof_outer.csv"}
        for name, p in paths.items():
            require(p.is_file(), f"{name} OOF predictions exist for task {task}", expected=str(p), found="missing")
        P = {name: ev.load_pred(p, gold, thr) for name, p in paths.items()}
        G = a.graph
        preds = {G: P["graph"], "C0": P["C0"], "C4": P["C4"]}
        prm = {}

        # ---- controls: C0 at the graph's mean entropy
        target = ev.entropy(P["graph"][0], kind).mean()
        prior = ev.fold_prior(cfg, gold, "outer_fold")
        pc0 = P["C0"][0]
        lam, err = ev.match_entropy(lambda l: ev.blend(pc0, prior, l), np.linspace(0, 1, 1001), target, kind)
        if err < 1e-3:
            preds["C0 prior-blend"] = (ev.blend(pc0, prior, lam), None)
            prm["C0 prior-blend lambda"] = round(float(lam), 4)
        else:
            print(f"  {task}: prior blend cannot reach the graph's entropy (err {err:.4f}); skipped")
        T, err = ev.match_entropy(lambda t: ev.temper(pc0, t, kind), np.exp(np.linspace(-2, 2.5, 2001)),
                                  target, kind)
        preds["C0 entropy-T"] = (ev.temper(pc0, T, kind), None)
        prm["C0 entropy-T"] = round(float(T), 4)

        # ---- late blend with C4, weight per fold from the other four folds
        p4 = P["C4"][0]
        for name in (G, "C0"):
            pm = preds[name][0]
            pb, w = per_fold(fold, gold, kind, W_GRID,
                             lambda x, r, pm=pm: renorm((1 - x) * pm[r] + x * p4[r], kind))
            preds[f"{name}+C4"] = (pb, None)
            prm[f"{name}+C4 weight of C4 per fold"] = w

        # ---- temperature per fold by soft CE (diagnostic)
        for name in (G, "C0"):
            pm = preds[name][0]
            pt, Ts = per_fold(fold, gold, kind, T_GRID, lambda x, r, pm=pm: ev.temper(pm[r], x, kind))
            preds[f"{name} T"] = (pt, None)
            prm[f"{name} T per fold"] = Ts

        for name, (p, h) in list(preds.items()):
            if h is None:
                preds[name] = (p, ev.derive_hard(p, task, thr))
                df = pd.DataFrame(p, columns=[f"p_{c}" for c in classes])
                df.insert(0, "id_EXIST", gold.ids)
                df.to_csv(out / f"{name.replace(' ', '_').replace('+', 'plus')}_{task}_oof.csv", index=False)
        params[task] = prm

        pairs = [  # (check, A, B): reported as B - A
            ("controls", "C0 prior-blend", G), ("controls", "C0 entropy-T", G),
            ("blend", G, f"{G}+C4"), ("blend", "C4", f"{G}+C4"), ("blend", "C0", f"{G}+C4"),
            ("blend", "C0", "C0+C4"), ("blend", "C0+C4", f"{G}+C4"),
            ("temper", G, f"{G} T"), ("temper", "C0", "C0 T"), ("temper", "C0 T", f"{G} T"),
        ]
        for check, A, B in pairs:
            if A not in preds or B not in preds:
                continue
            for s in a.subsets:
                r = ev.bootstrap_diff(gold, *preds[A], *preds[B], a.n_boot, 0, s)
                for k in METRICS:
                    lo, hi = r[k]["ci95"]
                    rows.append({"task": task, "check": check, "B": B, "A": A, "subset": s, "metric": k,
                                 "A_score": round(r[k]["a"], 4), "B_score": round(r[k]["b"], 4),
                                 "diff": round(r[k]["diff"], 4), "ci95": [round(lo, 4), round(hi, 4)],
                                 "sig": "*" if lo > 0 or hi < 0 else ""})
        print(f"  task {task} done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "checks.csv", index=False)
    with open(out / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=1)
    print("\nchosen parameters:")
    print(json.dumps(params, indent=1))
    pd.set_option("display.width", 200)
    for check, g in df.groupby("check", sort=False):
        print(f"\n== {check}: B - A (author bootstrap, {a.n_boot} resamples; * = 95 % CI excludes 0)")
        print(g.drop(columns="check").to_string(index=False))
    print(f"\nwrote {out}/checks.csv, params.json and the transformed predictions")


if __name__ == "__main__":
    main()
