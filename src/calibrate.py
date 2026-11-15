"""Step 27, calibration on the predictions already made (no training).

For every seed ensemble of a results folder (graph_report.py) and C0 quad, the
out-of-fold predictions are post-processed by cross-fitting: the setting used
on outer fold k is chosen on the other four folds, so fold k's labels are not
used for it.

  T_ce    temperature that minimises the soft cross-entropy (the usual
          temperature scaling: a proper score, so it measures miscalibration)
  T_icm   temperature that maximises ICM-Soft-Norm, the official soft metric
          (the sharpness of sharp_check.py, here as a submission setting)
  thr     task 3.3 only: one category threshold for all categories that
          maximises ICM-Norm, and one threshold per category that maximises
          that category's F1; both change the hard decisions only

Temperatures leave the hard decisions as they are. T_all / thr_all are the
same settings fitted on all five folds: what a test submission would use.

Writes calibration.csv in the results folder; with --n-boot, the paired author
bootstrap of each setting against the raw predictions for --boot approaches.

Usage:
    python src/calibrate.py --config config.perun.yaml --results results/graphs_v3
    python src/calibrate.py --config config.yaml --results results/graphs_v2 --tasks 3_3 --n-boot 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import evaluate as ev
from common import TASKS, load_config, require, setup_stdout
from graph_report import THRESHOLD
from sharp_check import GRID, HYPER

T_GRID = GRID + [2.0, 3.0]
THR_GRID = [round(0.05 * i, 2) for i in range(2, 15)]          # 0.10 .. 0.70
SOFT = ["icm_soft_norm", "ce", "brier", "ece", "auc"]
HARD = ["icm_norm", "f1_macro"]


def _masks(fold):
    """Weights of "every fold but k" for each outer fold k, and of all folds (key None)."""
    return {**{int(k): (fold != k).astype(float) for k in sorted(set(fold))}, None: np.ones(len(fold))}


def fit_temperatures(gold, p, hard, fold, kind):
    """Cross-fitted temperatures, one Evaluator per temperature: {key: (tempered predictions,
    temperature of each fold, temperature of all folds)} for ce (lowest) and icm_soft_norm
    (highest)."""
    every = np.ones(len(fold), bool)
    tempered = {T: ev.temper(p, T, kind) for T in T_GRID}
    evs = {T: ev.Evaluator(gold, tempered[T], hard, every) for T in T_GRID}
    sc = {k: {T: evs[T].scores(w) for T in T_GRID} for k, w in _masks(fold).items()}
    out = {}
    for key, sign in (("ce", 1), ("icm_soft_norm", -1)):
        best = {k: T_GRID[int(np.argmin([sign * sc[k][T][key] for T in T_GRID]))] for k in sc}
        q = p.copy()
        for k in best:
            if k is not None:
                q[fold == k] = tempered[best[k]][fold == k]
        out[key] = (q, {k: t for k, t in best.items() if k is not None}, best[None])
    return out


def fit_threshold(gold, p, task, fold):
    """Task 3.3: cross-fitted common threshold (ICM-Norm) -> hard, per fold, on all folds."""
    every = np.ones(len(fold), bool)
    hards = {t: ev.derive_hard(p, task, t) for t in THR_GRID}
    evs = {t: ev.Evaluator(gold, p, hards[t], every) for t in THR_GRID}
    best = {k: THR_GRID[int(np.argmax([evs[t].scores(w)["icm_norm"] for t in THR_GRID]))]
            for k, w in _masks(fold).items()}
    out = [None] * len(fold)
    for k, t in best.items():
        if k is not None:
            for i in np.flatnonzero(fold == k):
                out[i] = hards[t][i]
    return out, {k: t for k, t in best.items() if k is not None}, best[None]


def fit_class_thresholds(gold, p, task, fold):
    """Task 3.3: cross-fitted threshold per category (that category's F1) -> hard, on all folds."""
    classes = TASKS[task]["classes"]
    known = np.array([h is not None for h in gold.hard])
    y = np.array([[c in (h or []) for c in classes] for h in gold.hard])

    def best(mask):
        m = mask & known
        return [THR_GRID[int(np.argmax([f1_score(y[m, c], p[m, c] >= t, zero_division=0) for t in THR_GRID]))]
                for c in range(1, len(classes))]
    out, every = [None] * len(fold), np.ones(len(fold), bool)
    for k in sorted(set(fold)):
        thr = best(fold != k)
        for i in np.flatnonzero(fold == k):
            cats = [c for c, v, t in zip(classes[1:], p[i, 1:], thr) if v >= t]
            out[i] = cats or ["NO"]
    return out, dict(zip(classes[1:], best(every)))


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--results", default="results/graphs_v3")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot", nargs="*", default=["C1", "C0 quad"], help="approaches bootstrapped against raw")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = Path(a.results)
    folds = pd.read_csv(cfg.out("folds.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST")
    rows, boot = [], []
    for task in a.tasks:
        gold = ev.load_gold_frame(cfg, task)
        kind = TASKS[task]["type"]
        fold = folds.outer_fold.loc[gold.ids].to_numpy()
        m = ev.subsets(gold)["all"]
        files = {f.name[:-len(f"_{task}_{HYPER}_oof.csv")]: f
                 for f in sorted((root / "ensemble").glob(f"*_{task}_{HYPER}_oof.csv"))}
        require(files, f"seed ensembles of task {task} in {root / 'ensemble'}", expected=">0", found=0)
        c0 = Path("results/C0/ensemble") / f"C0_quad_{task}_oof.csv"
        if c0.is_file():
            files["C0 quad"] = c0
        for name, f in files.items():
            p, h = ev.load_pred(f, gold, THRESHOLD[task])
            variants = {"raw": (p, h, "", "")}
            fits = fit_temperatures(gold, p, h, fold, kind)
            for key, tag in (("ce", "T_ce"), ("icm_soft_norm", "T_icm")):
                q, temps, t_all = fits[key]
                variants[tag] = (q, h, [temps[k] for k in sorted(temps)], t_all)
            if kind == "multilabel":
                h2, thr, thr_all = fit_threshold(gold, p, task, fold)
                variants["thr"] = (p, h2, [thr[k] for k in sorted(thr)], thr_all)
                h3, per_cat = fit_class_thresholds(gold, p, task, fold)
                variants["thr_per_cat"] = (p, h3, "", per_cat)
            for tag, (q, hh, per_fold, on_all) in variants.items():
                sc = ev.Evaluator(gold, q, hh, m).scores()
                rows.append({"task": task, "approach": name, "setting": tag,
                             **{k: round(sc[k], 4) for k in SOFT + HARD},
                             "per_fold": per_fold, "on_all_folds": on_all})
                if tag != "raw" and a.n_boot > 0 and name in a.boot:
                    res = ev.bootstrap_diff(gold, p, h, q, hh, a.n_boot, 0, "all")
                    for k in (HARD if tag.startswith("thr") else SOFT):
                        if k in res:
                            lo, hi = res[k]["ci95"]
                            boot.append({"task": task, "approach": name, "setting": tag, "metric": k,
                                         "diff": round(res[k]["diff"], 4), "lo": round(lo, 4), "hi": round(hi, 4),
                                         "sig": "*" if lo > 0 or hi < 0 else ""})
            print(f"  done: {task} {name}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(root / "calibration.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    for task, g in df.groupby("task"):
        print(f"\ntask {task}: scores per setting (ce, brier, ece: lower is better)")
        print(g.drop(columns="task").to_string(index=False))
    if boot:
        bdf = pd.DataFrame(boot)
        bdf.to_csv(root / "calibration_boot.csv", index=False)
        print("\nsetting - raw, same model (* = 95 % interval excludes 0)")
        print(bdf.to_string(index=False))
    print(f"\nwrote {root / 'calibration.csv'}" + (f", {root / 'calibration_boot.csv'}" if boot else ""))


if __name__ == "__main__":
    main()
