"""Sharpness control (choices.md): is a difference in ICM-Soft-Norm more than a
difference in how confident the predictions are?

ICM-Soft-Norm rises when predictions are only made sharper (temperature < 1),
so two models with the same ranking can differ by more than any graph effect.
For every seed ensemble this script chooses a temperature by cross-fitting:
for the videos of outer fold k, the temperature that maximises ICM-Soft-Norm
on the out-of-fold predictions of the other four folds (the labels of fold k
are not used). Hard decisions are left as they are, so ICM-Norm and F1 do not
change. Every comparison is then read twice, raw and at the chosen sharpness,
next to the proper scores (soft cross-entropy, Brier) and AUC.

Reads results/<setup>/ensemble (graph_report.py) and writes, in the same folder,
sharp_scores.csv (per approach: raw and tempered scores, temperatures) and
sharp_compare.csv (paired author bootstrap, raw and tempered).

Usage:
    python src/sharp_check.py --config config.perun.yaml --results results/graphs_v3         --replicas results/graphs_v3_mean results/graphs_v3_wmean
    python src/sharp_check.py --config config.yaml --results results/graphs_v2 --n-boot 200 --tasks 3_1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate as ev
from common import TASKS, load_config, require, setup_stdout
from graph_report import THRESHOLD, VS

GRID = [0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5]
HYPER = "tuned+e5"
KEYS = ["icm_soft_norm", "ce", "brier", "auc"]          # ce, brier: lower is better


def cross_fit(gold, p, hard, fold, kind):
    """Tempered predictions and the temperature of each outer fold."""
    q, temps = p.copy(), {}
    tempered = {T: ev.temper(p, T, kind) for T in GRID}
    for k in sorted(set(fold)):
        rest = fold != k
        sc = [ev.Evaluator(gold, tempered[T], hard, rest).scores()["icm_soft_norm"] for T in GRID]
        T = GRID[int(np.argmax(sc))]
        temps[int(k)] = T
        q[fold == k] = tempered[T][fold == k]
    return q, temps


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--results", default="results/graphs_v3")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--replicas", nargs="*", default=[],
                    help="result folders whose C1 is a replica of C1 (the noise floor), e.g. results/graphs_v3_mean")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = Path(a.results)
    folds = pd.read_csv(cfg.out("folds.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST")
    scores, comp = [], []
    for task in a.tasks:
        gold = ev.load_gold_frame(cfg, task)
        kind = TASKS[task]["type"]
        fold = folds.outer_fold.loc[gold.ids].to_numpy()
        m = ev.subsets(gold)["all"]
        files = {f.name[:-len(f"_{task}_{HYPER}_oof.csv")]: f
                 for f in sorted((root / "ensemble").glob(f"*_{task}_{HYPER}_oof.csv"))}
        require(files, f"seed ensembles of task {task} in {root / 'ensemble'}", expected="at least C1", found=None)
        for r in a.replicas:                           # C1 under another message function
            f = Path(r) / "ensemble" / f"C1_{task}_{HYPER}_oof.csv"
            require(f.is_file(), f"replica {f}", expected="file", found=None)
            files[f"C1 [{Path(r).name}]"] = f
        c0 = Path("results/C0/ensemble") / f"C0_quad_{task}_oof.csv"
        if c0.is_file():
            files["C0 quad"] = c0
        raw, sharp = {}, {}
        for name, f in files.items():
            t = time.time()
            p, h = ev.load_pred(f, gold, THRESHOLD[task])
            q, temps = cross_fit(gold, p, h, fold, kind)
            raw[name], sharp[name] = (p, h), (q, h)
            sr, ss = ev.Evaluator(gold, p, h, m).scores(), ev.Evaluator(gold, q, h, m).scores()
            edge = [k for k, T in temps.items() if T in (GRID[0], GRID[-1])]
            scores.append({"task": task, "approach": name, "T_folds": [temps[k] for k in sorted(temps)],
                           "T_at_grid_edge": len(edge), "entropy_raw": round(sr["mean_entropy"], 4),
                           "entropy_sharp": round(ss["mean_entropy"], 4),
                           **{f"{k}_raw": round(sr[k], 4) for k in KEYS},
                           **{f"{k}_sharp": round(ss[k], 4) for k in KEYS}})
            print(f"  {task} {name}: T {scores[-1]['T_folds']}, ICM-Soft-Norm {sr['icm_soft_norm']:.4f} -> "
                  f"{ss['icm_soft_norm']:.4f} ({time.time() - t:.0f} s)", flush=True)
        if a.n_boot <= 0:
            continue
        for name in files:
            if name.startswith("C1 ["):                  # replicas: against C1 and the replicas before it
                refs = ["C1", *[r for r in files if r.startswith("C1 [") and r < name]]
            else:
                refs = [r for r in ["C1", *VS.get(name, []), *(["W2"] if name == "W3" else []), "C0 quad"]
                        if r in files and r != name]
            for ref in refs:
                for tag, P in (("raw", raw), ("sharp", sharp)):
                    res = ev.bootstrap_diff(gold, *P[ref], *P[name], a.n_boot, 0, "all")
                    for k in KEYS:
                        if tag == "sharp" and k != "icm_soft_norm":
                            continue                       # the temperature is fitted for ICM-Soft-Norm only
                        lo, hi = res[k]["ci95"]
                        comp.append({"task": task, "approach": name, "vs": ref, "reading": tag, "metric": k,
                                     "diff": round(res[k]["diff"], 4), "lo": round(lo, 4), "hi": round(hi, 4),
                                     "sig": "*" if lo > 0 or hi < 0 else ""})
            print(f"  bootstrap done: {task} {name}", flush=True)
    sdf = pd.DataFrame(scores)
    sdf.to_csv(root / "sharp_scores.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print("\nscores, raw and at the cross-fitted temperature (ce, brier: lower is better)")
    print(sdf.to_string(index=False))
    if comp:
        cdf = pd.DataFrame(comp)
        cdf.to_csv(root / "sharp_compare.csv", index=False)
        cdf["cell"] = cdf["diff"].map("{:+.4f}".format) + cdf.sig
        cdf["col"] = cdf.metric + " " + cdf.reading
        for task, g in cdf.groupby("task"):
            print(f"\ntask {task}: approach - reference (* = 95 % interval excludes 0)")
            piv = g.pivot(index=["approach", "vs"], columns="col", values="cell")
            print(piv[[c for c in ["icm_soft_norm raw", "icm_soft_norm sharp", "ce raw", "brier raw", "auc raw"]
                       if c in piv]].to_string())
    print(f"\nwrote {root / 'sharp_scores.csv'}" + (f", {root / 'sharp_compare.csv'}" if comp else ""))


if __name__ == "__main__":
    main()
