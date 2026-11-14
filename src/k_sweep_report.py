"""k sweep (choices.md): how the number of neighbours changes a between-video approach.

Reads the seed ensembles of the sweep (graph_report.py on runs/graphs_v3_k:
<approach>_<task>_tuned_<param><k>+e5_oof.csv) and, as references, C1 and the
approach with its tuned k from the main results. For every k it prints
ICM-Soft-Norm raw and at the cross-fitted temperature (sharp_check.py),
cross-entropy, Brier and AUC, and the paired author bootstrap against C1.
Descriptive: the main table keeps the k tuned on inner folds.

Writes k_sweep_<approach>.csv in the sweep's results folder.

Usage:
    python src/k_sweep_report.py --config config.perun.yaml --approach B1 --param nbr_k
    python src/k_sweep_report.py --config config.perun.yaml --approach B3 --param keep_k --n-boot 0
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

import evaluate as ev
from common import TASKS, load_config, require, setup_stdout
from graph_report import THRESHOLD
from sharp_check import HYPER, KEYS, cross_fit


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--approach", default="B1")
    ap.add_argument("--param", default="nbr_k")
    ap.add_argument("--sweep", default="results/graphs_v3_k", help="graph_report.py's output for the sweep runs")
    ap.add_argument("--main", default="results/graphs_v3", help="results with C1 and the tuned approach")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    a = ap.parse_args()
    cfg = load_config(a.config)
    sweep, main_ = Path(a.sweep) / "ensemble", Path(a.main) / "ensemble"
    folds = pd.read_csv(cfg.out("folds.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST")
    rows = []
    for task in a.tasks:
        gold = ev.load_gold_frame(cfg, task)
        kind = TASKS[task]["type"]
        fold = folds.outer_fold.loc[gold.ids].to_numpy()
        m = ev.subsets(gold)["all"]
        pat = re.compile(rf"{a.approach}_{task}_tuned_{a.param}(\d+)\+e5_oof\.csv$")
        ks = sorted((int(pat.match(f.name).group(1)), f) for f in sweep.glob(f"{a.approach}_{task}_*_oof.csv")
                    if pat.match(f.name))
        require(len(ks) > 0, f"sweep ensembles of {a.approach}, task {task}, in {sweep}", expected=">0", found=0)
        files = {"C1": main_ / f"C1_{task}_{HYPER}_oof.csv",
                 f"{a.approach} tuned": main_ / f"{a.approach}_{task}_{HYPER}_oof.csv",
                 **{f"{a.approach} {a.param}={k}": f for k, f in ks}}
        ref = None
        for name, f in files.items():
            require(f.is_file(), f"ensemble {f}", expected="file", found="missing")
            p, h = ev.load_pred(f, gold, THRESHOLD[task])
            q, temps = cross_fit(gold, p, h, fold, kind)
            sr, ss = ev.Evaluator(gold, p, h, m).scores(), ev.Evaluator(gold, q, h, m).scores()
            row = {"task": task, "run": name, "icm_soft_norm_raw": round(sr["icm_soft_norm"], 4),
                   "icm_soft_norm_sharp": round(ss["icm_soft_norm"], 4),
                   **{k: round(sr[k], 4) for k in KEYS if k != "icm_soft_norm"},
                   "T_folds": [temps[k] for k in sorted(temps)]}
            if name == "C1":
                ref = (p, q, h)
            elif a.n_boot > 0:                         # run - C1, raw and at matched sharpness
                raw = ev.bootstrap_diff(gold, ref[0], ref[2], p, h, a.n_boot, 0, "all")
                sharp = ev.bootstrap_diff(gold, ref[1], ref[2], q, h, a.n_boot, 0, "all")
                for tag, res, keys in (("raw", raw, KEYS), ("sharp", sharp, ["icm_soft_norm"])):
                    for k in keys:
                        lo, hi = res[k]["ci95"]
                        row[f"d_{k}_{tag}"] = f"{res[k]['diff']:+.4f}" + ("*" if lo > 0 or hi < 0 else "")
            rows.append(row)
            print(f"  done: {task} {name}", flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.sweep) / f"k_sweep_{a.approach}.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    for task, g in df.groupby("task"):
        print(f"\ntask {task}: scores, then run - C1 (d_*; * = 95 % interval excludes 0; ce, brier: lower is better)")
        print(g.drop(columns=["task", "T_folds"]).fillna("").to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
