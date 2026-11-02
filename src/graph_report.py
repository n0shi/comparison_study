"""Collect graph runs (train.py): status, tuning, scores, comparisons.

Reads runs/graphs/<approach>/ and writes results/graphs/:

  tuning.csv     per approach x task x fold: the winning trial, its inner
                 loss, the default trial's (trial 0) and the spread over trials
  runs.csv       one row per finished run (all 5 folds) x subset
  summary.csv    per approach x task x hyper: mean and sd over seeds, and the
                 seed ensemble (mean of the seeds' OOF probabilities)
  ensemble/      <approach>_<task>_<hyper>_{oof,test}.csv
  compare.csv    paired author bootstrap of seed ensembles: every approach
                 against C1 (do edges help?), C0 quad and C4

Runs on fewer than 5 folds (first-slice `_f1` runs) and smoke runs are listed
but not scored.

Usage:
    python src/graph_report.py --config config.perun.yaml                # everything
    python src/graph_report.py --config config.perun.yaml --n-boot 0     # no bootstrap (fast)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate as ev
from common import TASKS, load_config, setup_stdout

SUBSETS = ["all", "contested"]
METRICS = ["icm_soft_norm", "icm_norm", "auc", "ce"]
THRESHOLD = {t: json.load(open(Path(__file__).resolve().parents[1] / "configs" / "c0" / f"task{t}.json"))
             .get("multilabel_threshold", 0.5) for t in TASKS}


def tuning_table(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.glob("*/tuning_*.json")):
        t = json.load(open(p))
        for k, f in t["folds"].items():
            losses = [r["inner_loss"] for r in f["trials"]]
            rows.append({"approach": t["approach"], "task": t["task"], "text": t.get("text_features", "xlmr"),
                         "fold": int(k),
                         "best_trial": f["best_trial"], "best_loss": f["best_inner_loss"],
                         "default_loss": f["trials"][0]["inner_loss"],
                         "gain_vs_default": round(f["trials"][0]["inner_loss"] - f["best_inner_loss"], 5),
                         "median_loss": round(float(np.median(losses)), 5), "worst_loss": max(losses),
                         **{f"best_{p_}": v for p_, v in f["best"].items()}})
    return pd.DataFrame(rows)


def find_runs(root: Path) -> tuple[list[dict], list[str]]:
    done, other = [], []
    for d in sorted(root.glob("*/*_seed*")):
        man = d / "manifest.json"
        if not man.is_file():
            other.append(f"{d.name} (running or failed)")
            continue
        m = json.load(open(man))
        if len(m["folds"]) != 5 or d.name.endswith("_smoke"):
            other.append(f"{d.name} ({len(m['folds'])} fold(s))")
            continue
        done.append({"dir": d, "approach": m["approach"], "task": m["task"], "seed": m["seed"],
                     "hyper": m.get("hyper_source", "default") + ("" if m.get("text_features", "xlmr") == "xlmr"
                                                                   else f"+{m['text_features']}"),
                     "best_epochs": [f["best_epoch"] for f in m["folds"].values()]})
    return done, other


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--runs", default="runs/graphs")
    ap.add_argument("--out", default="results/graphs")
    ap.add_argument("--n-boot", type=int, default=2000)
    pd.set_option("display.max_columns", 30)
    a = ap.parse_args()
    cfg = load_config(a.config)
    root, out = Path(a.runs), Path(a.out)
    (out / "ensemble").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    # ---- tuning
    tun = tuning_table(root)
    if len(tun):
        tun.to_csv(out / "tuning.csv", index=False)
        print("TUNING (inner fold 1 loss; lower is better)")
        g = tun.groupby(["task", "approach", "text"]).agg(
            folds=("fold", "count"), best=("best_loss", "mean"), default=("default_loss", "mean"),
            gain=("gain_vs_default", "mean"), median_trial=("median_loss", "mean"),
            winners=("best_trial", lambda s: " ".join(map(str, s))))
        print(g.round(4).to_string())
        hp = [c for c in tun.columns if c.startswith("best_") and c not in ("best_trial", "best_loss")]
        print("\n  winning settings (count over approach x task x fold):")
        for c in hp:
            print(f"    {c[5:]:13s} {tun[c].value_counts().to_dict()}")

    # ---- runs
    runs, other = find_runs(root)
    print(f"\nRUNS: {len(runs)} finished (5 folds); not scored: {len(other)}"
          + (f" -> {other[:8]}" if other else ""))
    if not runs:
        return
    golds = {t: ev.load_gold_frame(cfg, t) for t in sorted({r["task"] for r in runs})}
    subs = {t: ev.subsets(g) for t, g in golds.items()}
    rows, groups = [], {}
    for r in runs:
        gold = golds[r["task"]]
        p, hard = ev.load_pred(r["dir"] / "oof_outer.csv", gold, THRESHOLD[r["task"]])
        for s in SUBSETS:
            sc = ev.Evaluator(gold, p, hard, subs[r["task"]][s]).scores()
            rows.append({"approach": r["approach"], "task": r["task"], "hyper": r["hyper"], "seed": r["seed"],
                         "subset": s, **{k: sc[k] for k in METRICS}, "best_epochs": r["best_epochs"]})
        groups.setdefault((r["approach"], r["task"], r["hyper"]), []).append(r)
    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(out / "runs.csv", index=False)

    ens, summ = {}, []
    for (appr, task, hyper), rs in sorted(groups.items()):
        gold = golds[task]
        cols = [f"p_{c}" for c in TASKS[task]["classes"]]
        P = np.mean([ev.load_pred(r["dir"] / "oof_outer.csv", gold, THRESHOLD[task])[0] for r in rs], axis=0)
        pd.DataFrame(P, columns=cols).assign(id_EXIST=gold.ids)[["id_EXIST", *cols]] \
            .to_csv(out / "ensemble" / f"{appr}_{task}_{hyper}_oof.csv", index=False)
        tests = [pd.read_csv(r["dir"] / "test_mean.csv", dtype={"id_EXIST": str}).set_index("id_EXIST")[cols]
                 for r in rs if (r["dir"] / "test_mean.csv").is_file()]
        if tests:
            (sum(tests) / len(tests)).reset_index().to_csv(out / "ensemble" / f"{appr}_{task}_{hyper}_test.csv",
                                                           index=False)
        hard = ev.derive_hard(P, task, THRESHOLD[task])
        ens[(appr, task, hyper)] = (P, hard)
        for s in SUBSETS:
            sc = ev.Evaluator(gold, P, hard, subs[task][s]).scores()
            seeds = runs_df[(runs_df.approach == appr) & (runs_df.task == task) & (runs_df.hyper == hyper)
                            & (runs_df.subset == s)]
            row = {"approach": appr, "task": task, "hyper": hyper, "subset": s, "n_seeds": len(rs)}
            for k in METRICS:
                row[f"{k}_mean"] = round(seeds[k].mean(), 4)
                row[f"{k}_sd"] = round(seeds[k].std(ddof=1), 4) if len(seeds) > 1 else np.nan
                row[f"{k}_ens"] = round(sc[k], 4)
            summ.append(row)

    # references on the same videos: C0 quad seed ensemble and C4
    refs = {}
    for task, gold in golds.items():
        for name, path in (("C0 quad", Path("results/C0/ensemble") / f"C0_quad_{task}_oof.csv"),
                           ("C4", Path("runs/C4") / f"C4_{task}" / "oof_outer.csv")):
            if path.is_file():
                refs[(name, task)] = ev.load_pred(path, gold, THRESHOLD[task])
                for s in SUBSETS:
                    sc = ev.Evaluator(gold, *refs[(name, task)], subs[task][s]).scores()
                    summ.append({"approach": name, "task": task, "hyper": "-", "subset": s, "n_seeds": np.nan,
                                 **{f"{k}_ens": round(sc[k], 4) for k in METRICS}})
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(out / "summary.csv", index=False)
    print("\nOOF scores: seed mean ± sd, and the seed ensemble (ens)")
    for (task, s), g in summ_df.groupby(["task", "subset"], sort=True):
        print(f"\n  task {task} · {s}")
        for _, r in g.iterrows():
            sd = "" if pd.isna(r.get("icm_soft_norm_sd")) else f"±{r['icm_soft_norm_sd']:.4f}"
            mean = "" if pd.isna(r.get("icm_soft_norm_mean")) else f"{r['icm_soft_norm_mean']:.4f}{sd} "
            seeds = "" if pd.isna(r["n_seeds"]) else f"({int(r['n_seeds'])} seeds)"
            print(f"    {r['approach']:8s} {r['hyper']:8s} {seeds:10s} ICM-Soft-Norm {mean}ens {r['icm_soft_norm_ens']:.4f}"
                  f" | ICM-Norm {r['icm_norm_ens']:.4f} | AUC {r['auc_ens']:.4f} | CE {r['ce_ens']:.4f}")

    # ---- bootstrap: every approach against C1 (same hyper source), C0 quad and C4
    if a.n_boot <= 0:
        return
    comp = []
    for (appr, task, hyper), (p, h) in sorted(ens.items()):
        gold = golds[task]
        against = []
        if appr != "C1" and ("C1", task, hyper) in ens:
            against.append((f"C1 {hyper}", ens[("C1", task, hyper)]))
        against += [(name, refs[(name, t)]) for (name, t) in refs if t == task]
        for name, (pb, hb) in against:
            for s in SUBSETS:
                res = ev.bootstrap_diff(gold, pb, hb, p, h, a.n_boot, 0, s)
                for k in ("icm_soft_norm", "icm_norm", "auc"):
                    comp.append({"task": task, "approach": f"{appr} {hyper}", "vs": name, "subset": s,
                                 "metric": k, "diff": round(res[k]["diff"], 4),
                                 "ci95": [round(x, 4) for x in res[k]["ci95"]],
                                 "sig": "*" if res[k]["ci95"][0] > 0 or res[k]["ci95"][1] < 0 else ""})
        print(f"  bootstrap done: {appr} {hyper} task {task}", flush=True)
    cdf = pd.DataFrame(comp)
    cdf.to_csv(out / "compare.csv", index=False)
    print("\napproach - reference (seed ensembles, author bootstrap; * = 95 % CI excludes 0)")
    print(cdf.to_string(index=False))
    print(f"\nwrote {out}/tuning.csv, runs.csv, summary.csv, compare.csv, ensemble/")


if __name__ == "__main__":
    main()
