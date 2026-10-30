"""Collect every finished C0 run: scores, seed ensembles, quad vs trio, test submissions.

Reads runs/C0/<run>/ (oof_outer.csv, test_mean.csv, c0_manifest.json) and
writes results/C0/:

  runs.csv          one row per run x subset: OOF scores (evaluate.py)
  summary.csv       per modality x task: mean and sd over seeds, and the
                    seed ensemble (mean of the seeds' OOF probabilities)
  <b>_vs_<a>.csv    paired author bootstrap of two modality sets' ensembles, per task:
                    quad_vs_trio (physiology), trio_vs_text and quad_vs_text (video)
  ensemble/         C0_<mod>_<task>_{oof,test}.csv: seed-averaged predictions
  submissions/      the test ensembles as EXIST submission files, one run id
                    per modality (quad = 1, trio = 2, text = 3), to upload to the site

Runs still in progress (no oof_outer.csv yet) are skipped and listed.

Usage:
    python src/c0_report.py --config config.perun.yaml --runs runs/C0 --out results/C0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate as ev
from common import TASKS, load_config, setup_stdout
from make_submission import test_ids, write_runs

RUN_ID = {"quad": 1, "trio": 2, "text": 3}
# hard-label threshold for 3.3: C0 uses its own config's (the paper's run: 0.4)
THRESHOLD = {t: json.load(open(Path(__file__).resolve().parents[1] / "configs" / "c0" / f"task{t}.json"))
             .get("multilabel_threshold", 0.5) for t in TASKS}
SUBSETS = ["all", "contested", "speech", "no_speech", "en", "es"]
METRICS = ["icm_soft_norm", "icm_norm", "auc", "f1_macro", "ce"]


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--runs", default="runs/C0")
    ap.add_argument("--out", default="results/C0")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--team", default="AegisAthena")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="comparisons to bootstrap, e.g. quad-text trio-text (default: every pair present)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out)
    (out / "ensemble").mkdir(parents=True, exist_ok=True)

    runs, pending = [], []
    for d in sorted(Path(a.runs).glob("C0_*")):
        man = d / "c0_manifest.json"
        if not (d / "oof_outer.csv").is_file() or not man.is_file() or d.name.endswith("_smoke"):
            pending.append(d.name)
            continue
        m = json.load(open(man, encoding="utf-8"))
        runs.append({"dir": d, "mod": m["modalities"], "task": m["task"], "seed": m["seed"],
                     "best_epochs": [f["best_epoch"] for f in m.get("folds", {}).values()]})
    print(f"finished runs: {len(runs)}; in progress / missing: {len(pending)} {pending[:6]}")

    rows, ens = [], {}
    golds = {t: ev.load_gold_frame(cfg, t) for t in sorted({r["task"] for r in runs})}
    for r in runs:
        gold = golds[r["task"]]
        p, hard = ev.load_pred(r["dir"] / "oof_outer.csv", gold, THRESHOLD[r["task"]])
        subs = ev.subsets(gold)
        for s in SUBSETS:
            if s in subs:
                sc = ev.Evaluator(gold, p, hard, subs[s]).scores()
                rows.append({"mod": r["mod"], "task": r["task"], "seed": r["seed"], "subset": s,
                             **{k: sc[k] for k in METRICS}, "n": sc["n"],
                             "best_epochs": r["best_epochs"]})
        ens.setdefault((r["mod"], r["task"]), []).append(r)
    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(out / "runs.csv", index=False)

    summ, ens_pred = [], {}
    tids = test_ids(cfg)
    for (mod, task), rs in sorted(ens.items()):
        gold = golds[task]
        P = np.mean([ev.load_pred(r["dir"] / "oof_outer.csv", gold, THRESHOLD[task])[0] for r in rs], axis=0)
        cols = [f"p_{c}" for c in TASKS[task]["classes"]]
        oof = pd.DataFrame(P, columns=cols)
        oof.insert(0, "id_EXIST", gold.ids)
        oof.to_csv(out / "ensemble" / f"C0_{mod}_{task}_oof.csv", index=False)
        tests = [pd.read_csv(r["dir"] / "test_mean.csv", dtype={"id_EXIST": str}).set_index("id_EXIST")
                 .loc[tids, cols] for r in rs]
        test = (sum(tests) / len(tests)).reset_index()
        test.to_csv(out / "ensemble" / f"C0_{mod}_{task}_test.csv", index=False)
        write_runs(test, task, out / "submissions", a.team, RUN_ID[mod], tids, THRESHOLD[task])
        hard = ev.derive_hard(P, task, THRESHOLD[task])
        ens_pred[(mod, task)] = (P, hard)
        subs = ev.subsets(gold)
        for s in SUBSETS:
            if s not in subs:
                continue
            sc = ev.Evaluator(gold, P, hard, subs[s]).scores()
            seeds = runs_df[(runs_df["mod"] == mod) & (runs_df.task == task) & (runs_df.subset == s)]
            row = {"mod": mod, "task": task, "subset": s, "n_seeds": len(rs)}
            for k in METRICS:
                row[f"{k}_mean"] = round(seeds[k].mean(), 4)
                row[f"{k}_sd"] = round(seeds[k].std(ddof=1), 4) if len(seeds) > 1 else np.nan
                row[f"{k}_ensemble"] = round(sc[k], 4)
            summ.append(row)
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(out / "summary.csv", index=False)

    show = summ_df[summ_df.subset.isin(["all", "contested", "speech", "no_speech"])]
    print("\nC0 OOF (nested CV): mean ± sd over seeds, and the seed ensemble")
    for (task, s), g in show.groupby(["task", "subset"], sort=False):
        print(f"\n  task {task} · {s}")
        for _, r in g.iterrows():
            print(f"    {r['mod']:4s} ({r.n_seeds} seeds)  ICM-Soft-Norm {r.icm_soft_norm_mean:.4f}±{r.icm_soft_norm_sd:.4f}"
                  f" ens {r.icm_soft_norm_ensemble:.4f} | ICM-Norm {r.icm_norm_mean:.4f} ens {r.icm_norm_ensemble:.4f}"
                  f" | AUC {r.auc_mean:.4f} | F1 {r.f1_macro_mean:.4f}")

    # (a, b): b - a. quad - trio = physiology; trio - text and quad - text = video (+ physiology)
    pairs = [(x, y) for x, y in (("trio", "quad"), ("text", "trio"), ("text", "quad"))
             if a.pairs is None or f"{y}-{x}" in a.pairs]
    for x, y in pairs:
        comp = []
        for task in sorted(golds):
            if (x, task) in ens_pred and (y, task) in ens_pred:
                (px, hx), (py, hy) = ens_pred[(x, task)], ens_pred[(y, task)]
                for s in ("all", "contested"):
                    res = ev.bootstrap_diff(golds[task], px, hx, py, hy, a.n_boot, 0, s)
                    for k in ("icm_soft_norm", "icm_norm", "auc"):
                        comp.append({"task": task, "subset": s, "metric": k, x: round(res[k]["a"], 4),
                                     y: round(res[k]["b"], 4), f"{y}-{x}": round(res[k]["diff"], 4),
                                     "ci95": [round(v, 4) for v in res[k]["ci95"]]})
        if comp:
            cdf = pd.DataFrame(comp)
            cdf.to_csv(out / f"{y}_vs_{x}.csv", index=False)
            print(f"\n{y} - {x} (seed ensembles, author bootstrap):")
            print(cdf.to_string(index=False), flush=True)
    print(f"\nwrote {out}/runs.csv, summary.csv, ensemble/, submissions/exist2026_{a.team}/")


if __name__ == "__main__":
    main()
