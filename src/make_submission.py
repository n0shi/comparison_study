"""Write test predictions as EXIST 2026 submission files (PyEvALL format).

Format (EXIST 2026 Guidelines V0.5, "Submission Format"): a JSON list of
{"id": id_EXIST, "value": ..., "test_case": "EXIST2025"}, one file per task
and output type, named <task>_<subtask>_<hard|soft>_<team>_<run>, in a folder
exist2026_<team>.

  soft  3.1 / 3.2: the class distribution (sums to 1); 3.3: one probability
        per label (NO + 5 categories, independent, as the gold)
  hard  3.1 / 3.2: the argmax label; 3.3: the categories at or above the
        threshold, or ["NO"]

Usage:
    python src/make_submission.py --config config.yaml --task 3_1 \\
        --pred runs/C0/ensemble/C0_quad_3_1_test.csv --team AegisAthena --run 1 --out submissions/C0_quad
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import TASKS, load_config, load_records, require, setup_stdout
from evaluate import derive_hard

TEST_CASE = "EXIST2025"


def write_runs(pred: pd.DataFrame, task: str, out_dir: Path, team: str, run: int,
               test_ids: list[str], threshold: float = 0.5) -> list[Path]:
    classes = TASKS[task]["classes"]
    cols = [f"p_{c}" for c in classes]
    require(set(cols) <= set(pred.columns), "prediction has every class column",
            expected=cols, found=list(pred.columns))
    pred = pred.drop_duplicates("id_EXIST").set_index("id_EXIST")
    missing = set(test_ids) - set(pred.index)
    require(not missing, "prediction covers every test video", expected=len(test_ids),
            found=f"{len(test_ids) - len(missing)} ({sorted(missing)[:3]})")
    p = pred.loc[test_ids, cols].to_numpy(float)
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), "probabilities in [0, 1]",
            expected="[0,1]", found=f"{p.min()}..{p.max()}")
    if TASKS[task]["type"] != "multilabel":
        p = p / p.sum(1, keepdims=True)
    hard = derive_hard(p, task, threshold)
    d = out_dir / f"exist2026_{team}"
    d.mkdir(parents=True, exist_ok=True)
    t, sub = task.split("_")
    paths = []
    for kind in ("soft", "hard"):
        recs = [{"id": i, "test_case": TEST_CASE,
                 "value": {c: float(v) for c, v in zip(classes, row)} if kind == "soft" else h}
                for i, row, h in zip(test_ids, p, hard)]
        path = d / f"task{t}_{sub}_{kind}_{team}_{run}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False)
        paths.append(path)
    return paths


def test_ids(cfg) -> list[str]:
    return [str(r["id_EXIST"]) for r in load_records(cfg.path("test_json"))]


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--pred", required=True, help="CSV: id_EXIST, p_<CLASS>...")
    ap.add_argument("--team", default="AegisAthena")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5, help="3_3 hard threshold")
    a = ap.parse_args()
    cfg = load_config(a.config)
    pred = pd.read_csv(a.pred, dtype={"id_EXIST": str})
    for p in write_runs(pred, a.task, Path(a.out), a.team, a.run, test_ids(cfg), a.threshold):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
