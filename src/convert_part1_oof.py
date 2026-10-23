"""Convert a part-1 run's out-of-fold predictions to our prediction format.

Input: a part-1 run directory (or its oof_predictions.csv). If the run has no
oof_predictions.csv, the per-fold val_predictions_best.csv files are
concatenated. Output: id_EXIST, p_<CLASS>..., hard, part1_fold.

Part-1 packages disagree on what `video_id` holds (id_EXIST in
vlm_transcript_crossattn, id_Tiktok in every package that reads H5 clips).
The id space is detected from the data, not assumed, and the run's own
gold_soft_* columns are checked against the official gold after the join, so
a wrong join fails here instead of producing a plausible score.

Usage:
    python src/convert_part1_oof.py --config config.yaml --task 3_1 \\
        --run /mnt/project/perun2601420/vlm_quad/runs/<run> --out preds/C0_part1_quad_3_1.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import TASKS, load_config, load_gold, require, setup_stdout


def read_oof(run: Path) -> pd.DataFrame:
    if run.is_file():
        return pd.read_csv(run, dtype={"video_id": str})
    oof = run / "oof_predictions.csv"
    if oof.is_file():
        return pd.read_csv(oof, dtype={"video_id": str})
    parts = sorted(run.rglob("val_predictions_best.csv"))
    require(len(parts) > 0, f"{run} has oof_predictions.csv or fold val_predictions_best.csv",
            expected="one of them", found=sorted(p.name for p in run.iterdir())[:20])
    print(f"concatenating {len(parts)} fold files: {[str(p.relative_to(run)) for p in parts]}")
    return pd.concat([pd.read_csv(p, dtype={"video_id": str}) for p in parts], ignore_index=True)


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--run", required=True, help="part-1 run dir or an OOF csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    classes = TASKS[a.task]["classes"]

    df = read_oof(Path(a.run))
    need = ["video_id"] + [f"pred_soft_{c}" for c in classes] + [f"gold_soft_{c}" for c in classes]
    require(set(need) <= set(df.columns), "OOF has the part-1 prediction columns",
            expected=need, found=list(df.columns))

    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str, "id_Tiktok": str})
    vids = vids[vids.split == "train"]
    in_exist = df.video_id.isin(vids.id_EXIST).mean()
    in_tiktok = df.video_id.isin(vids.id_Tiktok).mean()
    print(f"video_id matches id_EXIST: {in_exist:.1%}, id_Tiktok: {in_tiktok:.1%}")
    require(max(in_exist, in_tiktok) == 1.0, "every OOF video_id is a known train id",
            expected="100 % in one id space", found=f"EXIST {in_exist:.1%} / Tiktok {in_tiktok:.1%}")
    if in_tiktok == 1.0:
        df["id_EXIST"] = df.video_id.map(vids.set_index("id_Tiktok").id_EXIST)
    else:
        df["id_EXIST"] = df.video_id

    if "epoch" in df.columns and df.id_EXIST.duplicated().any():
        raise SystemExit("duplicate ids with an epoch column: pass the best-epoch files, "
                         "not every epoch's predictions")
    require(df.id_EXIST.is_unique, "one OOF prediction per video",
            expected="unique", found=int(df.id_EXIST.duplicated().sum()))

    soft, _ = load_gold(cfg, a.task)
    gold = np.array([[soft[i].get(c, 0.0) for c in classes] for i in df.id_EXIST])
    run_gold = df[[f"gold_soft_{c}" for c in classes]].to_numpy(float)
    if TASKS[a.task]["type"] != "multilabel":
        gold = gold / gold.sum(1, keepdims=True)
    err = np.abs(gold - run_gold).max()
    require(err < 1e-4, "run's gold_soft equals the official gold after the id join",
            expected="max abs diff < 1e-4", found=err)

    out = pd.DataFrame({"id_EXIST": df.id_EXIST})
    for c in classes:
        out[f"p_{c}"] = df[f"pred_soft_{c}"].to_numpy(float)
    if TASKS[a.task]["type"] == "multilabel":
        if "pred_tags" in df.columns:
            tags = df.pred_tags.map(lambda t: json.loads(t) if isinstance(t, str) else t)
            out["hard"] = tags.map(lambda t: "|".join(t) if t else "NO")
    elif "pred_label" in df.columns:
        out["hard"] = df.pred_label.astype(int).map(dict(enumerate(classes)))
    if "fold" in df.columns:
        out["part1_fold"] = df.fold
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(out)} videos")


if __name__ == "__main__":
    main()
