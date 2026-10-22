"""Train the paper's cross-attention model (C0) on our author-grouped folds.

Part-1 code is read-only, so it is imported from its own directory and two
things are swapped at runtime, never in its files:

  * cv.StratifiedKFold -> our folds from outputs/folds.csv (outer_fold);
  * cv._split_ambiguous -> no-op. Ties stay in the frame, never validate, and
    are left out of training for the fold that holds their author (the same
    rule s03_build_folds applies), so no author spans train and validation.

One deliberate change from the paper: VideoMAE is loaded with its attention
biases (videomae_fix.py); part 1 ran with them re-initialised. --paper-videomae
restores the paper's behaviour.

Everything else comes from the part-1 run's own config_resolved.json (the
source of truth for what actually ran, see PART1_MODELS.md), with data paths,
output location, seed and modality toggles overridden. `trio` is the same
vlm_quad code with use_sensorial=false, so trio vs quad differs in exactly one
modality.

Usage (one task, one seed, one modality set):
    python src/c0_part1_folds.py --config config.perun.yaml \\
        --part1-root /mnt/project/perun2601420/vlm_quad \\
        --part1-config /mnt/project/perun2601420/vlm_quad/runs/<run>/config_resolved.json \\
        --task 3_1 --seed 42 --modalities quad --runs-dir runs/C0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import TASKS, load_config, require, setup_stdout

MODALITIES = {"quad": True, "trio": False}  # -> use_sensorial
N_FOLDS = 5


class GivenFolds:
    """Stands in for sklearn's StratifiedKFold inside part-1 cv.py."""

    def __init__(self, folds: pd.DataFrame, smoke: bool = False):
        self.fold = folds.set_index("id_EXIST").outer_fold
        self.eligible = folds.set_index("id_EXIST").val_eligible.astype(bool)
        self.smoke = smoke  # smoke runs use a random subset of videos

    def __call__(self, n_splits=None, shuffle=None, random_state=None):
        require(n_splits == N_FOLDS, "part-1 n_folds equals our outer folds",
                expected=N_FOLDS, found=n_splits)
        return self

    def split(self, df, y=None):
        ids = df["exist_id"].astype(str)
        missing = set() if self.smoke else set(self.fold.index) - set(ids)
        extra = set(ids) - set(self.fold.index)
        require(not missing and not extra, "part-1 frame holds exactly the videos in folds.csv",
                expected=len(self.fold), found=f"{len(ids)} rows, missing {sorted(missing)[:5]}, "
                f"extra {sorted(extra)[:5]}")
        f = ids.map(self.fold).to_numpy()
        ok = ids.map(self.eligible).to_numpy()
        for k in range(1, N_FOLDS + 1):
            val = np.nonzero((f == k) & ok)[0]
            train = np.nonzero(f != k)[0]  # ties of fold k are in neither part
            yield train, val


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=Path(__file__).parent).strip()
    except Exception:
        return "unknown"


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--part1-root", required=True, help="part-1 vlm_quad code directory")
    ap.add_argument("--part1-config", required=True, help="the run's config_resolved.json")
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--modalities", required=True, choices=list(MODALITIES))
    ap.add_argument("--runs-dir", default="runs/C0")
    ap.add_argument("--paper-videomae", action="store_true",
                    help="keep part-1's VideoMAE loading (attention biases re-initialised); "
                         "by default the checkpoint biases are loaded (videomae_fix.py)")
    ap.add_argument("--smoke", action="store_true",
                    help="200 videos, 1 epoch: checks the whole chain in minutes")
    a = ap.parse_args()
    cfg = load_config(a.config)

    root = Path(a.part1_root).resolve()
    need = ["config.py", "cv.py", "run.py", "data_builder.py", "trainer.py", "model.py"]
    require(all((root / n).is_file() for n in need), f"{root} is the part-1 vlm_quad code",
            expected=need, found=sorted(p.name for p in root.glob("*.py")))
    p1cfg_path = Path(a.part1_config).resolve()
    # Parse with json, not yaml: PyYAML (YAML 1.1) reads JSON numbers such as
    # 1e-05 as strings, which then breaks part-1's learning-rate arithmetic.
    require(p1cfg_path.suffix == ".json", "part-1 config is a run's config_resolved.json",
            expected="*.json", found=p1cfg_path.name)
    with open(p1cfg_path, encoding="utf-8") as f:
        p1raw = json.load(f)
    require(p1raw.get("task") == f"task{a.task}", "part-1 config is for this task",
            expected=f"task{a.task}", found=p1raw.get("task"))

    folds = pd.read_csv(cfg.out("folds.csv"), dtype={"id_EXIST": str})
    folds_sha = hashlib.sha256(cfg.out("folds.csv").read_bytes()).hexdigest()
    run_name = f"C0_{a.modalities}_{a.task}_seed{a.seed}" + ("_smoke" if a.smoke else "")
    runs_dir = Path(a.runs_dir).resolve()
    require(not (runs_dir / run_name).exists(), f"{runs_dir / run_name} does not exist yet",
            expected="a fresh run directory", found="already there (delete it or keep it)")

    overrides = {
        "task": f"task{a.task}",
        "seed": a.seed,
        "cv_mode": "kfold",
        "n_folds": N_FOLDS,
        "output_dir": str(runs_dir),
        "run_name": run_name,
        "train_json": str(cfg.path("train_json")),
        "h5_path": str(cfg.path("train_h5")),
        "soft_paths": {f"task{t}": str(cfg.gold_path(t, "soft")) for t in TASKS},
        "hard_paths": {f"task{t}": str(cfg.gold_path(t, "hard")) for t in TASKS},
        "use_vlm_text": True,
        "use_transcript": True,
        "use_video": True,
        "use_sensorial": MODALITIES[a.modalities],
    }
    if a.smoke:
        overrides.update({"debug_limit": 200, "n_epochs": 1, "freeze_epochs": 0})

    sys.path.insert(0, str(root))
    import config as p1config  # noqa: E402  (part-1 modules, imported by path)
    import cv as p1cv          # noqa: E402
    import run as p1run        # noqa: E402
    import model as p1model    # noqa: E402

    orig_load = p1config.load_config

    def load_with_overrides(path=None, cli_overrides=None):
        # The run's resolved config, then part-1 CLI overrides (none), then ours.
        # path=None keeps part-1 from re-reading the JSON through YAML.
        merged = dict(p1raw)
        merged.update(cli_overrides or {})
        merged.update(overrides)
        return orig_load(None, merged)

    p1run.load_config = load_with_overrides
    p1cv.StratifiedKFold = GivenFolds(folds, smoke=a.smoke)
    if not a.paper_videomae:
        from videomae_fix import FixedVideoMAEModel
        p1model.VideoMAEModel = FixedVideoMAEModel
    p1cv._split_ambiguous = lambda df, task, log: (df.reset_index(drop=True),
                                                   df.iloc[0:0].reset_index(drop=True))

    manifest = {"run_name": run_name, "task": a.task, "seed": a.seed,
                "modalities": a.modalities, "part1_root": str(root),
                "videomae_biases": "paper (re-initialised)" if a.paper_videomae else "loaded",
                "part1_config": str(p1cfg_path), "folds_csv_sha256": folds_sha,
                "comparison_study_commit": git_commit(),
                "overrides": {k: v for k, v in overrides.items()}}
    print(json.dumps(manifest, indent=2))

    sys.argv = ["run.py", "--config", str(p1cfg_path)]
    p1run.main()

    out = runs_dir / run_name
    with open(out / "c0_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    oof = out / "oof_predictions.csv"
    require(oof.is_file(), "part-1 wrote oof_predictions.csv", expected=str(oof), found="missing")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
