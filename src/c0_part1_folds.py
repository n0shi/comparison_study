"""Train the paper's cross-attention model (C0) on our author-grouped folds.

Part-1 code is read-only, so it is imported from its own directory and
patched at runtime, never in its files.

Evaluation protocol (nested; the outer fold is touched exactly once):
  For outer fold k, part 1 trains on the outer training part minus inner fold
  ES_INNER and uses that inner fold for checkpoint selection (lowest inner
  validation loss) and early stopping (part 1's own, also on that loss). Part 1 would otherwise select on fold k itself, which makes its
  out-of-fold scores optimistic. Whenever the inner score improves, the same
  weights predict outer fold k and the test set. The predictions of the
  finally selected epoch are fold k's OOF and fold k's test predictions.

Predicting inside the evaluation step, rather than reloading a checkpoint,
matters: part 1 evaluates with EMA weights but saves the raw weights, so its
best_model.pt is not the model that was selected. Its checkpoints are
deleted after the run unless --keep-checkpoints is given.

Patches:
  cv.StratifiedKFold   -> GivenFolds (our folds; see above)
  cv._split_ambiguous  -> no-op (ties stay; never validated, kept out of
                          training wherever their author is held out)
  cv.make_loaders      -> also builds outer-fold and test loaders
  Trainer._eval_epoch  -> also predicts outer fold and test on improvement
  model.VideoMAEModel  -> videomae_fix (attention biases loaded; part 1 ran
                          with them re-initialised; --paper-videomae undoes it)

Inputs are the regenerated JSONs (config keys train_json_v3 / test_json_v3,
written by vlm_describe.py --merge): the new descriptions, and as transcript
the organisers' cleaned text, as in the paper. Their descriptions are longer, so the VLM text limit is raised from 384
to 512 tokens.

Everything else comes from the part-1 run's own config_resolved.json (the
source of truth for what ran, PART1_MODELS.md); data paths, output location,
seed and modality toggles are overridden. `trio` is the same vlm_quad code
with use_sensorial=false.

Outputs in runs/<dir>/<run_name>/:
  oof_outer.csv           id_EXIST, p_<CLASS>..., fold, best_epoch  (2508 videos)
  test_fold{k}.csv        id_EXIST, p_<CLASS>...                    (674 videos)
  test_mean.csv           mean over the five folds
  c0_manifest.json        what ran, with which code and folds
  (part-1 files; its oof_predictions.csv holds INNER-fold predictions)

Usage:
    python src/c0_part1_folds.py --config config.perun.yaml \\
        --part1-root /mnt/project/perun2601420/vlm_quad/vlm_quad \\
        --part1-config <run>/config_resolved.json \\
        --task 3_1 --seed 42 --modalities quad --runs-dir runs/C0
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import TASKS, load_config, require, setup_stdout

MODALITIES = {"quad": True, "trio": False}  # -> use_sensorial
N_FOLDS = 5
ES_INNER = 1  # inner fold used for checkpoint selection and early stopping


class GivenFolds:
    """Stands in for sklearn's StratifiedKFold inside part-1 cv.py."""

    def __init__(self, folds: pd.DataFrame, smoke: bool = False):
        f = folds.set_index("id_EXIST")
        self.outer = f.outer_fold
        self.inner = {k: f[f"inner_for_outer_{k}"] for k in range(1, N_FOLDS + 1)}
        self.eligible = f.val_eligible.astype(bool)
        self.smoke = smoke  # smoke runs use a random subset of videos
        self.outer_df: dict[int, pd.DataFrame] = {}  # 0-based fold -> outer rows

    def __call__(self, n_splits=None, shuffle=None, random_state=None):
        require(n_splits == N_FOLDS, "part-1 n_folds equals our outer folds",
                expected=N_FOLDS, found=n_splits)
        return self

    def split(self, df, y=None):
        ids = df["exist_id"].astype(str)
        missing = set() if self.smoke else set(self.outer.index) - set(ids)
        extra = set(ids) - set(self.outer.index)
        require(not missing and not extra, "part-1 frame holds exactly the videos in folds.csv",
                expected=len(self.outer), found=f"{len(ids)} rows, missing {sorted(missing)[:5]}, "
                f"extra {sorted(extra)[:5]}")
        outer = ids.map(self.outer).to_numpy()
        ok = ids.map(self.eligible).to_numpy()
        for k in range(1, N_FOLDS + 1):
            inner = ids.map(self.inner[k]).to_numpy()
            rest = outer != k
            train = np.nonzero(rest & (inner != ES_INNER))[0]
            es = np.nonzero(rest & (inner == ES_INNER) & ok)[0]
            out = np.nonzero((outer == k) & ok)[0]
            require(not (set(train) | set(es)) & set(out), f"fold {k}: outer videos unused",
                    expected="disjoint", found="overlap")
            require((inner[rest] >= 0).all(), f"fold {k}: every training video has an inner fold",
                    expected=">= 0", found=sorted(set(inner[rest]))[:5])
            self.outer_df[k - 1] = df.iloc[out].reset_index(drop=True)
            yield train, es


def dry_run(p1raw: dict, overrides: dict, out: Path) -> None:
    """Print the config part 1 will train with, marking what we changed."""
    import dataclasses as dc
    import config as p1config  # part-1 module; imports nothing heavy
    merged = dict(p1raw)
    merged.update(overrides)
    cfg1 = p1config.load_config(None, merged)  # part 1's own validation of keys and types
    keys = ["task", "seed", "train_json", "use_vlm_text", "use_transcript", "use_video",
            "use_sensorial", "max_seq_len_vlm", "max_seq_len_transcript", "n_epochs",
            "freeze_epochs", "patience", "batch_size", "grad_accum", "use_ema", "ema_decay",
            "num_fusion_layers", "dirichlet_alpha", "label_smoothing", "mixup_alpha", "mixup_prob",
            "primary_val_metric", "early_stop_metric", "freeze_text_layers", "freeze_video_layers"]
    vals = dc.asdict(cfg1)
    print(f"run directory: {out} ({'EXISTS: delete it or keep it' if out.exists() else 'free'})")
    print("selection: lowest inner validation loss (this wrapper; primary_val_metric is ignored)")
    for k in keys:
        if k not in vals:
            continue
        mark = "  <- ours" if k in overrides and p1raw.get(k) != vals[k] else ""
        if k in overrides and k in p1raw and p1raw.get(k) != vals[k]:
            mark = f"  <- ours (paper run: {p1raw.get(k)})"
        print(f"  {k:24s} {vals[k]}{mark}")


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
                    help="keep part-1's VideoMAE loading (attention biases re-initialised)")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="keep part-1's model.pt / best_model.pt (raw, non-EMA weights)")
    ap.add_argument("--smoke", action="store_true",
                    help="200 videos, 1 epoch: checks the whole chain in minutes")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the final training config (the part-1 run's + ours) and exit")
    a = ap.parse_args()
    cfg = load_config(a.config)
    classes = TASKS[a.task]["classes"]

    root = Path(a.part1_root).resolve()
    need = ["config.py", "cv.py", "run.py", "data_builder.py", "trainer.py", "model.py",
            "inference.py", "dataset.py", "preprocessing.py"]
    require(all((root / n).is_file() for n in need), f"{root} is the part-1 vlm_quad code",
            expected=need, found=sorted(p.name for p in root.glob("*.py")))
    p1cfg_path = Path(a.part1_config).resolve()
    # json, not yaml: PyYAML reads JSON numbers such as 1e-05 as strings.
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
    out = runs_dir / run_name
    require(a.dry_run or not out.exists(), f"{out} does not exist yet",
            expected="a fresh run directory", found="already there (delete it or keep it)")

    overrides = {
        "task": f"task{a.task}", "seed": a.seed, "cv_mode": "kfold", "n_folds": N_FOLDS,
        "output_dir": str(runs_dir), "run_name": run_name,
        "train_json": str(cfg.path("train_json_v3")), "h5_path": str(cfg.path("train_h5")),
        "soft_paths": {f"task{t}": str(cfg.gold_path(t, "soft")) for t in TASKS},
        "hard_paths": {f"task{t}": str(cfg.gold_path(t, "hard")) for t in TASKS},
        "use_vlm_text": True, "use_transcript": True, "use_video": True,
        "use_sensorial": MODALITIES[a.modalities],
        # The regenerated descriptions are longer (median 348 XLM-R tokens vs 252):
        # 23 % exceed part 1's 384-token limit, 0.4 % exceed 512 (XLM-R's maximum).
        "max_seq_len_vlm": 512,
        # Part 1's patience (4-6 epochs) ends training before its EMA weights are
        # used at all: the EMA is rebuilt when the backbones unfreeze and applied
        # only after 500 updates, then lags ~1000 updates. On 1520 training videos
        # (~190 updates per epoch) that is ~6 epochs of flat or rising validation
        # loss, so task 3.3 stopped at epoch 5-7 with an untrained model.
        "patience": 10,
    }
    if a.smoke:
        overrides.update({"debug_limit": 200, "n_epochs": 1, "freeze_epochs": 0})

    sys.path.insert(0, str(root))
    if a.dry_run:
        dry_run(p1raw, overrides, out)
        return
    import torch                     # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402
    import config as p1config        # noqa: E402  (part-1 modules, imported by path)
    import cv as p1cv                # noqa: E402
    import run as p1run              # noqa: E402
    import model as p1model          # noqa: E402
    import trainer as p1trainer      # noqa: E402
    from dataset import VLMQuadDataset, make_collate_fn  # noqa: E402
    from inference import build_test_df                  # noqa: E402
    from preprocessing import SensorialPreprocessor      # noqa: E402

    # -- config: the run's resolved config, then ours ---------------------------
    orig_load = p1config.load_config

    def load_with_overrides(path=None, cli_overrides=None):
        merged = dict(p1raw)
        merged.update(cli_overrides or {})
        merged.update(overrides)
        return orig_load(None, merged)  # None: never re-read the JSON through YAML
    p1run.load_config = load_with_overrides

    # -- folds -----------------------------------------------------------------
    given = GivenFolds(folds, smoke=a.smoke)
    p1cv.StratifiedKFold = given
    p1cv._split_ambiguous = lambda df, task, log: (df.reset_index(drop=True),
                                                   df.iloc[0:0].reset_index(drop=True))
    if not a.paper_videomae:
        from videomae_fix import FixedVideoMAEModel
        p1model.VideoMAEModel = FixedVideoMAEModel

    # -- extra loaders: outer fold k and test, same preprocessing as fold k ----
    extra: dict[int, dict] = {}
    orig_make_loaders = p1cv.make_loaders

    def make_loaders(train_df, val_df, cfg1, tokenizer, video_processor, fold_dir, log):
        tl, vl, pdim = orig_make_loaders(train_df, val_df, cfg1, tokenizer, video_processor,
                                         fold_dir, log)
        k0 = int(Path(fold_dir).name.split("_")[1]) - 1
        sens = SensorialPreprocessor.load(str(Path(fold_dir) / "preprocessor.json"))
        cfg_test = dataclasses.replace(cfg1, h5_path=str(cfg.path("test_h5")))
        test_df = build_test_df(str(cfg.path("test_json_v3")), cfg_test, log)
        require(len(test_df) == 674, "test frame has all 674 test videos",
                expected=674, found=len(test_df))
        outer_df = given.outer_df[k0]
        collate = make_collate_fn(cfg1)

        def loader(df, c):
            return DataLoader(VLMQuadDataset(df, c, tokenizer, video_processor, sens, mode="eval"),
                              batch_size=cfg1.eval_batch_size, shuffle=False,
                              num_workers=cfg1.num_workers, collate_fn=collate)
        extra[k0] = {"outer": loader(outer_df, cfg1), "test": loader(test_df, cfg_test),
                     "outer_ids": dict(zip(outer_df.video_id.astype(str), outer_df.exist_id)),
                     "test_ids": dict(zip(test_df.video_id.astype(str), test_df.exist_id)),
                     "best": math.inf, "best_epoch": None, "log": []}
        log.info(f"  [C0] fold {k0 + 1}: train={len(train_df)} inner-val={len(val_df)} "
                 f"outer={len(outer_df)} test={len(test_df)}")
        return tl, vl, pdim
    p1cv.make_loaders = make_loaders

    multilabel = TASKS[a.task]["type"] == "multilabel"

    def predict(tr, model, loader, idmap):
        dev = next(model.parameters()).device
        probs, ids = [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = p1trainer._to_device(batch, dev)
                with torch.amp.autocast("cuda", enabled=tr.use_amp, dtype=tr.amp_dtype):
                    logits = p1trainer._model_forward(model, batch).float()
                p = torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=-1)
                probs.append(p.cpu().numpy())
                ids.extend(idmap[str(v)] for v in batch["video_id"])
        df = pd.DataFrame(np.concatenate(probs), columns=[f"p_{c}" for c in classes])
        df.insert(0, "id_EXIST", ids)
        return df

    orig_eval = p1trainer.Trainer._eval_epoch

    def eval_epoch(self, model, loader, criterion, fold, epoch):
        # Runs inside part 1's apply_shadow/restore block: these are the EMA
        # weights that the inner score below belongs to.
        res = orig_eval(self, model, loader, criterion, fold, epoch)
        st = extra[fold]
        # Selection by inner validation loss (lower is better), the paper's stated
        # criterion and the one the graphs use. Part 1 selected by ICM-Soft-Norm,
        # which is 0 on every epoch for task 3.3 (it is truncated at 0), so the
        # first, untrained epoch was kept; it also rewards late overconfidence.
        loss = float(res["loss"])
        loss = math.inf if math.isnan(loss) else loss
        icm = float(res["metrics"].get("val/icm_soft_norm", float("nan")))
        st["log"].append((epoch, loss, icm))
        if loss < st["best"] - float(getattr(self.cfg, "min_delta", 0.0)):
            st["best"], st["best_epoch"] = loss, epoch
            st["oof"] = predict(self, model, st["outer"], st["outer_ids"])
            st["test_pred"] = predict(self, model, st["test"], st["test_ids"])
            self.log.info(f"    [C0] inner val loss={loss:.4f} (icm_soft_norm {icm:.4f}) "
                          f"-> predicted outer fold {fold + 1} and test at epoch {epoch + 1}")
        return res
    p1trainer.Trainer._eval_epoch = eval_epoch

    manifest = {"run_name": run_name, "task": a.task, "seed": a.seed,
                "modalities": a.modalities, "protocol": f"nested: select by lowest val loss on inner fold "
                f"{ES_INNER}, predict outer fold once",
                "videomae_biases": "paper (re-initialised)" if a.paper_videomae else "loaded",
                "part1_root": str(root), "part1_config": str(p1cfg_path),
                "folds_csv_sha256": folds_sha, "comparison_study_commit": git_commit(),
                "overrides": overrides}
    print(json.dumps(manifest, indent=2))

    sys.argv = ["run.py", "--config", str(p1cfg_path)]
    p1run.main()

    # -- collect ---------------------------------------------------------------
    require(sorted(extra) == list(range(N_FOLDS)), "all five folds trained",
            expected=list(range(N_FOLDS)), found=sorted(extra))
    oof, tests = [], []
    for k0 in range(N_FOLDS):
        st = extra[k0]
        require(st["best_epoch"] is not None, f"fold {k0 + 1} selected an epoch",
                expected="an epoch with a finite inner loss", found=st["log"])
        oof.append(st["oof"].assign(fold=k0 + 1, best_epoch=st["best_epoch"] + 1))
        t = st["test_pred"]
        t.to_csv(out / f"test_fold{k0 + 1}.csv", index=False)
        tests.append(t.set_index("id_EXIST"))
        manifest.setdefault("folds", {})[k0 + 1] = {
            "best_epoch": st["best_epoch"] + 1, "inner_best_loss": st["best"],
            "inner_loss": [round(l, 4) for _, l, _ in st["log"]],
            "inner_icm_soft_norm": [round(i, 4) for _, _, i in st["log"]]}
    oof = pd.concat(oof, ignore_index=True)
    if not a.smoke:
        eligible = set(folds.id_EXIST[folds.val_eligible.astype(bool)])
        require(set(oof.id_EXIST) == eligible and oof.id_EXIST.is_unique,
                "OOF covers every validatable video exactly once",
                expected=len(eligible), found=f"{len(oof)} rows, {oof.id_EXIST.nunique()} unique")
    oof.to_csv(out / "oof_outer.csv", index=False)
    idx = tests[0].index
    mean = sum(t.loc[idx] for t in tests) / len(tests)
    mean.reset_index().to_csv(out / "test_mean.csv", index=False)

    if not a.keep_checkpoints:
        for p in out.glob("fold_*/*.pt"):
            p.unlink()
    with open(out / "c0_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"done: {out}  (oof_outer.csv: {len(oof)} videos; test_mean.csv: {len(mean)})")


if __name__ == "__main__":
    main()
