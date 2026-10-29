"""Step 20, control C4: kNN vote, no network.

A video's prediction is the similarity-weighted mean of its nearest training
videos' soft labels (s07_candidates.py), shrunk towards the training prior:

    p = (sum_i w_i y_i + m * prior) / (sum_i w_i + m),   w_i = max(sim_i, 0)

m > 0 keeps a video with few or weak neighbours near the prior instead of
copying one neighbour (ICM-Soft punishes such confident guesses). For
3.1 / 3.2 p is a distribution; for 3.3 each label is averaged on its own.

Settings are chosen as for C0 (nested): for outer fold k, every setting is
scored by soft cross-entropy on the 4 inner folds of its training part, each
predicted from that inner split's own index; the best mean wins. Then the
outer fold is predicted once from outer split k, and the test set from each
of the 5 outer splits (averaged), as every model's test pass is.

  source  bge_desc | bge_transcript | siglip | all (neighbours of all three,
          weights summed where a video is found by several)
  k       3, 5, 10, 20, 50
  m       0, 0.5, 1, 2, 5, 10, 20

Outputs in runs/C4/C4_<task>/ (the layout C0 uses): oof_outer.csv,
test_fold{k}.csv, test_mean.csv, c4_manifest.json (chosen settings, inner
losses). The OOF scores are printed next to C0's seed ensemble if
results/C0/ensemble/ is there.

Usage:
    python src/c4_knn_vote.py --config config.perun.yaml            # all tasks
    python src/c4_knn_vote.py --config config.perun.yaml --task 3_1
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import evaluate as ev
from common import TASKS, load_config, load_gold, require, setup_stdout

SOURCES = ["bge_desc", "bge_transcript", "siglip", "all"]
KS = [3, 5, 10, 20, 50]
MS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
EPS = 1e-6


class Candidates:
    def __init__(self, path: Path):
        self.h = h5py.File(path, "r")
        self.ids = np.array([i.decode() for i in self.h["ids"][()]])
        self.row = {k: i for i, k in enumerate(self.ids)}
        self.sources = list(self.h.attrs["sources"])
        self._cache = {}

    def lists(self, split: str, source: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(nbr, w) of shape (n, k * n_sources): neighbour rows (-1 = none) and weights."""
        srcs = self.sources if source == "all" else [source]
        nb, w = [], []
        for s in srcs:
            key = (split, s)
            if key not in self._cache:
                g = self.h[split][s]
                self._cache[key] = (g["nbr"][()], g["sim"][()].astype(np.float32))
            n, sim = self._cache[key]
            nb.append(n[:, :k])
            w.append(np.where(n[:, :k] >= 0, np.clip(sim[:, :k], 0, None), 0))
        return np.concatenate(nb, 1), np.concatenate(w, 1)

    def mask(self, split: str) -> np.ndarray:
        return self.h[split]["index_mask"][()].astype(bool)


def vote(Y: np.ndarray, prior: np.ndarray, nbr: np.ndarray, w: np.ndarray, m: float) -> np.ndarray:
    """Y: (n_all, C) soft labels (rows of index videos valid); nbr / w: (q, L)."""
    lab = Y[np.where(nbr >= 0, nbr, 0)]                      # (q, L, C)
    num = (w[..., None] * lab).sum(1) + m * prior
    den = w.sum(1, keepdims=True) + m
    return np.where(den > 0, num / np.where(den > 0, den, 1), prior)


def soft_ce(p: np.ndarray, y: np.ndarray, multilabel: bool) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    if multilabel:
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    return float(-(y * np.log(p)).sum(1).mean())


def run_task(cfg, cand: Candidates, task: str, out_root: Path) -> pd.DataFrame:
    classes = TASKS[task]["classes"]
    ml = TASKS[task]["type"] == "multilabel"
    soft, _ = load_gold(cfg, task)
    Y = np.full((len(cand.ids), len(classes)), np.nan)
    for vid, s in soft.items():
        if vid in cand.row:
            Y[cand.row[vid]] = [s.get(c, 0.0) for c in classes]
    folds = pd.read_csv(cfg.out("folds.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST")
    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str})
    test_rows = np.array([cand.row[i] for i in vids.id_EXIST[vids.split == "test"]])
    elig = folds.val_eligible.astype(bool)

    grid = list(itertools.product(SOURCES, KS, MS))
    oof, tests, manifest = [], [], {"task": task, "grid": {"source": SOURCES, "k": KS, "m": MS}, "folds": {}}
    for k_out in range(1, 6):
        inner_loss = np.zeros(len(grid))
        inner_col = f"inner_for_outer_{k_out}"
        for j in range(1, 5):
            split = f"inner{k_out}.{j}"
            mask = cand.mask(split)
            require(not np.isnan(Y[mask]).any(), f"{split}: every index video has a soft label",
                    expected=0, found=int(np.isnan(Y[mask]).any(1).sum()))
            prior = Y[mask].mean(0)
            val = folds.index[(folds[inner_col] == j) & elig]
            rows = np.array([cand.row[i] for i in val])
            require(not mask[rows].any(), f"{split}: validation videos are not in the index",
                    expected=0, found=int(mask[rows].sum()))
            for g, (src, k, m) in enumerate(grid):
                nbr, w = cand.lists(split, src, k)
                inner_loss[g] += soft_ce(vote(Y, prior, nbr[rows], w[rows], m), Y[rows], ml) / 4
        best = int(inner_loss.argmin())
        src, k, m = grid[best]
        split = f"outer{k_out}"
        mask = cand.mask(split)
        prior = Y[mask].mean(0)
        nbr, w = cand.lists(split, src, k)
        val = folds.index[(folds.outer_fold == k_out) & elig]
        rows = np.array([cand.row[i] for i in val])
        require(not mask[rows].any(), f"{split}: outer-fold videos are not in the index",
                expected=0, found=int(mask[rows].sum()))
        oof.append(pd.DataFrame(vote(Y, prior, nbr[rows], w[rows], m), columns=[f"p_{c}" for c in classes])
                   .assign(id_EXIST=list(val), fold=k_out))
        tests.append(vote(Y, prior, nbr[test_rows], w[test_rows], m))
        top = np.argsort(inner_loss)[:5]
        manifest["folds"][k_out] = {"source": src, "k": k, "m": m, "inner_ce": round(float(inner_loss[best]), 5),
                                    "next_best": [{"source": grid[i][0], "k": grid[i][1], "m": grid[i][2],
                                                   "inner_ce": round(float(inner_loss[i]), 5)} for i in top[1:]]}
        print(f"  {task} outer {k_out}: {src} k={k} m={m}  inner CE {inner_loss[best]:.4f}", flush=True)

    d = out_root / f"C4_{task}"
    d.mkdir(parents=True, exist_ok=True)
    cols = [f"p_{c}" for c in classes]
    o = pd.concat(oof)[["id_EXIST", *cols, "fold"]]
    o.to_csv(d / "oof_outer.csv", index=False)
    test_ids = cand.ids[test_rows]
    for i, t in enumerate(tests, 1):
        pd.DataFrame(t, columns=cols).assign(id_EXIST=test_ids)[["id_EXIST", *cols]] \
            .to_csv(d / f"test_fold{i}.csv", index=False)
    pd.DataFrame(np.mean(tests, 0), columns=cols).assign(id_EXIST=test_ids)[["id_EXIST", *cols]] \
        .to_csv(d / "test_mean.csv", index=False)
    with open(d / "c4_manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
    return o


def report(cfg, task: str, out_root: Path, threshold: float) -> None:
    gold = ev.load_gold_frame(cfg, task)
    subs = ev.subsets(gold)
    rows = []
    paths = {"C4": out_root / f"C4_{task}" / "oof_outer.csv",
             "C0 quad": Path("results/C0/ensemble") / f"C0_quad_{task}_oof.csv"}
    prior = np.tile(gold.soft.mean(0), (len(gold.ids), 1))
    preds = {"prior": (prior, ev.derive_hard(prior, task, threshold))}
    for name, p in paths.items():
        if p.is_file():
            preds[name] = ev.load_pred(p, gold, threshold)
    for name, (p, hard) in preds.items():
        for s in ("all", "contested"):
            sc = ev.Evaluator(gold, p, hard, subs[s]).scores()
            rows.append({"model": name, "subset": s, **{k: round(sc[k], 4) for k in
                                                         ("icm_soft_norm", "icm_norm", "auc", "ce")}})
    print(f"\n  task {task}, out of fold (prior = training class shares, the floor):")
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--task", choices=list(TASKS), default=None, help="default: all three")
    ap.add_argument("--candidates", default=None, help="default: outputs/candidates.h5")
    ap.add_argument("--runs-dir", default="runs/C4")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cand = Candidates(Path(a.candidates) if a.candidates else cfg.out("candidates.h5"))
    thr = {t: json.load(open(Path(__file__).resolve().parents[1] / "configs" / "c0" / f"task{t}.json"))
           .get("multilabel_threshold", 0.5) for t in TASKS}
    for task in ([a.task] if a.task else list(TASKS)):
        run_task(cfg, cand, task, Path(a.runs_dir))
        report(cfg, task, Path(a.runs_dir), thr[task])


if __name__ == "__main__":
    main()
