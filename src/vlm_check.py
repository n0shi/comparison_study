"""Two checks of the regenerated VLM descriptions against the paper's.

  signal  Do the new descriptions carry as much information about the labels?
          A plain text model (TF-IDF + logistic regression) on the four fields
          the classifier reads (TARGET, MESSAGE, SITUATION, PEOPLE) is trained
          on our outer folds for each task, once on the paper's descriptions
          and once on the new ones; the OOF predictions are scored with
          evaluate.py and compared with the paired author bootstrap.
  shift   Can a model tell train descriptions from test descriptions? Test
          videos differ in content too (more English, less speech), so the
          reference is the same check on the organisers' text, which one
          pipeline produced for both splits: new descriptions should come
          close to it, the paper's (different models and prompt versions by
          split) should not. 5-fold AUC of TF-IDF + logistic regression.

Nothing here is a model of the thesis: the probe only asks whether the
descriptions lost or gained label information, and whether the source gap
between train and test is gone.

Usage:
    python src/vlm_check.py --config config.yaml \\
        --old ../data/text_llms/training_vlm.json ../data/text_llms/test_vlm.json \\
        --new outputs/vlm/training_vlm_v3.json outputs/vlm/test_vlm_v3.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.sparse import vstack

from common import TASKS, load_config, require, setup_stdout
import evaluate as ev

FIELDS = ["TARGET", "MESSAGE", "SITUATION", "PEOPLE"]  # what the classifier reads (SCENE dropped)


def load_desc(paths: list[str]) -> dict[str, str]:
    out = {}
    for p in paths:
        for k, r in json.load(open(p, encoding="utf-8")).items():
            sec = r.get("vlm_sections") or {}
            out[str(k)] = " ".join(f"[{f}] {sec.get(f, '')}" for f in FIELDS)
    return out


def vectoriser():
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50000,
                           sublinear_tf=True)


def oof_probe(text: dict, gold: ev.Gold, folds: pd.Series) -> np.ndarray:
    """OOF class probabilities: one-vs-rest weighted LR on soft labels, per outer fold."""
    ids = gold.ids
    X_txt = [text[i] for i in ids]
    f = folds.loc[ids].to_numpy()
    C = gold.soft.shape[1]
    kind = TASKS[gold.task]["type"]
    P = np.zeros((len(ids), C))
    for k in np.unique(f):
        tr, va = f != k, f == k
        vec = vectoriser().fit([X_txt[i] for i in np.nonzero(tr)[0]])
        Xtr, Xva = vec.transform([X_txt[i] for i in np.nonzero(tr)[0]]), \
            vec.transform([X_txt[i] for i in np.nonzero(va)[0]])
        for c in range(C):
            y = gold.soft[tr, c]
            # soft label as two weighted copies of each video: positive (w=p), negative (w=1-p)
            lr = LogisticRegression(max_iter=2000, C=4.0)
            lr.fit(vstack([Xtr, Xtr]), np.r_[np.ones(len(y)), np.zeros(len(y))],
                   sample_weight=np.r_[y, 1 - y] + 1e-6)
            P[va, c] = lr.predict_proba(Xva)[:, 1]
    if kind != "multilabel":
        P = P / P.sum(1, keepdims=True)
    return P


def cmd_signal(cfg, a, old: dict, new: dict) -> None:
    folds = pd.read_csv(cfg.out("folds.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST").outer_fold
    rows = []
    for task in a.tasks:
        gold = ev.load_gold_frame(cfg, task)
        require(all(i in old and i in new for i in gold.ids), "both description sets cover the gold ids",
                expected=len(gold.ids), found=sum(i in old and i in new for i in gold.ids))
        p_old, p_new = oof_probe(old, gold, folds), oof_probe(new, gold, folds)
        h_old, h_new = ev.derive_hard(p_old, task, 0.5), ev.derive_hard(p_new, task, 0.5)
        res = ev.bootstrap_diff(gold, p_old, h_old, p_new, h_new, a.n_boot, 0, "all")
        for m in ("icm_soft_norm", "icm_norm", "auc", "ce"):
            r = res[m]
            rows.append({"task": task, "metric": m, "paper": round(r["a"], 4), "new": round(r["b"], 4),
                         "new-paper": round(r["diff"], 4),
                         "ci95": f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"})
    print("\nSIGNAL: TF-IDF + LR probe on the four description fields, OOF on our folds "
          f"(author bootstrap, {a.n_boot} resamples)")
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_shift(cfg, old: dict, new: dict) -> None:
    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str})
    ids, y = vids.id_EXIST.tolist(), (vids.split == "test").to_numpy().astype(int)
    print("\nSHIFT: can a text model tell test descriptions from train ones? (5-fold AUC; 0.5 = no)")
    # reference: the organisers' text came from one pipeline for both splits, so
    # its train/test separability is the part of the shift that is content
    from common import load_records
    from text_clean import clean_text
    org = {str(r["id_EXIST"]): clean_text(r.get("text"))[0] or "empty"
           for k in ("train_json", "test_json") for r in load_records(cfg.path(k))}
    for name, d in (("organisers' text (content reference)", org), ("paper descriptions", old),
                    ("new descriptions", new)):
        X = vectoriser().fit_transform([d[i] for i in ids])
        p = cross_val_predict(LogisticRegression(max_iter=2000, C=4.0), X, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        per_lang = {l: round(roc_auc_score(y[m], p[m]), 3)
                    for l in ("en", "es") for m in [(vids.lang == l).to_numpy()]}
        print(f"  {name:38s} AUC {roc_auc_score(y, p):.3f}   by language {per_lang}")


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--old", nargs=2, required=True, help="paper's train and test JSON")
    ap.add_argument("--new", nargs=2, required=True, help="regenerated train and test JSON")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--only", choices=["signal", "shift"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    old, new = load_desc(a.old), load_desc(a.new)
    if a.only != "shift":
        cmd_signal(cfg, a, old, new)
    if a.only != "signal":
        cmd_shift(cfg, old, new)


if __name__ == "__main__":
    main()
