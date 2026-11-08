"""How much does each input carry on its own? Linear probes on frozen features.

For every source (one text field, the speech, the organisers' text, the VLM
transcript, frames, W3's VLM links ...) and every text encoder (XLM-R mean
pooling, BGE-M3, multilingual-e5), a logistic regression is trained on that
source alone, under C0's nested protocol: for outer fold k the regulariser C
is chosen on inner fold 1 of the training part by soft cross-entropy, then
the outer fold is predicted once. The OOF predictions are scored with
evaluate.py on the same 2508 videos as every model.

Soft labels: each video is repeated once per class with sample weight equal
to that class's annotator share (the plan's Step 27 recipe); 3.3 fits one
binary probe per label the same way.

Sources (n = dimensions):
  <enc>:TARGET | MESSAGE | SITUATION | PEOPLE   one description field
  <enc>:fields      the four fields side by side
  <enc>:speech      mean over the moments with speech, plus a has-speech flag
  <enc>:org         the organisers' cleaned text
  <enc>:all_text    fields + speech + organisers' text
  bge:desc          BGE-M3 on TARGET + MESSAGE (the neighbour-search vector)
  bge:transcript    BGE-M3 on the VLM transcript
  siglip:video      mean SigLIP 2 frame vector
  videomae:clips    mean VideoMAE clip vector (C0's video input, frozen)
  siglip:field_match  each field's best SigLIP similarity to the clips (4)
  e5:all_text_pca<r>  e5:all_text with every text vector reduced to r = 16 / 64 /
                    256 dimensions by one PCA (label-free, fitted on all videos)
  tfidf_char / tfidf_word   TF-IDF of the same texts (fields, speech, organisers'
                    text; char 2-5-grams / words 1-2), SVD to 256 (label-free, all videos)
  videomae:clips_pca64  the mean clip vector after a 64-dimension PCA of the clips
  e5:all_text+videomae / +siglip   text and video side by side (what video adds)
  w3:links          W3's VLM scores: frame and speech score per field and
                    moment, edges and "not shown" per field (outputs/w3/edges.jsonl)

Usage:
    python src/text_probe.py --config config.perun.yaml --jobs 32
    python src/text_probe.py --config config.perun.yaml --only w3:links bge:desc --tasks 3_1
    python src/text_probe.py --config config.perun.yaml --cs 0.01 0.1 1 10 100 --out results/probe_lowdim_c
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import evaluate as ev
from common import TASKS, load_config, load_gold, require, setup_stdout

FIELDS = ["TARGET", "MESSAGE", "SITUATION", "PEOPLE"]
ENCODERS = ["xlmr", "bge_text", "e5", "e5_small", "m2v"]
PCA_RS = [16, 64, 256]
SVD_DIMS = 256
CS = [0.01, 0.1, 1.0]
ES_INNER = 1
THRESHOLD = {t: json.load(open(Path(__file__).resolve().parents[1] / "configs" / "c0" / f"task{t}.json"))
             .get("multilabel_threshold", 0.5) for t in TASKS}


def load_sources(feat: Path, w3: Path, ids: list[str]) -> dict[str, np.ndarray]:
    def get(stage, key):
        with h5py.File(feat / f"{stage}.h5", "r") as h:
            return np.stack([h[i][key][()] for i in ids]).astype(np.float32)

    src = {}
    for enc in ENCODERS:
        if not (feat / f"{enc}.h5").is_file():
            print(f"  {enc}.h5 not found, skipped")
            continue
        f, s, m, o = get(enc, "fields"), get(enc, "speech"), get(enc, "speech_mask"), get(enc, "org")[:, 0]
        name = "bge" if enc == "bge_text" else enc
        for i, fld in enumerate(FIELDS):
            src[f"{name}:{fld}"] = f[:, i]
        src[f"{name}:fields"] = f.reshape(len(ids), -1)
        has = m.sum(1, keepdims=True)
        sp = (s * m[..., None]).sum(1) / np.maximum(has, 1)
        src[f"{name}:speech"] = np.concatenate([sp, (has > 0).astype(np.float32)], 1)
        src[f"{name}:org"] = o
        src[f"{name}:all_text"] = np.concatenate([src[f"{name}:fields"], src[f"{name}:speech"], o], 1)
    src["bge:desc"] = get("bge", "desc")
    src["bge:transcript"] = get("bge", "transcript")
    src["siglip:video"] = get("siglip", "video")
    src["videomae:clips"] = get("videomae", "clips").mean(1)
    sig = np.einsum("nfd,ncd->nfc", get("siglip", "fields"), get("siglip", "clips"))
    src["siglip:field_match"] = sig.max(2)
    if w3.is_file():
        rows = {}
        for line in open(w3, encoding="utf-8"):
            r = json.loads(line)
            e = r.get("edges") or {}
            if not e:
                continue
            v = []
            for fld in FIELDS:
                fe = e[fld]
                fs, ss = fe.get("frame_scores", [0] * 4), fe.get("speech_scores", [0] * 4)
                v += list(fs) + list(ss) + [len(fe["frames"]), len(fe["speech"]), float(not fe["frames"])]
            v += [float(x) for x in r["has_speech"]]
            rows[r["id_EXIST"]] = v
        miss = [i for i in ids if i not in rows]
        require(not miss, "W3 has every video", expected=len(ids), found=f"{len(miss)} missing")
        src["w3:links"] = np.array([rows[i] for i in ids], np.float32)
    return src


def low_dim_sources(cfg, feat: Path, all_ids: list[str], train_ids: list[str]) -> dict[str, np.ndarray]:
    """Encoder check (choices.md): the same inputs in fewer dimensions, and TF-IDF.
    Every reduction is label-free and fitted on all videos (train and test), as
    s06's pca64 stage does for the graphs; rows are returned for train_ids."""
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    def get(stage, key):
        with h5py.File(feat / f"{stage}.h5", "r") as h:
            miss = [i for i in all_ids if i not in h]
            require(not miss, f"{stage}.h5 has every video", expected=len(all_ids), found=f"{len(miss)} missing")
            return np.stack([h[i][key][()] for i in all_ids]).astype(np.float32)
    pos = {k: j for j, k in enumerate(all_ids)}
    rows = np.array([pos[i] for i in train_ids])
    src = {}
    f, sp, o = get("e5", "fields"), get("e5", "speech"), get("e5", "org")[:, 0]
    m = get("e5", "speech_mask").astype(bool)
    pool = np.concatenate([f.reshape(-1, f.shape[-1]), sp[m], o])
    n = len(all_ids)
    for r in PCA_RS:
        p = PCA(r, random_state=0).fit(pool)
        fr = p.transform(f.reshape(-1, f.shape[-1])).reshape(n, -1)
        spr = p.transform(sp.reshape(-1, sp.shape[-1])).reshape(n, 4, r) * m[..., None]
        has = m.sum(1, keepdims=True)
        spm = spr.sum(1) / np.maximum(has, 1)
        src[f"e5:all_text_pca{r}"] = np.concatenate([fr, spm, (has > 0).astype(np.float32), p.transform(o)], 1)[rows]
        print(f"  e5 PCA {r}: {p.explained_variance_ratio_.sum():.1%} of the variance", flush=True)
    clips = get("videomae", "clips")
    pc = PCA(64, random_state=0).fit(clips.reshape(-1, clips.shape[-1]))
    src["videomae:clips_pca64"] = pc.transform(clips.reshape(-1, clips.shape[-1])).reshape(n, 4, 64).mean(1)[rows]
    # TF-IDF of the raw texts the encoders read (s06's inputs)
    from s06_features import Inputs
    inp = Inputs(cfg, 0)
    miss = [i for i in all_ids if i not in inp.videos.index]
    require(not miss, "s06 inputs cover every video", expected=len(all_ids), found=f"{len(miss)} missing")
    texts = []
    for k in all_ids:
        mo = inp.moments[k]
        texts.append(" | ".join(inp.fields(k) + [s for s in mo["speech"] if s.strip()] + [mo["org_text"] or ""]))
    for name, vec in (("tfidf_char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                                       sublinear_tf=True, max_features=200000)),
                      ("tfidf_word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                                     max_features=200000))):
        X = vec.fit_transform(texts)
        svd = TruncatedSVD(SVD_DIMS, random_state=0).fit(X)
        src[name] = svd.transform(X)[rows].astype(np.float32)
        print(f"  {name}: vocabulary {X.shape[1]}, SVD {SVD_DIMS} keeps {svd.explained_variance_ratio_.sum():.1%}",
              flush=True)
    return src


def soft_ce(p, y, multilabel):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if multilabel:
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    return float(-(y * np.log(p)).sum(1).mean())


def fit_predict(X_tr, Y_tr, X_te, C, multilabel):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_tr)
    A, B = sc.transform(X_tr), sc.transform(X_te)
    n, k = Y_tr.shape
    if multilabel:
        out = np.zeros((len(B), k))
        for j in range(k):
            y = Y_tr[:, j]
            Xr = np.concatenate([A, A])
            yr = np.r_[np.ones(n), np.zeros(n)]
            w = np.r_[y, 1 - y]
            keep = w > 0
            if len(np.unique(yr[keep])) < 2:
                out[:, j] = y.mean()
                continue
            m = LogisticRegression(C=C, max_iter=3000).fit(Xr[keep], yr[keep], sample_weight=w[keep])
            out[:, j] = m.predict_proba(B)[:, 1]
        return out
    Xr = np.concatenate([A] * k)
    yr = np.repeat(np.arange(k), n)
    w = Y_tr.T.reshape(-1)
    keep = w > 0
    m = LogisticRegression(C=C, max_iter=3000).fit(Xr[keep], yr[keep], sample_weight=w[keep])
    p = np.zeros((len(B), k))
    p[:, m.classes_] = m.predict_proba(B)
    return p


def probe(name, X, task, Y, folds, train_ids, gold, subs, cs=CS):
    ml = TASKS[task]["type"] == "multilabel"
    row = {i: j for j, i in enumerate(train_ids)}
    oof = np.zeros((len(gold.ids), Y.shape[1]))
    gidx = {i: j for j, i in enumerate(gold.ids)}
    chosen = []
    for k in range(1, 6):
        inner = folds[f"inner_for_outer_{k}"]
        rest = folds.outer_fold != k
        tr = [row[i] for i in train_ids if rest[i] and inner[i] != ES_INNER]
        sel = [row[i] for i in train_ids if rest[i] and inner[i] == ES_INNER and folds.val_eligible[i]]
        out = [i for i in train_ids if folds.outer_fold[i] == k and folds.val_eligible[i]]
        losses = [soft_ce(fit_predict(X[tr], Y[tr], X[sel], C, ml), Y[sel], ml) for C in cs]
        C = cs[int(np.argmin(losses))]
        chosen.append(C)
        p = fit_predict(X[tr], Y[tr], X[[row[i] for i in out]], C, ml)
        oof[[gidx[i] for i in out]] = p
    if not ml:
        oof = oof / oof.sum(1, keepdims=True)
    hard = ev.derive_hard(oof, task, THRESHOLD[task])
    res = []
    for s in ("all", "contested"):
        sc = ev.Evaluator(gold, oof, hard, subs[s]).scores()
        res.append({"source": name, "task": task, "subset": s, "dims": X.shape[1], "C": chosen,
                    **{m: round(sc[m], 4) for m in ("icm_soft_norm", "icm_norm", "auc", "ce")}})
    return res


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--features", default=None, help="default: outputs/features")
    ap.add_argument("--w3", default=None, help="default: outputs/w3/edges.jsonl")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--only", nargs="+", default=None, help="sources to run (default: all)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="results/probe")
    ap.add_argument("--cs", nargs="+", type=float, default=CS,
                    help="grid of the regulariser C (default: the grid of results/probe)")
    a = ap.parse_args()
    from joblib import Parallel, delayed
    cfg = load_config(a.config)
    feat = Path(a.features) if a.features else cfg.path("out_dir") / "features"
    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str}).sort_values("id_EXIST")
    train_ids = vids.id_EXIST[vids.split == "train"].tolist()
    folds = pd.read_csv(cfg.out("folds.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST")
    t0 = time.time()
    src = load_sources(feat, Path(a.w3) if a.w3 else cfg.out("w3/edges.jsonl"), train_ids)
    want_low = a.only is None or any(n.startswith(("e5:all_text_pca", "tfidf", "videomae:clips_pca"))
                                     for n in a.only)
    if want_low:
        src.update(low_dim_sources(cfg, feat, vids.id_EXIST.tolist(), train_ids))
    if "e5:all_text" in src:                 # what video adds to the text
        src["e5:all_text+videomae"] = np.concatenate([src["e5:all_text"], src["videomae:clips"]], 1)
        src["e5:all_text+siglip"] = np.concatenate([src["e5:all_text"], src["siglip:video"]], 1)
    if a.only:
        src = {k: v for k, v in src.items() if k in a.only}
        require(src, "--only names known sources", expected="e.g. bge:desc", found=a.only)
    print(f"{len(src)} sources, {len(train_ids)} training videos, loaded in {time.time() - t0:.0f} s", flush=True)

    jobs = []
    for task in a.tasks:
        classes = TASKS[task]["classes"]
        soft, _ = load_gold(cfg, task)
        Y = np.array([[soft[i].get(c, 0.0) for c in classes] for i in train_ids], np.float64)
        if TASKS[task]["type"] != "multilabel":
            Y = Y / Y.sum(1, keepdims=True)
        gold = ev.load_gold_frame(cfg, task)
        subs = ev.subsets(gold)
        jobs += [delayed(probe)(n, X, task, Y, folds, train_ids, gold, subs, a.cs) for n, X in src.items()]
    rows = [r for res in Parallel(n_jobs=a.jobs, verbose=5)(jobs) for r in res]

    # references: C0 quad and the text-only control, seed ensembles
    for task in a.tasks:
        gold = ev.load_gold_frame(cfg, task)
        subs = ev.subsets(gold)
        for name in ("quad", "text"):
            p = Path("results/C0/ensemble") / f"C0_{name}_{task}_oof.csv"
            if p.is_file():
                pr, hd = ev.load_pred(p, gold, THRESHOLD[task])
                for s in ("all", "contested"):
                    sc = ev.Evaluator(gold, pr, hd, subs[s]).scores()
                    rows.append({"source": f"[C0 {name}]", "task": task, "subset": s, "dims": None, "C": None,
                                 **{m: round(sc[m], 4) for m in ("icm_soft_norm", "icm_norm", "auc", "ce")}})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "text_probe.csv", index=False)
    pd.set_option("display.width", 220)
    for metric in ("auc", "icm_soft_norm"):
        piv = df[df.subset == "all"].pivot_table(index="source", columns="task", values=metric)
        piv = piv.sort_values(piv.columns[0], ascending=False)
        print(f"\n{metric} (all videos), sorted by task {piv.columns[0]}:")
        print(piv.round(4).to_string())
    piv = df[df.subset == "contested"].pivot_table(index="source", columns="task", values="auc")
    print("\nauc on contested videos:")
    print(piv.sort_values(piv.columns[0], ascending=False).round(4).to_string())
    print(f"\nwrote {out / 'text_probe.csv'} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
