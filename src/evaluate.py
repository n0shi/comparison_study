"""Step 3: evaluation for every model in the comparison.

Prediction CSV: id_EXIST, p_<CLASS> for every class of the task, optional
`hard` (label, or '|'-joined labels for 3_3). Missing `hard` is derived:
argmax for 3_1/3_2; categories with p >= --threshold for 3_3 (else NO).

Subcommands
  score     metrics overall, per language and per agreement subset
  compare   paired bootstrap of B - A, resampling videos within language
  controls  graph-free controls of BASE matched to CANDIDATE's mean entropy
            (blend toward the fold's training prior; temperature scaling),
            scored and compared with CANDIDATE. Phase 0 showed that graph
            "gains" can be pure shrinkage; a gain counts only if it beats
            these controls, and AUC (invariant to recalibration) moves.
  selftest  fast ICM == PyEvALL on synthetic predictions; gold scores top,
            the class prior scores near the bottom

ICM numbers come from icm_fast (PyEvALL-equivalent, verified by selftest);
`score --pyevall` additionally recomputes the overall row with PyEvALL itself
and fails if they differ.

Usage:
    python src/evaluate.py score    --task 3_1 --pred preds.csv
    python src/evaluate.py compare  --task 3_1 --a C1.csv --b W2.csv --n-boot 10000
    python src/evaluate.py controls --task 3_1 --base C1.csv --candidate B1.csv
    python src/evaluate.py selftest
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

import icm_fast
from common import TASKS, load_config, load_json, require, setup_stdout

EPS = 1e-7
N_BINS = 15


# ====================================================================== data
@dataclass
class Gold:
    task: str
    ids: list[str]
    soft_dicts: list[dict]          # file order, as PyEvALL sees them
    soft: np.ndarray                # (n, C) in TASKS class order
    hard: list                      # label / label list, None if no majority
    lang: np.ndarray
    contested: np.ndarray           # bool
    test_case: str
    author: np.ndarray              # resampling cluster for the bootstrap


def load_gold_frame(cfg, task: str) -> Gold:
    """Gold for the evaluation population: every training video that can be
    validated (val_eligible in folds.csv). The 16 task-3.1 ties are train-only
    in every fold, so no model ever predicts them out of fold; they are
    excluded for all three tasks so that every task scores the same videos."""
    classes = TASKS[task]["classes"]
    folds = pd.read_csv(cfg.out("folds.csv"), dtype={"id_EXIST": str})
    eligible = set(folds.id_EXIST[folds.val_eligible.astype(bool)])
    soft_raw = [r for r in load_json(cfg.gold_path(task, "soft")) if str(r["id"]) in eligible]
    require(len(soft_raw) == len(eligible), f"task{task} gold covers every eligible video",
            expected=len(eligible), found=len(soft_raw))
    hard_map = {str(r["id"]): r["value"] for r in load_json(cfg.gold_path(task, "hard"))}
    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST")
    ids = [str(r["id"]) for r in soft_raw]
    require(set(ids) <= set(vids.index), "gold ids present in videos.csv (run s01 first)",
            expected=len(ids), found=len(set(ids) & set(vids.index)))
    return Gold(
        task=task, ids=ids,
        soft_dicts=[r["value"] for r in soft_raw],
        soft=np.array([[r["value"].get(c, 0.0) for c in classes] for r in soft_raw]),
        hard=[hard_map.get(i) for i in ids],
        lang=vids.loc[ids, "lang"].to_numpy(),
        contested=~vids.loc[ids, f"unanimous_{task}"].astype(bool).to_numpy(),
        test_case=soft_raw[0]["test_case"],
        author=vids.loc[ids, "author"].to_numpy(),
    )


def load_pred(path: Path, gold: Gold, threshold: float) -> tuple[np.ndarray, list]:
    task, classes = gold.task, TASKS[gold.task]["classes"]
    df = pd.read_csv(path, dtype={"id_EXIST": str})
    cols = [f"p_{c}" for c in classes]
    require(set(cols) <= set(df.columns), f"{path.name} has probability columns",
            expected=cols, found=list(df.columns))
    require(df.id_EXIST.is_unique, f"{path.name} ids unique", expected="unique",
            found=int(df.id_EXIST.duplicated().sum()))
    missing = set(gold.ids) - set(df.id_EXIST)
    require(not missing, f"{path.name} covers every evaluated id",
            expected=len(gold.ids), found=f"{len(gold.ids) - len(missing)} ({sorted(missing)[:3]}...)")
    # Extra rows (e.g. ties predicted by part-1 folds for 3.2/3.3) are ignored.
    df = df.set_index("id_EXIST").loc[gold.ids]
    p = df[cols].to_numpy(float)
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(),
            f"{path.name} probabilities finite and in [0, 1]", expected="[0,1]",
            found=f"min {np.nanmin(p)}, max {np.nanmax(p)}")
    if TASKS[task]["type"] != "multilabel":
        s = p.sum(1)
        require(np.allclose(s, 1, atol=1e-3), f"{path.name} rows sum to 1",
                expected=1, found=f"min {s.min():.4f}, max {s.max():.4f}")
        p = p / s[:, None]
    if "hard" in df.columns and df["hard"].notna().all():
        hard = [h.split("|") if task == "3_3" else h for h in df["hard"]]
    else:
        hard = derive_hard(p, task, threshold)
    return p, hard


def derive_hard(p: np.ndarray, task: str, threshold: float) -> list:
    classes = TASKS[task]["classes"]
    if TASKS[task]["type"] != "multilabel":
        return [classes[i] for i in p.argmax(1)]
    out = []
    for row in p:
        cats = [c for c, v in zip(classes[1:], row[1:]) if v >= threshold]
        out.append(cats or ["NO"])
    return out


# =================================================================== metrics
class Evaluator:
    """Precomputes everything that does not depend on resampling weights."""

    def __init__(self, gold: Gold, p: np.ndarray, hard: list, mask: np.ndarray):
        self.task = gold.task
        self.type = TASKS[gold.task]["type"]
        classes = TASKS[gold.task]["classes"]
        hier = TASKS[gold.task]["hierarchy"]
        idx = np.nonzero(mask)[0]
        self.idx = idx
        self.g = gold.soft[idx]
        self.p = p[idx]
        pd_ = [dict(zip(classes, row)) for row in self.p]
        self.soft_problem = icm_fast.compile_soft([gold.soft_dicts[i] for i in idx], pd_, hier)
        hmask = np.array([gold.hard[i] is not None for i in idx])
        self.hidx = np.nonzero(hmask)[0]
        self.hard_problem = icm_fast.compile_hard(
            [gold.hard[i] for i in idx[hmask]], [hard[i] for i in idx[hmask]], hier)

        g, q = self.g, np.clip(self.p, EPS, 1 - EPS)
        if self.type == "multilabel":
            self.ce_row = -(g * np.log(q) + (1 - g) * np.log(1 - q)).mean(1)
            self.brier_row = ((self.p - g) ** 2).mean(1)
        else:
            self.ce_row = -(g * np.log(q)).sum(1)
            self.brier_row = ((self.p - g) ** 2).sum(1)

        # Hard targets for AUC / F1 as one-hot / multi-hot bits.
        ht = [gold.hard[i] for i in idx[hmask]]
        hp = [hard[i] for i in idx[hmask]]
        self.y_true = np.array([[c in _as_set(h) for c in classes] for h in ht], dtype=int)
        self.y_pred = np.array([[c in _as_set(h) for c in classes] for h in hp], dtype=int)
        self.p_hard = self.p[hmask]

    def scores(self, w: np.ndarray | None = None) -> dict:
        n = len(self.idx)
        w = np.ones(n) if w is None else w
        wh = w[self.hidx]
        out = {"n": int(w.sum()), "n_hard": int(wh.sum())}
        out.update(icm_fast.soft_scores(self.soft_problem, w))
        out.update(icm_fast.hard_scores(self.hard_problem, wh))
        out["ce"] = float((w * self.ce_row).sum() / w.sum())
        out["brier"] = float((w * self.brier_row).sum() / w.sum())
        out["ece"] = classwise_ece(self.p, self.g, w)
        out["auc"] = macro_auc(self.y_true, self.p_hard, wh, self.type)
        out["f1_macro"] = macro_f1(self.y_true, self.y_pred, wh, self.type)
        if self.type == "binary":  # official EXIST F1 for 3.1: positive class
            out["f1_yes"] = float(f1_score(self.y_true[:, 1], self.y_pred[:, 1],
                                           sample_weight=wh, zero_division=0))
        out["mean_entropy"] = float((w * entropy(self.p, self.type)).sum() / w.sum())
        return out


def _as_set(h):
    return {h} if isinstance(h, str) else set(h)


def entropy(p: np.ndarray, kind: str) -> np.ndarray:
    q = np.clip(p, EPS, 1 - EPS)
    if kind == "multilabel":
        return -(q * np.log(q) + (1 - q) * np.log(1 - q)).mean(1)
    return -(q * np.log(q)).sum(1)


def classwise_ece(p, g, w) -> float:
    """Soft-target ECE: per class, |mean p - mean gold| over equal-width bins."""
    eces = []
    for c in range(p.shape[1]):
        b = np.minimum((p[:, c] * N_BINS).astype(int), N_BINS - 1)
        tot = np.bincount(b, w, N_BINS)
        gap = np.abs(np.bincount(b, w * p[:, c], N_BINS) - np.bincount(b, w * g[:, c], N_BINS))
        eces.append(gap.sum() / tot.sum())
    return float(np.mean(eces))


def macro_auc(y, p, w, kind) -> float:
    cols = [1] if kind == "binary" else range(y.shape[1])
    vals = []
    for c in cols:
        if 0 < y[:, c].sum() < len(y):
            pos = w[y[:, c] == 1].sum()
            if 0 < pos < w.sum():
                vals.append(roc_auc_score(y[:, c], p[:, c], sample_weight=w))
    return float(np.mean(vals)) if vals else float("nan")


def macro_f1(y, yp, w, kind) -> float:
    if kind == "multilabel":
        return float(f1_score(y, yp, average="macro", sample_weight=w, zero_division=0))
    return float(f1_score(y.argmax(1), yp.argmax(1), average="macro", sample_weight=w,
                          labels=range(y.shape[1]), zero_division=0))


def subsets(gold: Gold) -> dict[str, np.ndarray]:
    n = len(gold.ids)
    return {"all": np.ones(n, bool), "unanimous": ~gold.contested,
            "contested": gold.contested, "en": gold.lang == "en", "es": gold.lang == "es"}


def lang_weights(rng, lang: np.ndarray) -> np.ndarray:
    """Bootstrap counts over videos, resampling within each language."""
    w = np.zeros(len(lang))
    for l in np.unique(lang):
        ix = np.nonzero(lang == l)[0]
        w[ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)))
    return w


class ClusterSampler:
    """Cluster bootstrap: draws authors (all their videos together), within
    each language stratum. Videos of one author share labels far more often
    than chance (85 % hard-label purity on 3.1), so resampling single videos
    would treat them as independent and make the intervals too narrow. An
    author whose videos span both languages is placed in the stratum of the
    majority of its videos."""

    def __init__(self, lang: np.ndarray, author: np.ndarray):
        df = pd.DataFrame({"lang": lang, "author": author})
        major = df.groupby("author").lang.agg(lambda x: x.value_counts().index[0])
        self.strata = []
        codes, uniq = pd.factorize(df.author)
        self.codes = codes
        for l in np.unique(lang):
            members = np.nonzero(np.isin(uniq, major.index[major == l]))[0]
            self.strata.append(members)
        self.n_clusters = len(uniq)

    def __call__(self, rng) -> np.ndarray:
        draws = np.zeros(self.n_clusters)
        for members in self.strata:
            draws[members] = rng.multinomial(len(members), np.full(len(members), 1 / len(members)))
        return draws[self.codes]


# ===================================================================== pyevall
def pyevall_scores(gold: Gold, p: np.ndarray, hard: list, mask: np.ndarray) -> dict:
    from pyevall.evaluation import PyEvALLEvaluation
    from pyevall.metrics.metricfactory import MetricFactory as M
    from pyevall.utils.utils import PyEvALLUtils as U

    classes = TASKS[gold.task]["classes"]
    hier = TASKS[gold.task]["hierarchy"]
    idx = np.nonzero(mask)[0]
    tc = gold.test_case
    gs = [{"test_case": tc, "id": gold.ids[i], "value": gold.soft_dicts[i]} for i in idx]
    ps = [{"test_case": tc, "id": gold.ids[i], "value": dict(zip(classes, map(float, p[i])))}
          for i in idx]
    hi = [i for i in idx if gold.hard[i] is not None]
    gh = [{"test_case": tc, "id": gold.ids[i], "value": gold.hard[i]} for i in hi]
    ph = [{"test_case": tc, "id": gold.ids[i], "value": hard[i]} for i in hi]
    params = {U.PARAM_REPORT: U.PARAM_OPTION_REPORT_EMBEDDED,
              U.PARAM_LOG_LEVEL: U.PARAM_OPTION_LOG_LEVEL_NONE}
    if hier is not None:
        params[U.PARAM_HIERARCHY] = json.loads(json.dumps(hier))
    out = {}
    with tempfile.TemporaryDirectory() as d:
        files = {}
        for name, obj in (("gs", gs), ("ps", ps), ("gh", gh), ("ph", ph)):
            files[name] = str(Path(d) / f"{name}.json")
            with open(files[name], "w", encoding="utf-8") as f:
                json.dump(obj, f)
        for kind, (pf, gf), metrics in (
                ("soft", ("ps", "gs"), [M.ICMSoft.value, M.ICMSoftNorm.value, M.CrossEntropy.value]),
                ("hard", ("ph", "gh"), [M.ICM.value, M.ICMNorm.value])):
            if hier is not None:  # PyEvALL mutates the hierarchy it is given
                params[U.PARAM_HIERARCHY] = json.loads(json.dumps(hier))
            rep = PyEvALLEvaluation().evaluate(files[pf], files[gf], metrics, **params).report
            for m in metrics:
                node = rep.get("metrics", {}).get(m, {})
                val = node.get("results", {}).get("average_per_test_case")
                require(val is not None, f"PyEvALL produced {m} ({kind})",
                        expected="a number", found=json.dumps(node)[:300])
                out[m] = float(val)
    return out


def check_against_pyevall(fast: dict, ref: dict, tol: float = 1e-6) -> None:
    pairs = {"icm_soft": "ICMSoft", "icm_soft_norm": "ICMSoftNorm", "icm": "ICM",
             "icm_norm": "ICMNorm"}
    for k, m in pairs.items():
        require(abs(fast[k] - ref[m]) <= tol, f"fast {k} equals PyEvALL {m}",
                expected=ref[m], found=fast[k])


# ================================================================== commands
def cmd_score(cfg, a) -> dict:
    gold = load_gold_frame(cfg, a.task)
    p, hard = load_pred(Path(a.pred), gold, a.threshold)
    res = {}
    for name, m in subsets(gold).items():
        res[name] = Evaluator(gold, p, hard, m).scores()
    if a.pyevall:
        ref = pyevall_scores(gold, p, hard, subsets(gold)["all"])
        check_against_pyevall(res["all"], ref)
        res["all"]["pyevall_ce"] = ref["CrossEntropy"]
        print("PyEvALL cross-check passed")
    print_table(res, f"{Path(a.pred).name}  task{a.task}")
    _write(a.out, res)
    return res


def bootstrap_diff(gold: Gold, pa, ha, pb, hb, n_boot: int, seed: int, subset: str,
                   resample: str = "author") -> dict:
    m = subsets(gold)[subset]
    ea, eb = Evaluator(gold, pa, ha, m), Evaluator(gold, pb, hb, m)
    base_a, base_b = ea.scores(), eb.scores()
    rng = np.random.default_rng(seed)
    lang = gold.lang[m]
    if resample == "author":
        draw = ClusterSampler(lang, gold.author[m])
    else:
        draw = lambda r: lang_weights(r, lang)  # noqa: E731
    keys = [k for k in base_a if k not in ("n", "n_hard")]
    diffs = {k: [] for k in keys}
    for _ in range(n_boot):
        w = draw(rng)
        sa, sb = ea.scores(w), eb.scores(w)
        for k in keys:
            diffs[k].append(sb[k] - sa[k])
    out = {}
    for k in keys:
        d = np.asarray(diffs[k], float)
        d = d[np.isfinite(d)]
        out[k] = {"a": base_a[k], "b": base_b[k], "diff": base_b[k] - base_a[k],
                  "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                  "p_b_not_better": float((d <= 0).mean())}
    return out


def cmd_compare(cfg, a) -> dict:
    gold = load_gold_frame(cfg, a.task)
    pa, ha = load_pred(Path(a.a), gold, a.threshold)
    pb, hb = load_pred(Path(a.b), gold, a.threshold)
    res = {}
    for sub in a.subsets:
        t = time.time()
        res[sub] = bootstrap_diff(gold, pa, ha, pb, hb, a.n_boot, a.seed, sub, a.resample)
        print(f"\n[{sub}] B - A  ({a.n_boot} resamples, {time.time() - t:.0f}s)")
        for k, v in res[sub].items():
            lo, hi = v["ci95"]
            print(f"  {k:14s} A={v['a']:.4f} B={v['b']:.4f} diff={v['diff']:+.4f} "
                  f"CI95=[{lo:+.4f}, {hi:+.4f}]")
    _write(a.out, res)
    return res


def fold_prior(cfg, gold: Gold, fold_col: str) -> np.ndarray:
    """Per video, the mean soft gold of its outer fold's training part."""
    folds = pd.read_csv(cfg.out("folds.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST")
    require(fold_col in folds.columns, f"folds.csv has column {fold_col}",
            expected=fold_col, found=list(folds.columns))
    f = folds.loc[gold.ids, fold_col].to_numpy()
    prior = np.zeros_like(gold.soft)
    for k in np.unique(f):
        prior[f == k] = gold.soft[f != k].mean(0)
    return prior


def blend(p, prior, lam):
    return (1 - lam) * p + lam * prior


def temper(p, T, kind):
    q = np.clip(p, EPS, 1 - EPS)
    if kind == "multilabel":
        z = np.log(q / (1 - q)) / T
        return 1 / (1 + np.exp(-z))
    z = np.log(q) / T
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def match_entropy(fn, grid, target, kind):
    """Grid point whose transformed predictions' mean entropy is closest to target."""
    errs = [abs(entropy(fn(x), kind).mean() - target) for x in grid]
    i = int(np.argmin(errs))
    return grid[i], errs[i]


def cmd_controls(cfg, a) -> dict:
    gold = load_gold_frame(cfg, a.task)
    kind = TASKS[a.task]["type"]
    pb, _ = load_pred(Path(a.base), gold, a.threshold)
    pc, hc = load_pred(Path(a.candidate), gold, a.threshold)
    target = entropy(pc, kind).mean()
    prior = fold_prior(cfg, gold, a.fold_col)
    out_dir = Path(a.out_dir or cfg.out("controls"))
    out_dir.mkdir(parents=True, exist_ok=True)
    controls = {}
    lam, err = match_entropy(lambda l: blend(pb, prior, l), np.linspace(0, 1, 1001), target, kind)
    if err < 1e-3:
        controls["prior_blend"] = (blend(pb, prior, lam), {"lambda": float(lam)})
    else:
        print(f"prior blend cannot reach candidate entropy (closest err {err:.4f}); skipped")
    T, err = match_entropy(lambda t: temper(pb, t, kind), np.exp(np.linspace(-2, 2.5, 2001)),
                           target, kind)
    controls["temperature"] = (temper(pb, T, kind), {"T": float(T), "entropy_err": float(err)})
    classes = TASKS[a.task]["classes"]
    res = {"target_mean_entropy": float(target)}
    for name, (p, params) in controls.items():
        path = out_dir / f"{Path(a.base).stem}__{name}__matched_to__{Path(a.candidate).stem}.csv"
        df = pd.DataFrame(p, columns=[f"p_{c}" for c in classes])
        df.insert(0, "id_EXIST", gold.ids)
        df.to_csv(path, index=False)
        hp = derive_hard(p, a.task, a.threshold)
        res[name] = {"params": params, "file": str(path),
                     "candidate_vs_control": {
                         sub: bootstrap_diff(gold, p, hp, pc, hc, a.n_boot, a.seed, sub,
                                             a.resample)
                         for sub in a.subsets}}
        print(f"\n== candidate - {name} {params}")
        for sub, r in res[name]["candidate_vs_control"].items():
            for k in ("icm_soft_norm", "ce", "auc"):
                v = r[k]
                print(f"  [{sub}] {k:14s} control={v['a']:.4f} cand={v['b']:.4f} "
                      f"diff={v['diff']:+.4f} CI95=[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    _write(a.out, res)
    return res


def cmd_selftest(cfg, a) -> None:
    rng = np.random.default_rng(0)
    for task in a.tasks:
        gold = load_gold_frame(cfg, task)
        kind, C = TASKS[task]["type"], len(TASKS[task]["classes"])
        n = len(gold.ids)
        prior = np.tile(gold.soft.mean(0), (n, 1))
        if kind == "multilabel":
            noisy = np.clip(gold.soft + rng.normal(0, 0.25, gold.soft.shape), 0.01, 0.99)
            rand = rng.uniform(0.01, 0.99, (n, C))
        else:
            noisy = np.array([rng.dirichlet(1 + 5 * g) for g in gold.soft])
            rand = rng.dirichlet(np.ones(C), n)
        cases = {"gold": gold.soft.copy(), "prior": prior, "noisy_gold": noisy, "random": rand}
        all_m = subsets(gold)["all"]
        scored = {}
        for name, p in cases.items():
            hard = derive_hard(p, task, a.threshold)
            if name == "gold":
                hard = [h if h is not None else derive_hard(p[i:i + 1], task, a.threshold)[0]
                        for i, h in enumerate(gold.hard)]
            fast = Evaluator(gold, p, hard, all_m).scores()
            t = time.time()
            ref = pyevall_scores(gold, p, hard, all_m)
            check_against_pyevall(fast, ref)
            scored[name] = fast
            print(f"task{task} {name:10s} ok  icm_soft_norm={fast['icm_soft_norm']:.4f} "
                  f"icm_norm={fast['icm_norm']:.4f} auc={fast['auc']:.4f} "
                  f"(PyEvALL {time.time() - t:.0f}s)")
        # Sub-sample check: statistics must be re-estimated on the subset.
        m = subsets(gold)["contested"]
        hard = derive_hard(cases["noisy_gold"], task, a.threshold)
        check_against_pyevall(Evaluator(gold, cases["noisy_gold"], hard, m).scores(),
                              pyevall_scores(gold, cases["noisy_gold"], hard, m))
        # Bootstrap weights: integer weights must equal physically duplicated rows.
        w = ClusterSampler(gold.lang, gold.author)(rng)
        dup = np.repeat(np.arange(n), w.astype(int))
        g2 = Gold(task, [f"{gold.ids[i]}_{k}" for k, i in enumerate(dup)],
                  [gold.soft_dicts[i] for i in dup], gold.soft[dup],
                  [gold.hard[i] for i in dup], gold.lang[dup], gold.contested[dup],
                  gold.test_case, gold.author[dup])
        e_w = Evaluator(gold, cases["noisy_gold"], hard, all_m).scores(w)
        e_d = Evaluator(g2, cases["noisy_gold"][dup], [hard[i] for i in dup],
                        np.ones(len(dup), bool)).scores()
        for k in ("icm_soft_norm", "icm_norm", "ce", "auc"):
            require(abs(e_w[k] - e_d[k]) < 1e-9, f"task{task} weighted {k} equals duplicated rows",
                    expected=e_d[k], found=e_w[k])
        require(scored["gold"]["icm_soft_norm"] > 0.99, f"task{task} gold scores ~1",
                expected="> 0.99", found=scored["gold"]["icm_soft_norm"])
        require(scored["prior"]["icm_soft_norm"] < scored["noisy_gold"]["icm_soft_norm"],
                f"task{task} prior below noisy gold", expected="prior < noisy",
                found=(scored["prior"]["icm_soft_norm"], scored["noisy_gold"]["icm_soft_norm"]))
        print(f"task{task}: subset, bootstrap-weight and ordering checks passed")
    print("SELFTEST PASSED")


# ===================================================================== misc
def print_table(res: dict, title: str) -> None:
    keys = ["n", "icm_soft_norm", "icm_norm", "ce", "brier", "ece", "auc", "f1_macro",
            "mean_entropy"]
    print(f"\n{title}")
    print(f"  {'subset':10s}" + "".join(f"{k:>14s}" for k in keys))
    for sub, r in res.items():
        print(f"  {sub:10s}" + "".join(
            f"{r[k]:>14d}" if isinstance(r[k], int) else f"{r[k]:>14.4f}" for k in keys))


def _write(path, obj) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        print(f"wrote {path}")


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config.yaml")
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p, task=True):
        if task:
            p.add_argument("--task", required=True, choices=list(TASKS))
        p.add_argument("--threshold", type=float, default=0.5, help="3_3 hard threshold")
        p.add_argument("--out", help="write results JSON here")

    s = sp.add_parser("score"); common(s)
    s.add_argument("--pred", required=True)
    s.add_argument("--pyevall", action="store_true", help="cross-check overall row with PyEvALL")
    c = sp.add_parser("compare"); common(c)
    c.add_argument("--a", required=True); c.add_argument("--b", required=True)
    for p in (c, k := sp.add_parser("controls")):
        p.add_argument("--n-boot", type=int, default=10000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--subsets", nargs="+", default=["all", "contested"])
        p.add_argument("--resample", choices=["author", "video"], default="author",
                       help="bootstrap unit (author = cluster bootstrap)")
    common(k)
    k.add_argument("--base", required=True); k.add_argument("--candidate", required=True)
    k.add_argument("--fold-col", default="outer_fold")
    k.add_argument("--out-dir")
    t = sp.add_parser("selftest"); common(t, task=False)
    t.add_argument("--tasks", nargs="+", default=list(TASKS))

    a = ap.parse_args()
    cfg = load_config(a.config)
    {"score": cmd_score, "compare": cmd_compare, "controls": cmd_controls,
     "selftest": cmd_selftest}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
