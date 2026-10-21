"""Vectorised ICM / ICM-Norm / ICM-Soft / ICM-Soft-Norm, faithful to PyEvALL 0.1.x.

PyEvALL evaluates one file pair in ~10 s, which rules out bootstrap. Its
recursion, however, touches values only through min/max and class identity:
the information content of a row is a signed sum of per-class terms
ic_c(v), and only ic_c depends on the gold statistics. So each row is expanded
once into (coef, class, value) terms; a bootstrap resample only re-estimates
the per-class statistics with resample weights and re-sums the terms.

Equivalence with PyEvALL is checked by evaluate.py --selftest; do not trust
numbers from this module on a new task or hierarchy before that passes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from statistics import NormalDist

ALPHA1, ALPHA2, BETA = 2.0, 2.0, 3.0


# ---------------------------------------------------------------- hierarchy
def parents(hier, cls):
    """Mirror of PyEvALL get_parents_dict: path from root to cls, or None."""
    if hier is None:
        return None
    if isinstance(hier, dict):
        for k, v in hier.items():
            if k == cls:
                return [k]
            p = parents(v, cls)
            if p:
                return [k] + p
    elif isinstance(hier, list):
        for x in hier:
            if x == cls:
                return [x]
    return None


def class_list(hier, gold_rows) -> list[str]:
    """Class columns PyEvALL builds statistics for."""
    if hier is None:
        out = []
        for row in gold_rows:
            for c in (row if isinstance(row, dict) else _as_list(row)):
                if c not in out:
                    out.append(c)
        return out
    out = []

    def walk(h):
        if isinstance(h, dict):
            for k, v in h.items():
                out.append(k)
                walk(v)
        else:
            out.extend(h)
    walk(hier)
    return out


def _as_list(v):
    return [v] if isinstance(v, str) else list(v)


def complete_hierarchy(hier, gold_rows):
    """PyEvALL adds gold classes missing from the hierarchy as new roots."""
    if hier is None:
        return None
    hier = {k: (list(v) if isinstance(v, list) else v) for k, v in hier.items()}
    known = set(class_list(hier, []))
    for row in gold_rows:
        for c in (row if isinstance(row, dict) else _as_list(row)):
            if c not in known:
                hier[c] = []
                known.add(c)
    return hier


# ---------------------------------------------------------------- soft terms
def _union_soft(a, b):
    out = list(a)
    for cb, vb in b:
        hit = False
        for i, (cu, vu) in enumerate(out):
            if cb == cu:
                out[i] = (cu, max(vb, vu))
                hit = True
        if not hit:
            out.append((cb, vb))
    return out


def _dca_soft(hier, first, rest):
    out = []
    for c in rest:
        pa, pb = parents(hier, first[0]), parents(hier, c[0])
        if pa is None or pb is None:
            continue
        common = [e for e in pa if e in pb]
        if common:
            out = _union_soft(out, [(common[-1], min(first[1], c[1]))])
    return out


def soft_terms(items, hier, coef=1.0, acc=None):
    """IC(items) as a list of (coef, class, value) terms."""
    acc = [] if acc is None else acc
    if not items:
        return acc
    acc.append((coef, items[0][0], items[0][1]))
    soft_terms(items[1:], hier, coef, acc)
    soft_terms(_dca_soft(hier, items[0], items[1:]), hier, -coef, acc)
    return acc


# ---------------------------------------------------------------- hard terms
def _dca_hard(hier, first, rest):
    out = set()
    for c in rest:
        pa, pb = parents(hier, first), parents(hier, c)
        if pa is None or pb is None:
            continue
        common = [e for e in pa if e in pb]
        out |= set(common[-1:])
    return sorted(out)


def hard_terms(items, hier, coef=1.0, acc=None):
    acc = [] if acc is None else acc
    if not items:
        return acc
    acc.append((coef, items[0]))
    hard_terms(items[1:], hier, coef, acc)
    hard_terms(_dca_hard(hier, items[0], items[1:]), hier, -coef, acc)
    return acc


# ---------------------------------------------------------------- compiled
@dataclass
class SoftProblem:
    classes: list[str]
    gold_mat: np.ndarray          # (n, C) gold values per class, hierarchy-propagated
    t_row: np.ndarray             # term arrays for ICM(pred, gold) rows
    t_cls: np.ndarray
    t_val: np.ndarray
    t_coef: np.ndarray
    g_row: np.ndarray             # term arrays for ICM(gold, gold) rows
    g_cls: np.ndarray
    g_val: np.ndarray
    g_coef: np.ndarray
    n: int


def _propagate(gold: dict, hier) -> dict:
    """PyEvALL propagate_max_weigth_ancestors (used for statistics only)."""
    g = dict(gold)

    def walk(h, clas):
        if isinstance(h, dict):
            for c, sub in h.items():
                walk(sub, c)
                if clas is not None:
                    g[clas] = g[c] if clas not in g else max(g[clas], g[c])
        elif isinstance(h, list):
            for c in h:
                g.setdefault(c, 0.0)
                g[clas] = g[c] if clas not in g else max(g[clas], g[c])
    if hier is not None:
        walk(hier, None)
    return g


def compile_soft(gold: list[dict], pred: list[dict], hier) -> SoftProblem:
    """gold/pred: per-row {class: value} dicts in PyEvALL file order."""
    hier = complete_hierarchy(hier, gold)
    classes = class_list(hier, gold)
    extra = {c for p in pred for c in p} - set(classes)
    # PyEvALL gives such classes log2(N); not needed for EXIST, so refuse loudly.
    assert not extra, f"predicted classes absent from gold: {sorted(extra)}"
    idx = {c: i for i, c in enumerate(classes)}
    gm = np.array([[_propagate(g, hier).get(c, 0.0) for c in classes] for g in gold])

    def expand(rows_pairs):
        R, Cl, V, K = [], [], [], []
        for r, (g, p) in enumerate(rows_pairs):
            gs, ps = list(g.items()), list(p.items())
            parts = ((ALPHA1, ps), (ALPHA2, gs), (-BETA, _union_soft(gs, ps)))
            for w, items in parts:
                for coef, c, v in soft_terms(items, hier):
                    R.append(r); Cl.append(idx.get(c, -1)); V.append(v); K.append(w * coef)
        return (np.array(R), np.array(Cl), np.array(V, float), np.array(K, float))

    t = expand(list(zip(gold, pred)))
    gg = expand(list(zip(gold, gold)))
    return SoftProblem(classes, gm, *t, *gg, n=len(gold))


def _soft_ic(cls, val, mu, sd, n_eff):
    """-log2 P(X >= v) under N(mu_c, sd_c); 0 for v == 0; log2(N) when P == 0.

    Calls statistics.NormalDist.cdf itself: PyEvALL takes 1 - cdf, which
    cancels catastrophically in the upper tail, so the last ulp decides
    between a finite IC and the log2(N) cap. Its formula differs between
    Python versions (erf vs erfc), so it is not re-implemented here."""
    out = np.zeros_like(val)
    for c in np.unique(cls):
        sel = (cls == c) & (val != 0.0)
        if not sel.any():
            continue
        cdf = np.frompyfunc(NormalDist(mu=float(mu[c]), sigma=float(sd[c])).cdf, 1, 1)
        prob = 1.0 - cdf(val[sel]).astype(float)
        out[sel] = np.where(prob == 0.0, math.log2(n_eff),
                            -np.log2(np.where(prob == 0.0, 1.0, prob)))
    return out


def soft_scores(P: SoftProblem, w: np.ndarray | None = None) -> dict:
    """ICM-Soft and ICM-Soft-Norm; w = per-row bootstrap counts (None = all ones)."""
    w = np.ones(P.n) if w is None else w.astype(float)
    W = w.sum()
    mu = (w[:, None] * P.gold_mat).sum(0) / W
    sd = (w[:, None] * np.abs(P.gold_mat - mu)).sum(0) / W
    icm_rows = np.bincount(P.t_row, P.t_coef * _soft_ic(P.t_cls, P.t_val, mu, sd, W), P.n)
    gold_rows = np.bincount(P.g_row, P.g_coef * _soft_ic(P.g_cls, P.g_val, mu, sd, W), P.n)
    icm, ref = (w * icm_rows).sum() / W, (w * gold_rows).sum() / W
    return {"icm_soft": icm, "icm_soft_norm": max(0.0, (icm + ref) / (2 * ref))}


@dataclass
class HardProblem:
    classes: list[str]
    member: np.ndarray            # (n, C) 1 if row's gold set contains class or a descendant
    mono: bool
    hier: object
    t_row: np.ndarray
    t_cls: np.ndarray
    t_coef: np.ndarray
    g_row: np.ndarray
    g_cls: np.ndarray
    g_coef: np.ndarray
    n: int


def compile_hard(gold: list, pred: list, hier) -> HardProblem:
    """gold/pred: per-row label (str) or label list."""
    mono = all(isinstance(g, str) for g in gold)
    if not mono and hier is None:
        raise NotImplementedError("PyEvALL does not implement flat multi-label ICM")
    gl = [_as_list(g) for g in gold]
    pl = [_as_list(p) for p in pred]
    hier = complete_hierarchy(hier, gl)
    classes = class_list(hier, gl)
    for p in pl:
        for c in p:
            if c not in classes:
                classes.append(c)
    idx = {c: i for i, c in enumerate(classes)}
    member = np.zeros((len(gl), len(classes)))
    for r, g in enumerate(gl):
        for c in classes:
            if hier is None:
                member[r, idx[c]] = c in g
            else:
                path_hits = [c in (parents(hier, x) or [x]) for x in g]
                member[r, idx[c]] = c in g or any(path_hits)

    def expand(pairs):
        R, Cl, K = [], [], []
        for r, (g, p) in enumerate(pairs):
            for w, items in ((ALPHA1, p), (ALPHA2, g), (-BETA, sorted(set(p) | set(g)))):
                for coef, c in hard_terms(list(items), hier):
                    R.append(r); Cl.append(idx[c]); K.append(w * coef)
        return np.array(R), np.array(Cl), np.array(K, float)

    return HardProblem(classes, member, mono, hier, *expand(list(zip(gl, pl))),
                       *expand(list(zip(gl, gl))), n=len(gl))


def hard_scores(P: HardProblem, w: np.ndarray | None = None) -> dict:
    w = np.ones(P.n) if w is None else w.astype(float)
    W = w.sum()
    freq = (w[:, None] * P.member).sum(0)
    prob = freq / W
    prob = np.where(prob <= 0, 1.0 / W, prob)  # PyEvALL: unseen class -> 1/N
    ic = -np.log2(prob)
    icm_rows = np.bincount(P.t_row, P.t_coef * ic[P.t_cls], P.n)
    gold_rows = np.bincount(P.g_row, P.g_coef * ic[P.g_cls], P.n)
    icm, ref = (w * icm_rows).sum() / W, (w * gold_rows).sum() / W
    return {"icm": icm, "icm_norm": max(0.0, (icm + ref) / (2 * ref))}
