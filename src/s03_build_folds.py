"""Step 2: folds -> outputs/folds.csv + folds_stats.json.

Two outer schemes are built and compared:
  part1   - exact reproduction of part-1's StratifiedKFold (lang + hard label,
            seed 42, task3_1 ties train-only). Verified against the SHA256
            values in config. Part 1 used a different split per task; all
            three are recorded so part-1 OOF predictions can be traced.
  grouped - StratifiedGroupKFold on the same key, grouped by author and
            near-duplicate group, so no author spans two folds. The test set
            is author-disjoint from train, so this is the scheme that matches
            the test condition.
Inner folds (n_inner, grouped the same way) are nested in the chosen scheme.

Usage:
    python src/s03_build_folds.py --config config.yaml --scheme grouped
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from common import TASKS, load_config, load_gold, load_records, require, setup_stdout


def ids_sha256(ids) -> str:
    return hashlib.sha256(",".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()


def part1_frame(cfg, task: str) -> pd.DataFrame:
    """Mirror of part-1 data_builder + labels.load_labels: JSON order, gold hard
    label if present, else argmax of the normalised soft label."""
    recs = load_records(cfg.path("train_json"))
    soft, hard = load_gold(cfg, task)
    classes = TASKS[task]["classes"]
    rows = []
    for r in recs:
        vid = str(r["id_EXIST"])
        vec = np.array([soft[vid].get(c, 0.0) for c in classes], dtype=np.float32)
        if TASKS[task]["type"] != "multilabel":
            vec = vec / vec.sum() if vec.sum() > 1e-6 else vec
            h = hard.get(vid)
            key = classes.index(h) if h in classes else int(vec.argmax())
        else:
            key = classes[int(vec.argmax())]  # part-1 strat_key: dominant soft class
        rows.append({"id_EXIST": vid, "lang": str(r.get("lang")), "key": key,
                     "ambiguous": task == "3_1" and abs(vec[1] - vec[0]) < 1e-6})
    return pd.DataFrame(rows)


def part1_folds(cfg, task: str, seed: int, n: int) -> pd.Series:
    df = part1_frame(cfg, task)
    clear = df[~df.ambiguous].reset_index(drop=True)
    strat = (clear.lang + "_" + clear.key.astype(str)).values
    fold = pd.Series(0, index=df.id_EXIST, dtype=int)  # 0 = train-only
    for k, (_, va) in enumerate(StratifiedKFold(n, shuffle=True, random_state=seed)
                                .split(clear, strat), start=1):
        fold[clear.id_EXIST.iloc[va].tolist()] = k
    return fold


def grouped_folds(df: pd.DataFrame, n: int, seed: int) -> pd.Series:
    """df: id_EXIST, strat, group, train_only. Returns fold 1..n.

    Train-only rows (ties) never validate, but they take their group's fold so
    that the model validated on that fold never trains on a group-mate. A
    train-only row whose group has no validating video gets fold 0 (always
    train)."""
    fold = pd.Series(0, index=df.id_EXIST, dtype=int)
    clear = df[~df.train_only].reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(sgkf.split(clear, clear.strat, clear.group), start=1):
        fold[clear.id_EXIST.iloc[va].tolist()] = k
    group_fold = clear.assign(f=clear.id_EXIST.map(fold)).groupby("group").f.first()
    for _, r in df[df.train_only].iterrows():
        fold[r.id_EXIST] = int(group_fold.get(r.group, 0))
    return fold


def leakage(fold: pd.Series, vids: pd.DataFrame) -> dict:
    """Share of validation videos whose author also has a video in that fold's train part."""
    v = vids.set_index("id_EXIST")
    eligible = ~v.loc[fold.index, "train_only"].to_numpy()
    out, total, leaked = {}, 0, 0
    for k in sorted(set(fold) - {0}):
        val = fold.index[(fold == k).to_numpy() & eligible]
        train_authors = set(v.loc[fold.index[fold != k], "author"])
        n_leak = int(v.loc[val, "author"].isin(train_authors).sum())
        out[k] = round(n_leak / len(val), 4)
        total, leaked = total + len(val), leaked + n_leak
    out["overall"] = round(leaked / total, 4)
    return out


def balance(fold: pd.Series, vids: pd.DataFrame) -> dict:
    v = vids.set_index("id_EXIST").loc[fold.index]
    out = {}
    for k in sorted(set(fold) - {0}):
        m = (fold == k).to_numpy() & ~v.train_only.to_numpy()
        out[k] = {"n": int(m.sum()),
                  "share_es": round(float((v.lang[m] == "es").mean()), 3),
                  "share_yes": round(float((v.hard_3_1[m] == "YES").mean()), 3),
                  "share_contested": round(float((~v.unanimous_3_1[m].astype(bool)).mean()), 3)}
    return out


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--scheme", choices=["part1", "grouped"], required=True,
                    help="outer scheme used for outer_fold / inner_fold columns")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fc = cfg["folds"]
    seed, n_outer, n_inner = fc["seed"], fc["n_outer"], fc["n_inner"]

    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str, "id_Tiktok": str})
    vids = vids[vids.split == "train"].reset_index(drop=True)
    dups = pd.read_csv(cfg.out("dup_groups.csv"), dtype={"id_EXIST": str})
    vids = vids.merge(dups[["id_EXIST", "dup_group"]], on="id_EXIST", how="left")
    require(vids.dup_group.notna().all(), "every train video has a dup_group row",
            expected=len(vids), found=int(vids.dup_group.notna().sum()))

    # Part-1 folds, all three tasks; task3_1 is verified against recorded hashes.
    p1 = {t: part1_folds(cfg, t, seed, n_outer) for t in TASKS}
    expected = {int(k): v for k, v in fc["expected_task3_1_sha256"].items()}
    for k, sha in expected.items():
        got = ids_sha256(p1["3_1"].index[p1["3_1"] == k])
        require(got == sha, f"part-1 task3_1 fold {k} reproduces DATA.md hash",
                expected=sha, found=got)
    print("part-1 task3_1 folds: all 5 SHA256 match DATA.md")

    # Author-grouped scheme. Authors linked through a near-duplicate pair are
    # merged transitively into one group.
    parent = {a: a for a in vids.author}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for g, sub in vids[vids.dup_group >= 0].groupby("dup_group"):
        authors = sub.author.tolist()
        for a in authors[1:]:
            parent[find(a)] = find(authors[0])
    vids["group"] = vids.author.map(find)
    vids["strat"] = vids.lang + "_" + vids.hard_3_1.fillna("TIE")
    vids["train_only"] = vids.tie_3_1.astype(bool)
    grouped = grouped_folds(vids, n_outer, seed)

    outer = grouped if args.scheme == "grouped" else p1["3_1"]
    # Inner folds within each outer training part, grouped the same way.
    for k in range(1, n_outer + 1):
        tr = vids[vids.id_EXIST.map(outer).ne(k)].copy()
        f = grouped_folds(tr, n_inner, seed)
        # inner fold of a video is only meaningful for outer folds other than its own
        inner_k = f.rename(f"inner_for_outer_{k}")
        vids[f"inner_for_outer_{k}"] = vids.id_EXIST.map(inner_k).fillna(-1).astype(int)

    out = vids[["id_EXIST", "lang", "author", "group"]].copy()
    out["outer_fold"] = out.id_EXIST.map(outer).astype(int)
    # Ties never validate; under "grouped" they sit in their author's fold so
    # that fold's model never trains on them.
    out["val_eligible"] = ~vids.train_only.to_numpy()
    for k in range(1, n_outer + 1):
        out[f"inner_for_outer_{k}"] = vids[f"inner_for_outer_{k}"]
    for t in TASKS:
        out[f"part1_fold_{t}"] = out.id_EXIST.map(p1[t]).astype(int)
    out["grouped_fold"] = out.id_EXIST.map(grouped).astype(int)
    out.to_csv(cfg.out("folds.csv"), index=False, encoding="utf-8")

    # Sanity: grouped folds really are author-disjoint, and inner folds nest.
    for k in range(1, n_outer + 1):
        a_val = set(out.group[(out.grouped_fold == k) & out.val_eligible])
        a_tr = set(out.group[(out.grouped_fold != k)])
        require(not (a_val & a_tr), f"grouped fold {k} is group-disjoint from its train part",
                expected=0, found=len(a_val & a_tr))
        own = out[out.outer_fold == k][f"inner_for_outer_{k}"]
        require((own == -1).all(), f"outer fold {k} videos have no inner fold for outer {k}",
                expected="all -1", found=own.value_counts().to_dict())

    same_p1 = {f"{a}_vs_{b}": round(float((p1[a] == p1[b]).mean()), 4)
               for a, b in (("3_1", "3_2"), ("3_1", "3_3"))}
    s = {
        "scheme_used": args.scheme,
        "part1_task3_1_sha256_verified": True,
        "part1_same_fold_share_across_tasks": same_p1,
        "author_leakage_val_share": {"part1_3_1": leakage(p1["3_1"], vids),
                                     "grouped": leakage(grouped, vids)},
        "balance": {"part1_3_1": balance(p1["3_1"], vids), "grouped": balance(grouped, vids)},
        "n_groups": int(vids.group.nunique()),
        "largest_group": int(vids.group.value_counts().max()),
        "train_only_ties": int(vids.train_only.sum()),
    }
    with open(cfg.out("folds_stats.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
