"""Step 1: one row per video (train + test) -> outputs/videos.csv + videos_stats.json.

Labels come from the official gold files as-is. Raw per-annotator arrays are
used only for agreement flags (task 3.3 unanimity), never as targets.

Usage:
    python src/s01_build_videos.py --config config.yaml [--skip-probe]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from common import TASKS, load_config, load_gold, load_records, require, setup_stdout

AUTHOR_RE = re.compile(r"tiktok\.com/@([^/?#]+)/video/(\d+)")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?¡¿])\s+")
VLM_FIELDS = ["SCENE", "SITUATION", "PEOPLE", "MESSAGE", "TARGET"]


def author_from_url(url: str, tiktok_id: str) -> str | None:
    m = AUTHOR_RE.search(url or "")
    if not m:
        return None
    require(m.group(2) == str(tiktok_id), "url video id equals id_Tiktok",
            expected=tiktok_id, found=m.group(2))
    return m.group(1).lower()


def probe_video(path: Path) -> dict:
    """Container-level metadata only; decodes nothing."""
    import av

    out = {"video_exists": path.is_file()}
    if not out["video_exists"]:
        return out
    try:
        with av.open(str(path)) as c:
            out["duration_s"] = float(c.duration / 1e6) if c.duration else np.nan
            v = next((s for s in c.streams if s.type == "video"), None)
            a = next((s for s in c.streams if s.type == "audio"), None)
            if v is not None:
                out["fps"] = float(v.average_rate) if v.average_rate else np.nan
                out["n_frames"] = int(v.frames) if v.frames else np.nan
                out["width"], out["height"] = v.codec_context.width, v.codec_context.height
            out["has_audio"] = a is not None
            if a is not None:
                out["audio_sr"] = a.codec_context.sample_rate
                out["audio_channels"] = a.codec_context.channels
                out["audio_duration_s"] = (
                    float(a.duration * a.time_base) if a.duration else np.nan
                )
    except Exception as e:  # corrupt files are data, not crashes
        out["probe_error"] = f"{type(e).__name__}: {e}"
    return out


def sensorial_summary(sens: dict | None) -> dict:
    out = {}
    if not sens:
        return {"sens_users": 0}
    out["sens_users"] = len(sens.get("users", []))
    for mod in ("ET", "HR", "EEG"):
        by_user = (sens.get("modalities", {}).get(mod) or {}).get("by_user") or {}
        n_present = 0
        for feats in by_user.values():
            vals = [v for v in (feats or {}).values() if v is not None]
            if vals and not all(isinstance(v, float) and np.isnan(v) for v in vals):
                n_present += 1
        out[f"sens_{mod}_users"] = n_present
    return out


def build_rows(cfg, split: str, recs: list[dict], h5_ids: set[str], videos_dir: Path,
               probe: bool) -> list[dict]:
    rows = []
    for i, r in enumerate(recs):
        tid = str(r["id_Tiktok"])
        text = (r.get("text") or "").strip()
        sections = r.get("vlm_sections") or {}
        row = {
            "id_EXIST": str(r["id_EXIST"]),
            "id_Tiktok": tid,
            "split": split,
            "source_order": i,
            "lang": r.get("lang"),
            "author": author_from_url(r.get("url"), tid),
            "n_annotators": r.get("number_annotators"),
            "n_annot_F": sum(g == "F" for g in r.get("gender_annotators") or []),
            "n_annot_M": sum(g == "M" for g in r.get("gender_annotators") or []),
            "annotators": "|".join(r.get("annotators") or []),
            "transcript_chars": len(text),
            "transcript_words": len(text.split()),
            "transcript_has_punct": bool(re.search(r"[.!?]", text)),
            "transcript_n_sentences": len([s for s in SENT_SPLIT_RE.split(text) if s.strip()])
            if text else 0,
            "vlm_model": r.get("vlm_model"),
            "vlm_status": r.get("vlm_status"),
            "vlm_n_frames": r.get("vlm_num_frames"),
            "vlm_last_ts": max(r.get("vlm_frame_timestamps") or [np.nan]),
            "in_h5": tid in h5_ids,
        }
        for f in VLM_FIELDS:
            row[f"vlm_{f}_chars"] = len((sections.get(f) or "").strip())
        row.update(sensorial_summary(r.get("sensorial")))
        if probe:
            row.update(probe_video(videos_dir / f"{tid}.mp4"))
        if split == "train":
            # Raw per-annotator 3.3 sets: used for the unanimity flag only.
            sets = [tuple(sorted(x)) for x in (r.get("labels_task3_3") or [])]
            row["unanimous_3_3"] = len(set(sets)) == 1 if sets else np.nan
        rows.append(row)
    return rows


def add_gold(cfg, df: pd.DataFrame) -> pd.DataFrame:
    train = df["split"] == "train"
    ids = df.loc[train, "id_EXIST"]
    for task, meta in TASKS.items():
        soft, hard = load_gold(cfg, task)
        require(set(soft) == set(ids), f"task{task} soft gold covers exactly the train set",
                expected=len(ids), found=f"{len(soft)} (missing {len(set(ids) - set(soft))},"
                f" extra {len(set(soft) - set(ids))})")
        for c in meta["classes"]:
            df.loc[train, f"soft_{task}_{c}"] = ids.map(lambda i: soft[i].get(c, 0.0))
        h = ids.map(lambda i: hard.get(i))
        df.loc[train, f"hard_{task}"] = h.map(
            lambda v: "|".join(sorted(v)) if isinstance(v, list) else v)
        df.loc[train, f"has_hard_{task}"] = h.notna()
        if meta["type"] != "multilabel":
            p = df.loc[train, [f"soft_{task}_{c}" for c in meta["classes"]]].to_numpy(float)
            df.loc[train, f"unanimous_{task}"] = np.isclose(p.max(1), 1.0)
    df.loc[train, "tie_3_1"] = np.isclose(df.loc[train, "soft_3_1_YES"].astype(float), 0.5)
    return df


def stats(df: pd.DataFrame) -> dict:
    tr, te = df[df.split == "train"], df[df.split == "test"]
    s: dict = {"n": {"train": len(tr), "test": len(te)},
               "lang": {k: v.value_counts().to_dict() for k, v in
                        (("train", tr.lang), ("test", te.lang))}}
    s["annotators_per_video"] = tr.n_annotators.value_counts().sort_index().to_dict()
    s["agreement"] = {}
    for t in TASKS:
        u = tr[f"unanimous_{t}"].astype(bool)
        s["agreement"][t] = {"unanimous": int(u.sum()), "contested": int((~u).sum()),
                             "has_hard": int(tr[f"has_hard_{t}"].astype(bool).sum())}
    s["ties_3_1"] = int(tr.tie_3_1.astype(bool).sum())
    s["p_yes_values"] = tr.soft_3_1_YES.astype(float).round(4).value_counts().sort_index().to_dict()
    s["hard_dist"] = {t: tr[f"hard_{t}"].value_counts().to_dict() for t in ("3_1", "3_2")}
    cat = TASKS["3_3"]["classes"]
    s["hard_3_3_positives"] = {c: int(tr["hard_3_3"].fillna("").str.split("|")
                                      .map(lambda x: c in x).sum()) for c in cat}

    a = df.author
    s["authors"] = {
        "missing": int(a.isna().sum()),
        "unique_train": int(tr.author.nunique()),
        "unique_test": int(te.author.nunique()),
        "train_videos_in_multi_video_authors": int(tr.author.map(tr.author.value_counts()).gt(1).sum()),
        "max_videos_per_author_train": int(tr.author.value_counts().max()),
        "top_authors_train": tr.author.value_counts().head(10).to_dict(),
        "test_videos_with_train_author": int(te.author.isin(set(tr.author)).sum()),
    }
    # Label purity within multi-video authors: how predictable is a label from the author?
    multi = tr[tr.author.map(tr.author.value_counts()).gt(1)]
    if len(multi):
        purity = multi.groupby("author")["hard_3_1"].agg(
            lambda x: x.value_counts(normalize=True).iloc[0] if x.notna().any() else np.nan)
        s["authors"]["mean_hard_3_1_purity_multi_authors"] = round(float(purity.mean()), 4)

    s["transcript"] = {}
    for name, part in (("train", tr), ("test", te)):
        w = part.transcript_words
        s["transcript"][name] = {
            "empty": int((part.transcript_chars == 0).sum()),
            "lt_5_words": int((w < 5).sum()),
            "words_p50": float(w.median()), "words_p95": float(w.quantile(.95)),
            "with_punctuation": int(part.transcript_has_punct.sum()),
            "sentences_p50": float(part.transcript_n_sentences.median()),
        }
    s["vlm"] = {"model": df.groupby(["split", "vlm_model"]).size().unstack(0).fillna(0)
                .astype(int).to_dict(),
                "status_not_ok": int((df.vlm_status != "ok").sum()),
                "empty_fields": {f: int((df[f"vlm_{f}_chars"] == 0).sum()) for f in VLM_FIELDS},
                "n_frames_p50": float(df.vlm_n_frames.median())}
    s["sensorial"] = {m: df[f"sens_{m}_users"].value_counts().sort_index().to_dict()
                      for m in ("ET", "HR", "EEG")}
    s["in_h5"] = df.groupby("split").in_h5.sum().astype(int).to_dict()
    if "video_exists" in df:
        s["video"] = {
            "exists": df.groupby("split").video_exists.sum().astype(int).to_dict(),
            "probe_errors": int(df.get("probe_error", pd.Series(dtype=str)).notna().sum()),
            "no_audio": int((df.has_audio == False).sum()),  # noqa: E712
            "duration_s": df.duration_s.describe(percentiles=[.05, .5, .95]).round(2).to_dict(),
            "audio_sr": df.audio_sr.value_counts().to_dict(),
            "fps": df.fps.round(0).value_counts().head(8).to_dict(),
            "vlm_ts_beyond_duration": int((df.vlm_last_ts > df.duration_s + 0.5).sum()),
        }
    return s


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--skip-probe", action="store_true", help="skip mp4 metadata probing")
    args = ap.parse_args()
    cfg = load_config(args.config)

    rows = []
    for split, jkey, hkey, vkey in (("train", "train_json", "train_h5", "train_videos_dir"),
                                    ("test", "test_json", "test_h5", "test_videos_dir")):
        recs = load_records(cfg.path(jkey))
        with h5py.File(cfg.path(hkey), "r") as f:
            h5_ids = set(f["clips"].keys())
        rows += build_rows(cfg, split, recs, h5_ids, cfg.path(vkey), probe=not args.skip_probe)
        print(f"{split}: {len(recs)} records")

    df = pd.DataFrame(rows)
    require(df.id_EXIST.is_unique, "id_EXIST unique across train+test",
            expected="unique", found=df.id_EXIST[df.id_EXIST.duplicated()].tolist()[:5])
    require(df.id_Tiktok.is_unique, "id_Tiktok unique across train+test",
            expected="unique", found=df.id_Tiktok[df.id_Tiktok.duplicated()].tolist()[:5])
    df = add_gold(cfg, df)

    df.to_csv(cfg.out("videos.csv"), index=False, encoding="utf-8")
    s = stats(df)
    with open(cfg.out("videos_stats.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(s, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
