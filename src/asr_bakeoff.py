"""ASR bake-off on 30 videos: Whisper large-v3 (+WhisperX) vs Parakeet-TDT-0.6B-v3.

Decides which ASR the pipeline uses. Both systems get the same 16 kHz audio
(s04_asr.decode_audio).

  sample    30 train videos, fixed seed: per language 5 likely-music (looping
            transcript or < 1 word/s), 5 dense speech (>= 3.5 words/s, clean
            text), 5 random others -> outputs/asr/bakeoff_ids.txt, bakeoff_sample.csv
  parakeet  run Parakeet on the sample (exist_nemo env, GPU) -> same jsonl
            layout as s04_asr.py (words = [word, start, end, score])
  compare   metrics + a local HTML page for listening (run locally after
            copying both jsonl files back)

Metrics per system:
  words_outside_speech  share of words whose midpoint lies outside Silero VAD
                        speech (+-0.3 s); words invented over music land here
  agree_with_text       share of recognised words present in the organisers'
                        cleaned `text` (a word in neither the audio nor the
                        organisers' text is most likely invented)
  wrong_language        share of stop-words that belong to the other language

Usage:
    python src/asr_bakeoff.py sample   --config config.yaml
    python src/asr_bakeoff.py parakeet --config config.perun.yaml       # exist_nemo
    python src/asr_bakeoff.py compare  --config config.yaml \\
        --whisper outputs/asr/bakeoff_whisper.jsonl --parakeet outputs/asr/bakeoff_parakeet.jsonl
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config, load_records, require, setup_stdout
from text_clean import clean_text

PARAKEET = "nvidia/parakeet-tdt-0.6b-v3"
SEED = 0
STOP = {
    "en": set("the and is are you that this it of to in for with my your what was have not but they".split()),
    "es": set("el la los las que de y es en un una por para con mi tu no se lo pero como muy".split()),
}


def tokens(s: str) -> list[str]:
    return re.findall(r"[\w']+", (s or "").lower())


# ------------------------------------------------------------------ sample
def cmd_sample(cfg, a) -> None:
    v = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str, "id_Tiktok": str})
    v = v[v.split == "train"].copy()
    text = {str(r["id_EXIST"]): r.get("text") for r in load_records(cfg.path("train_json"))}
    rep = v.id_EXIST.map(lambda i: clean_text(text[i])[1])
    v["loops"] = rep.map(lambda r: r["loops"] > 0)
    v["leaks"] = rep.map(lambda r: len(r["leaks"]) > 0)
    v["wps"] = v.transcript_words / v.duration_s
    rng = np.random.default_rng(SEED)
    picks = []
    for lang in ("en", "es"):
        d = v[v.lang == lang]
        music = d[d.loops | (d.wps < 1.0)]
        dense = d[(d.wps >= 3.5) & ~d.loops & ~d.leaks]
        for name, pool in (("music", music), ("dense", dense)):
            idx = rng.choice(len(pool), 5, replace=False)
            picks.append(pool.iloc[idx].assign(stratum=name))
        taken = set(pd.concat(picks).id_EXIST)
        rest = d[~d.id_EXIST.isin(taken)]
        picks.append(rest.iloc[rng.choice(len(rest), 5, replace=False)].assign(stratum="random"))
    s = pd.concat(picks)[["id_EXIST", "id_Tiktok", "lang", "stratum", "duration_s", "wps",
                          "loops", "leaks", "hard_3_1"]]
    require(len(s) == 30 and s.id_EXIST.is_unique, "30 distinct videos", expected=30, found=len(s))
    out = cfg.path("out_dir") / "asr"
    out.mkdir(parents=True, exist_ok=True)
    s.to_csv(out / "bakeoff_sample.csv", index=False)
    (out / "bakeoff_ids.txt").write_text("\n".join(s.id_EXIST) + "\n", encoding="utf-8")
    print(s.groupby(["lang", "stratum"]).size().to_string())
    print(f"wrote {out / 'bakeoff_ids.txt'}")


# ---------------------------------------------------------------- parakeet
def cmd_parakeet(cfg, a) -> None:
    import soundfile as sf
    import tempfile
    from nemo.collections.asr.models import ASRModel
    from s04_asr import SR, decode_audio

    s = pd.read_csv(cfg.path("out_dir") / "asr" / "bakeoff_sample.csv",
                    dtype={"id_EXIST": str, "id_Tiktok": str})
    v = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST")
    dirs = {"train": cfg.path("train_videos_dir"), "test": cfg.path("test_videos_dir")}
    model = ASRModel.from_pretrained(PARAKEET).eval()
    out = Path(a.parakeet or cfg.path("out_dir") / "asr" / "bakeoff_parakeet.jsonl")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp, open(out, "w", encoding="utf-8") as f:
        for r in s.itertuples():
            audio = decode_audio(dirs[v.loc[r.id_EXIST, "split"]] / f"{r.id_Tiktok}.mp4")
            res = {"id_EXIST": r.id_EXIST, "lang": r.lang, "audio_s": round(len(audio) / SR, 3)}
            if len(audio) < SR // 4:
                res.update(segments=[], words=[])
            else:
                wav = Path(tmp) / f"{r.id_EXIST}.wav"
                sf.write(wav, audio, SR)
                hyp = model.transcribe([str(wav)], timestamps=True, verbose=False)[0]
                ts = hyp.timestamp or {}
                res["text"] = hyp.text
                res["words"] = [[w["word"], round(w["start"], 3), round(w["end"], 3), None]
                                for w in ts.get("word", [])]
                res["segments"] = [{"start": round(g["start"], 3), "end": round(g["end"], 3),
                                    "text": g["segment"]} for g in ts.get("segment", [])]
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
    print(f"wrote {out}: {len(s)} videos in {time.time() - t0:.0f} s")


# ----------------------------------------------------------------- compare
def metrics(rec: dict, vad: list, ref_tokens: set, lang: str) -> dict:
    words = [w for w in rec.get("words", []) if w[0]]
    toks = [t for w in words for t in tokens(w[0])]
    timed = [w for w in words if w[1] is not None and w[2] is not None]

    def in_speech(t):
        return any(s - 0.3 <= t <= e + 0.3 for s, e in vad)
    outside = sum(not in_speech((w[1] + w[2]) / 2) for w in timed)
    other = "es" if lang == "en" else "en"
    own_sw = sum(t in STOP[lang] for t in toks)
    oth_sw = sum(t in STOP[other] for t in toks)
    return {
        "n_words": len(words),
        "words_outside_speech": outside / len(timed) if timed else np.nan,
        "agree_with_text": np.mean([t in ref_tokens for t in toks]) if toks else np.nan,
        "wrong_language": oth_sw / (own_sw + oth_sw) if own_sw + oth_sw else np.nan,
        "text": " ".join(w[0] for w in words),
    }


def cmd_compare(cfg, a) -> None:
    s = pd.read_csv(cfg.path("out_dir") / "asr" / "bakeoff_sample.csv",
                    dtype={"id_EXIST": str, "id_Tiktok": str})
    load = lambda p: {d["id_EXIST"]: d for d in map(json.loads, open(p, encoding="utf-8")) if d}
    wh, pk = load(a.whisper), load(a.parakeet)
    require(set(s.id_EXIST) <= set(wh) and set(s.id_EXIST) <= set(pk),
            "both systems cover the sample", expected=30,
            found=f"whisper {len(set(s.id_EXIST) & set(wh))}, parakeet {len(set(s.id_EXIST) & set(pk))}")
    text = {str(r["id_EXIST"]): clean_text(r.get("text"))[0] for r in load_records(cfg.path("train_json"))}
    rows = []
    for r in s.itertuples():
        vad = wh[r.id_EXIST].get("vad", [])
        ref = set(tokens(text[r.id_EXIST]))
        mw = metrics(wh[r.id_EXIST], vad, ref, r.lang)
        mp = metrics(pk[r.id_EXIST], vad, ref, r.lang)
        a_, b_ = set(tokens(mw["text"])), set(tokens(mp["text"]))
        rows.append({"id_EXIST": r.id_EXIST, "lang": r.lang, "stratum": r.stratum,
                     "speech_ratio": wh[r.id_EXIST].get("speech_ratio"),
                     **{f"whisper_{k}": v for k, v in mw.items()},
                     **{f"parakeet_{k}": v for k, v in mp.items()},
                     "disagreement": 1 - len(a_ & b_) / len(a_ | b_) if a_ | b_ else 0.0})
    df = pd.DataFrame(rows)
    out = cfg.path("out_dir") / "asr"
    df.drop(columns=["whisper_text", "parakeet_text"]).to_csv(out / "bakeoff_metrics.csv", index=False)

    cols = ["n_words", "words_outside_speech", "agree_with_text", "wrong_language"]
    summ = {sys: df.groupby(["lang", "stratum"])[[f"{sys}_{c}" for c in cols]].mean().round(3)
            .rename(columns=lambda c: c.split("_", 1)[1]) for sys in ("whisper", "parakeet")}
    print(pd.concat(summ, axis=1).to_string())
    print("\noverall:\n" + pd.DataFrame({sys: df[[f"{sys}_{c}" for c in cols]].mean().values
                                         for sys in ("whisper", "parakeet")}, index=cols).round(3).to_string())

    # listening page: the 10 videos where the systems disagree most, then the rest
    v = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str, "id_Tiktok": str}).set_index("id_EXIST")
    dirs = {"train": cfg.path("train_videos_dir"), "test": cfg.path("test_videos_dir")}
    df = df.sort_values("disagreement", ascending=False).reset_index(drop=True)
    # Browsers block file:// video from a file:// page in different ways, so the
    # clips are copied next to the page and referenced relatively; serve the
    # folder over http (python -m http.server) to listen.
    import shutil
    media = out / "bakeoff_media"
    media.mkdir(exist_ok=True)
    cards = []
    for i, r in df.iterrows():
        name = f"{v.loc[r.id_EXIST, 'id_Tiktok']}.mp4"
        if not (media / name).is_file():
            shutil.copy2(dirs[v.loc[r.id_EXIST, "split"]] / name, media / name)
        src = f"bakeoff_media/{name}"
        tag = "listen" if i < 10 else "optional"
        cards.append(f"""
<section class="card {tag}"><h2>{r.id_EXIST} · {r.lang} · {r.stratum} · speech {r.speech_ratio}
 · disagreement {r.disagreement:.2f}{' · <b>LISTEN</b>' if i < 10 else ''}</h2>
<video controls preload="none" src="{src}"></video>
<div class="cols">
 <div><h3>Whisper ({r.whisper_n_words} w, {r.whisper_words_outside_speech:.0%} outside speech)</h3><p>{html.escape(r.whisper_text)}</p></div>
 <div><h3>Parakeet ({r.parakeet_n_words} w, {r.parakeet_words_outside_speech:.0%} outside speech)</h3><p>{html.escape(r.parakeet_text)}</p></div>
 <div><h3>Organisers' text (cleaned)</h3><p>{html.escape(text[r.id_EXIST])}</p></div>
</div>
<fieldset data-id="{r.id_EXIST}">Closer to what is said:
 <label><input type="radio" name="v{r.id_EXIST}" value="whisper">Whisper</label>
 <label><input type="radio" name="v{r.id_EXIST}" value="parakeet">Parakeet</label>
 <label><input type="radio" name="v{r.id_EXIST}" value="same">about the same</label>
 <label><input type="radio" name="v{r.id_EXIST}" value="both_bad">both bad</label>
 <label><input type="radio" name="v{r.id_EXIST}" value="no_speech">no speech</label></fieldset>
</section>""")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>ASR bake-off</title>
<style>
:root{{--bg:#fff;--fg:#1b1b1b;--card:#f4f4f6;--mut:#666}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16171a;--fg:#e8e8ea;--card:#23252a;--mut:#9a9aa0}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;max-width:1200px;margin:auto;padding:16px}}
.card{{background:var(--card);border-radius:10px;padding:12px 16px;margin:14px 0}}
.card.optional{{opacity:.85}} h2{{font-size:15px;margin:0 0 8px}} h3{{font-size:13px;color:var(--mut);margin:6px 0}}
video{{width:100%;max-height:360px;background:#000;border-radius:6px}}
.cols{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
p{{white-space:pre-wrap;margin:0}} fieldset{{border:none;padding:8px 0 0}} label{{margin-right:12px}}
textarea{{width:100%;height:120px}}
</style></head><body>
<h1>ASR bake-off: Whisper large-v3 vs Parakeet-TDT-0.6B-v3</h1>
<p>Listen to the first 10 (marked LISTEN; largest disagreement). For each, pick which transcript is closer to what is actually <i>said</i>
(on-screen text does not count). Then press the button and paste the JSON back into the chat.</p>
{''.join(cards)}
<button onclick="document.getElementById('out').value=JSON.stringify(Object.fromEntries([...document.querySelectorAll('fieldset')].map(f=>[f.dataset.id,(f.querySelector('input:checked')||{{}}).value||null]).filter(x=>x[1])),null,1)">Collect my answers</button>
<textarea id="out" readonly></textarea>
</body></html>"""
    (out / "bakeoff_review.html").write_text(page, encoding="utf-8")
    print(f"\nwrote {out / 'bakeoff_metrics.csv'} and {out / 'bakeoff_review.html'}")


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "parakeet", "compare"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--whisper", help="compare: Whisper jsonl (from s04_asr.py --ids-file)")
    ap.add_argument("--parakeet", help="parakeet: output / compare: input jsonl")
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"sample": cmd_sample, "parakeet": cmd_parakeet, "compare": cmd_compare}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
