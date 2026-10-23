"""Speech with word timestamps for every video -> outputs/asr/asr_<shard>.jsonl.

Runs in the exist_asr environment, on a GPU. Per video:
  1. decode the mp4 audio to 16 kHz mono with PyAV (no ffmpeg binary needed);
  2. Silero VAD -> speech segments and the share of the video that is speech;
  3. language ID on the speech only (Whisper); the record's `lang` if unsure;
  4. faster-whisper large-v3; segments outside VAD speech, with Whisper's own
     no-speech signal, looping, or a known silence phrase ("Thanks for
     watching!") are dropped and listed with the reason;
  5. WhisperX wav2vec2 forced alignment -> word times (en/es; Whisper's own
     word times for other languages or if alignment fails).

Why not a fixed language: in the ASR bake-off (asr_bakeoff.py) a forced `es`
made Whisper translate English speech and lyrics into Spanish.

The organisers' `text` field is not used here: it mixes speech with
on-screen text and leaked LLM prompts (data_note.md), so it cannot be
aligned as a whole. Step s05 matches its sentences against these words.

Output: one JSON line per video, written as soon as it is done, so a killed
job resumes where it stopped. Videos already in the shard file are skipped.

Usage:
    python src/s04_asr.py --config config.perun.yaml --shard 0 --n-shards 4
    python src/s04_asr.py --config config.perun.yaml --limit 5      # smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config, require, setup_stdout

SR = 16000
WHISPER = "Systran/faster-whisper-large-v3"


def decode_audio(path: Path) -> np.ndarray:
    import av
    with av.open(str(path)) as c:
        stream = next((s for s in c.streams if s.type == "audio"), None)
        if stream is None:
            return np.zeros(0, dtype=np.float32)
        res = av.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks = []
        for frame in c.decode(stream):
            for f in res.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in res.resample(None):  # flush
            chunks.append(f.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


# Whisper's well-known outputs on silence or music (normalised: lowercase,
# letters only). Dropped only where VAD finds little speech under the segment.
SILENCE_PHRASES = {
    "thanks for watching", "thank you for watching", "thank you", "thanks", "you", "bye",
    "subscribe", "please subscribe", "like and subscribe", "suscribete", "suscribanse",
    "gracias por ver", "gracias por ver el video", "gracias", "mua", "music", "musica",
    "subtitulos realizados por la comunidad de amaraorg", "amaraorg",
}
LANG_MIN_PROB = 0.7      # below this, the record's language is used
MIN_SPEECH_S = 0.3       # less VAD speech than this: no transcription at all


def _norm(t: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", t.lower())
    return " ".join("".join(c for c in t if c.isalpha() or c == " ").split())


def _overlap(a0: float, a1: float, spans: list) -> float:
    """Share of [a0, a1] covered by the VAD speech spans."""
    if a1 <= a0:
        return 0.0
    cov = sum(max(0.0, min(a1, e) - max(a0, s)) for s, e in spans)
    return cov / (a1 - a0)


class Pipeline:
    def __init__(self, device: str):
        import torch
        from faster_whisper import WhisperModel
        from silero_vad import get_speech_timestamps, load_silero_vad
        import whisperx

        self.torch, self.whisperx = torch, whisperx
        self.device = device
        self.vad = load_silero_vad()
        self.get_ts = get_speech_timestamps
        self.asr = WhisperModel(WHISPER, device=device,
                                compute_type="float16" if device == "cuda" else "int8")
        self.aligners: dict = {}

    def aligner(self, lang: str):
        if lang not in self.aligners:
            self.aligners[lang] = self.whisperx.load_align_model(language_code=lang,
                                                                 device=self.device)
        return self.aligners[lang]

    def detect_language(self, speech_audio: np.ndarray) -> tuple[str, float]:
        """Whisper language ID on the VAD speech only (music would mislead it)."""
        try:
            lang, prob, _ = self.asr.detect_language(speech_audio)
            return lang, float(prob)
        except (AttributeError, TypeError):  # older faster-whisper: no detect_language
            _, info = self.asr.transcribe(speech_audio[: 30 * SR], language=None, beam_size=1)
            return info.language, float(info.language_probability)

    def __call__(self, audio: np.ndarray, lang_record: str) -> dict:
        dur = len(audio) / SR
        out = {"audio_s": round(dur, 3), "lang_record": lang_record}
        empty = dict(speech_s=0.0, speech_ratio=0.0, vad=[], segments=[], dropped=[], words=[],
                     lang_used=lang_record, lang_detected=None, lang_prob=None, word_source=None)
        if len(audio) < SR // 4:
            out.update(empty)
            return out
        vad = self.get_ts(self.torch.from_numpy(audio), self.vad, sampling_rate=SR,
                          return_seconds=True)
        spans = [(float(v["start"]), float(v["end"])) for v in vad]
        speech = sum(e - s for s, e in spans)
        out.update(empty)
        out.update(speech_s=round(speech, 3), speech_ratio=round(speech / dur, 4),
                   vad=[[round(s, 3), round(e, 3)] for s, e in spans])
        if speech < MIN_SPEECH_S:
            return out

        # 1. language, detected on the speech only; the record's language if unsure
        #    (a forced language makes Whisper translate foreign speech and lyrics)
        speech_audio = np.concatenate([audio[int(s * SR):int(e * SR)] for s, e in spans])
        det, prob = self.detect_language(speech_audio)
        lang = det if prob >= LANG_MIN_PROB else lang_record
        out.update(lang_detected=det, lang_prob=round(prob, 3), lang_used=lang)

        # 2. transcription, with Whisper's own word times as a fallback
        segs, _ = self.asr.transcribe(audio, language=lang, beam_size=5, vad_filter=True,
                                      condition_on_previous_text=False, word_timestamps=True)
        kept, dropped = [], []
        for sg in segs:
            ov = _overlap(sg.start, sg.end, spans)
            rec = {"start": round(sg.start, 3), "end": round(sg.end, 3), "text": sg.text,
                   "avg_logprob": round(sg.avg_logprob, 3),
                   "no_speech_prob": round(sg.no_speech_prob, 3),
                   "compression_ratio": round(sg.compression_ratio, 3), "vad_overlap": round(ov, 3),
                   "fw_words": [[w.word, round(w.start, 3), round(w.end, 3), round(w.probability, 3)]
                                for w in (sg.words or [])]}
            # 3. hallucination filters
            reason = None
            if ov < 0.3:
                reason = "outside_speech"
            elif sg.no_speech_prob > 0.6 and sg.avg_logprob < -1.0:
                reason = "no_speech_prob"
            elif sg.compression_ratio > 2.4:
                reason = "repetition_loop"
            elif _norm(sg.text) in SILENCE_PHRASES and ov < 0.8:
                reason = "silence_phrase"
            if reason:
                rec["reason"] = reason
                dropped.append(rec)
            else:
                kept.append(rec)
        out["dropped"] = [{k: v for k, v in d.items() if k != "fw_words"} for d in dropped]
        out["segments"] = [{k: v for k, v in d.items() if k != "fw_words"} for d in kept]
        if not kept:
            return out

        # 4. word times: WhisperX forced alignment for en/es, Whisper's own otherwise
        out["words"] = [w for sg in kept for w in sg["fw_words"]]
        out["word_source"] = "whisper"
        if lang in ("en", "es"):
            try:
                model, meta = self.aligner(lang)
                aligned = self.whisperx.align(
                    [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in kept],
                    model, meta, audio, self.device, return_char_alignments=False)
                # words WhisperX cannot place (digits, symbols) keep start/end = None
                out["words"] = [[w.get("word"), _r(w.get("start")), _r(w.get("end")),
                                 _r(w.get("score"))] for w in aligned.get("word_segments", [])]
                out["word_source"] = "whisperx"
            except Exception as e:  # recorded; Whisper's own word times are kept
                out["align_error"] = f"{type(e).__name__}: {e}"
        return out


def _r(x):
    return None if x is None else round(float(x), 3)


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N videos of the shard (smoke)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ids-file", help="only these id_EXIST (one per line), e.g. the ASR bake-off sample")
    ap.add_argument("--out", help="output jsonl (default: outputs/asr/asr_<shard>of<n>.jsonl)")
    a = ap.parse_args()
    require(0 <= a.shard < a.n_shards, "0 <= shard < n-shards", expected=a.n_shards, found=a.shard)
    cfg = load_config(a.config)

    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str, "id_Tiktok": str})
    vids = vids.sort_values("id_EXIST").reset_index(drop=True)
    mine = vids.iloc[a.shard::a.n_shards]
    if a.ids_file:
        ids = [l.strip() for l in open(a.ids_file, encoding="utf-8") if l.strip()]
        require(set(ids) <= set(vids.id_EXIST), "ids-file ids are known videos",
                expected="all known", found=sorted(set(ids) - set(vids.id_EXIST))[:5])
        mine = mine[mine.id_EXIST.isin(ids)]
    if a.limit:
        mine = mine.head(a.limit)
    dirs = {"train": cfg.path("train_videos_dir"), "test": cfg.path("test_videos_dir")}

    out_dir = cfg.path("out_dir") / "asr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(a.out) if a.out else         out_dir / (f"asr_{a.shard}of{a.n_shards}" + ("_smoke" if a.limit else "") + ".jsonl")
    done = set()
    if out_path.is_file():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["id_EXIST"] for line in f if line.strip()}
    todo = mine[~mine.id_EXIST.isin(done)]
    print(f"shard {a.shard}/{a.n_shards}: {len(mine)} videos, {len(done)} done, {len(todo)} to go")

    pipe = Pipeline(a.device)
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for n, r in enumerate(todo.itertuples(), 1):
            path = dirs[r.split] / f"{r.id_Tiktok}.mp4"
            require(path.is_file(), "video file exists", expected=str(path), found="missing")
            res = {"id_EXIST": r.id_EXIST, "lang": r.lang}
            res.update(pipe(decode_audio(path), r.lang))
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el / n:.1f} s/video  eta {el / n * (len(todo) - n) / 60:.0f} min",
                      flush=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
