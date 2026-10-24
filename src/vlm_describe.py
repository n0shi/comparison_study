"""Regenerate every VLM description with one model and one prompt (stage 2).

Reads the frames from stage 1 (vlm_frames.py) and runs Qwen3-VL-30B-A3B-
Instruct through vLLM with part 1's prompt (src/vlm/prompt.py, verbatim; the
v2_5sections_fix1 version) and part 1's generation settings. Inputs per video
are what part 1's controller sent: the frames, their timestamps, a transcript,
no OCR, the record's language. The transcript is `vlm_transcript` from
s05_moments.py: our ASR where it is reliable, otherwise the organisers' `text`
as part 1 used it, cleaned (src/text_clean.py; its raw residue misled the
model in the smoke test, a looping "chicken, chicken, ..." read as mockery).

A description that comes back truncated or with a missing section is
generated again with a new seed, up to --max-retries times.

Output: outputs/vlm/descriptions.jsonl, one line per video, appended as
batches finish, so a killed job resumes. --merge then writes
training_vlm_v3.json / test_vlm_v3.json in the layout part 1 and this package
already read: the original records with the vlm_* fields replaced, `text`
replaced by the organisers' cleaned text (C0's transcript input, as in the
paper; the original kept as `text_raw`), and the ASR / moment fields added
(asr_reliable, asr_text, moment_windows, moment_clips, moment_speech,
vlm_transcript, vlm_transcript_source).

Usage:
    python src/vlm_describe.py --config config.perun.yaml                  # generate
    python src/vlm_describe.py --config config.perun.yaml --merge          # build v3 JSONs
    python src/vlm_describe.py --config config.perun.yaml --frames outputs/vlm/frames_smoke.h5 \\
        --out outputs/vlm/descriptions_smoke.jsonl                         # smoke
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import h5py
import numpy as np

from common import load_config, load_moments, load_records, require, setup_stdout
from vlm.normalizer import is_truncated, parse_sections, to_train_text, validate_sections
from vlm.prompt import PROMPT_VERSION, build_user_prompt, get_system_prompt

MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
# vlm_provider/config.py, mapped from Ollama options to vLLM SamplingParams.
# repeat_penalty is NOT carried over: Ollama (llama.cpp) applies 1.05 to the
# last 64 tokens only, while vLLM's repetition_penalty applies to every token
# of the prompt as well, which pushes the model away from the transcript's and
# the instructions' own words (paraphrase and over-interpretation in the smoke
# test). vLLM has no windowed equivalent, so it is left at 1.0; runaway
# repetition would surface as truncated output and be retried.
GEN = dict(temperature=0.2, top_p=0.9, top_k=20, repetition_penalty=1.0, max_tokens=2400)
BASE_SEED = 1234


def status_of(description: str, sections: dict) -> tuple[str, str | None]:
    """vlm_provider/batch_process_dataset.py::_compute_status."""
    if is_truncated(description):
        return "truncated", "description appears truncated"
    missing = validate_sections(sections)
    if missing:
        return "missing_sections", f"missing or empty: {missing}"
    return "ok", None


def conversation(g: h5py.Group, transcript: str, lang: str) -> list[dict]:
    ts = [float(t) for t in g.attrs["timestamps"]]
    images = [{"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64," + base64.b64encode(bytes(b)).decode()}}
        for b in g["jpeg"]]
    text = build_user_prompt(num_frames=len(ts), transcript=transcript or None, ocr_text=None,
                             lang=lang, frame_timestamps=ts)
    return [{"role": "system", "content": get_system_prompt(lang)},
            {"role": "user", "content": images + [{"type": "text", "text": text}]}]


def generate(cfg, a) -> None:
    from vllm import LLM, SamplingParams

    frames_path = Path(a.frames) if a.frames else cfg.path("out_dir") / "vlm" / "frames.h5"
    out_path = Path(a.out) if a.out else cfg.path("out_dir") / "vlm" / "descriptions.jsonl"
    recs = {str(r["id_EXIST"]): r for k in ("train_json", "test_json")
            for r in load_records(cfg.path(k))}
    moments = load_moments(cfg, a.moments)
    done = set()
    if out_path.is_file():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["id_EXIST"] for l in f if l.strip()}

    h = h5py.File(frames_path, "r")
    todo = [k for k in sorted(h.keys()) if k not in done]
    if a.limit:
        todo = todo[:a.limit]
    require(all(k in recs and k in moments for k in todo),
            "every sampled video has a dataset record and moments",
            expected="all", found=[k for k in todo if k not in recs or k not in moments][:5])
    print(f"{len(h)} videos with frames, {len(done)} described, {len(todo)} to go -> {out_path}")
    if not todo:
        return

    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=a.max_model_len,
              limit_mm_per_prompt={"image": 14, "video": 0},
              gpu_memory_utilization=0.9, seed=BASE_SEED)
    t0, n_done = time.time(), 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for start in range(0, len(todo), a.batch):
            batch = todo[start:start + a.batch]
            results: dict[str, dict] = {}
            pending = list(batch)
            for attempt in range(a.max_retries + 1):
                if not pending:
                    break
                convs, params = [], []
                for k in pending:
                    r = recs[k]
                    lang = (r.get("lang") or "en").strip().lower()
                    convs.append(conversation(h[k], moments[k]["vlm_transcript"], lang))
                    params.append(SamplingParams(seed=BASE_SEED + 1000 * attempt + int(k), **GEN))
                outs = llm.chat(convs, params, use_tqdm=False)
                still = []
                for k, o in zip(pending, outs):
                    desc = (o.outputs[0].text or "").strip()
                    sections = parse_sections(desc)
                    status, err = status_of(desc, sections)
                    results[k] = {
                        "id_EXIST": k, "status": status, "error": err, "attempt": attempt,
                        "description": desc, "sections": sections,
                        "finish_reason": o.outputs[0].finish_reason,
                        "n_prompt_tokens": len(o.prompt_token_ids or []),
                        "n_output_tokens": len(o.outputs[0].token_ids),
                        "n_frames": int(h[k]["jpeg"].shape[0]),
                        "frame_timestamps": [float(t) for t in h[k].attrs["timestamps"]],
                        "model": MODEL, "prompt_version": PROMPT_VERSION,
                        "transcript_source": "asr" if moments[k]["reliable"] else "organisers",
                        "generation": {**GEN, "seed": BASE_SEED + 1000 * attempt + int(k)},
                    }
                    if status != "ok":
                        still.append(k)
                pending = still
            for k in batch:
                fout.write(json.dumps(results[k], ensure_ascii=False) + "\n")
            fout.flush()
            n_done += len(batch)
            el = time.time() - t0
            bad = sum(results[k]["status"] != "ok" for k in batch)
            print(f"  {n_done}/{len(todo)}  {el / n_done:.2f} s/video  not ok in batch: {bad}  "
                  f"eta {el / n_done * (len(todo) - n_done) / 60:.0f} min", flush=True)


def merge(cfg, a) -> None:
    src = Path(a.out) if a.out else cfg.path("out_dir") / "vlm" / "descriptions.jsonl"
    with open(src, encoding="utf-8") as f:
        desc = {d["id_EXIST"]: d for d in map(json.loads, f) if d}
    out_dir = cfg.path("out_dir") / "vlm"
    moments = load_moments(cfg, a.moments)
    for key, name in (("train_json", "training_vlm_v3.json"), ("test_json", "test_vlm_v3.json")):
        recs = load_records(cfg.path(key))
        missing = [str(r["id_EXIST"]) for r in recs if str(r["id_EXIST"]) not in desc]
        require(not missing, f"every {key} record has a new description",
                expected=len(recs), found=f"{len(missing)} missing, e.g. {missing[:5]}")
        not_ok = [k for k in (str(r["id_EXIST"]) for r in recs) if desc[k]["status"] != "ok"]
        require(len(not_ok) <= a.allow_not_ok, f"{key}: descriptions with status ok",
                expected=f"at most {a.allow_not_ok} not ok", found=f"{len(not_ok)}: {not_ok[:10]}")
        merged = {}
        for r in recs:
            k = str(r["id_EXIST"])
            d = desc[k]
            lang = (r.get("lang") or "en").strip().lower()
            m = moments[k]
            new = dict(r)
            new.update(
                text=m["org_text"], text_raw=r.get("text"), text_cleaning="comparison_study/text_clean.py",
                asr_reliable=m["reliable"], asr_text=m["asr_text"], asr_lang=m["lang_used"],
                moment_windows=m["windows"], moment_clips=m["clips"], moment_speech=m["speech"],
                vlm_transcript=m["vlm_transcript"],
                vlm_transcript_source="asr" if m["reliable"] else "organisers",
                vlm_prompt_version=d["prompt_version"], vlm_model=d["model"], vlm_lang_used=lang,
                vlm_status=d["status"], vlm_error=d["error"], vlm_description=d["description"],
                vlm_sections=d["sections"],
                vlm_train_text=to_train_text({"description": d["description"], "lang_used": lang},
                                             transcript=m["vlm_transcript"], ocr_text=None, lang=lang)
                if d["status"] == "ok" else None,
                vlm_num_frames=d["n_frames"], vlm_frame_timestamps=d["frame_timestamps"],
                vlm_generation=d["generation"], vlm_regenerated="comparison_study/vlm_describe.py",
            )
            merged[k] = new
        path = out_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=1)
        print(f"wrote {path}: {len(merged)} records, {len(not_ok)} not ok")


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--frames", help="default: outputs/vlm/frames.h5")
    ap.add_argument("--out", help="default: outputs/vlm/descriptions.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--moments", help="default: outputs/moments/moments.jsonl")
    ap.add_argument("--allow-not-ok", type=int, default=0,
                    help="merge: tolerate this many descriptions that stayed truncated")
    a = ap.parse_args()
    cfg = load_config(a.config)
    (merge if a.merge else generate)(cfg, a)


if __name__ == "__main__":
    main()
