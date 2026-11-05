"""W3 smoke with audio-visual models: the model watches the whole video with its
sound, instead of one still frame per moment.

W3 (w3_vlm_edges.py) shows Qwen3-VL the middle frame of each H5 clip, and a clip
covers ~2 s of a moment that can last 15 s. Its links came back almost uniform:
91 % of fields linked to all 4 frames (data_note 16). This script asks the same
question, prompt v4 (score every moment 0 / 1 / 2 per field, edge = 2), of two
models that take the video file itself, with audio:

  omni     Qwen/Qwen3-Omni-30B-A3B-Instruct (thinker only), transformers >= 4.57.
           Frames at 2 fps (at most --max-frames, then evenly spread), audio
           16 kHz mono, interleaved in time (use_audio_in_video).
  salmonn  video-SALMONN 2+ 7B (Qwen2.5-VL based, Apache-2.0), through the
           authors' code (--salmonn-repo, bytedance/video-SALMONN-2,
           video_SALMONN2_plus/), with the frame and audio settings of their
           inference.py. Its audio is decoded with PyAV here instead of
           torchcodec (no FFmpeg libraries needed); see torchcodec_shim().

Both see the moments as time ranges (the windows of s05_moments.py), the ASR of
each moment (as W3 did, so only the visual / audio input changes) and the four
description fields. No labels anywhere.

The answers have W3's format, so w3_vlm_edges.py --summary / --review --compare
read them. With --compare, this script also prints how far the links agree with
another edges file (W3: outputs/w3/edges.jsonl).

Usage (Perun, one GPU; cluster/av_edges.slurm):
    python src/av_edges.py --config config.perun.yaml --model omni --sample 20 \\
        --out outputs/w3/edges_smoke_av_omni.jsonl --compare outputs/w3/edges.jsonl
    python src/av_edges.py --config config.perun.yaml --model salmonn --sample 20 \\
        --salmonn-repo /mnt/project/perun2601420/ext/video-SALMONN-2 \\
        --out outputs/w3/edges_smoke_av_salmonn.jsonl --compare outputs/w3/edges.jsonl
    python src/av_edges.py --config config.yaml --dry-run 120001     # prompt + decoded inputs, no GPU
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

import w3_vlm_edges as w3
from common import load_config, require, setup_stdout

MODELS = {"omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct", "salmonn": "tsinghua-ee/video-SALMONN2_plus_7B_full"}
SR = 16000
FPS = 2.0
MAX_TOKENS = 700
SEED = 4321

# W3's v4 task, with the moments as stretches of the video the model watches and
# hears instead of single frames; the keys stay "frames" / "speech", so W3's
# parser, summary and review read the answers unchanged
TASK_AV = (w3.TASK_V4
           .replace("its frame shows (its speech says) specifically",
                    "what is seen in it (what is said in it) shows specifically")
           .replace('Then give "frames": the scores of the frames of moments 1, 2, 3, 4, and "speech": '
                    'the scores of the speech of moments 1, 2, 3, 4.',
                    'Then give "frames": the scores of what is seen in moments 1, 2, 3, 4 (any part of the '
                    'moment, including actions and on-screen text), and "speech": the scores of what is said '
                    'in moments 1, 2, 3, 4.')
           .replace("Judge each moment on its own frame and speech", "Judge each moment on what is seen and said in it"))
assert TASK_AV.count("what is seen") == 3, "TASK_V4 changed: update TASK_AV"


def video_path(cfg, inp: w3.Inputs, k: str) -> Path:
    v = inp.videos.loc[k]
    p = cfg.path(f"{v.split}_videos_dir") / f"{v.id_Tiktok}.mp4"
    require(p.is_file(), f"{k}: mp4 exists", expected=str(p), found="missing")
    return p


def prompt_text(inp: w3.Inputs, k: str) -> str:
    m = inp.moments[k]
    moments = "\n".join(f"Moment {i}: {a:.1f}-{b:.1f} s" for i, (a, b) in enumerate(m["windows"], 1))
    speech = "\n".join(f'Moment {i}: "{s.strip()}"' if s.strip() else f"Moment {i}: (no speech)"
                       for i, s in enumerate(m["speech"], 1))
    fields = "\n".join(f"{f}: {t or '(empty)'}" for f, t in inp.fields(k).items())
    return (f"The video is {m['duration_s']:.1f} s long. It is split into four moments:\n{moments}\n\n"
            f"Speech in each moment (automatic transcript, may contain errors):\n{speech}\n\n"
            + TASK_AV.format(fields=fields))


def decode_audio(path: Path) -> np.ndarray:
    """16 kHz mono float32, broken packets skipped (s04_asr.py's decoder)."""
    from s04_asr import decode_audio_checked
    return decode_audio_checked(path)[0]


def decode_frames(path: Path, duration: float, max_frames: int, max_pixels: int) -> tuple[np.ndarray, float]:
    """RGB frames (T, H, W, 3) at FPS, evenly spread when more than max_frames (T even:
    the processors merge frames in pairs), each shrunk to at most max_pixels; and
    the rate they stand for (frames per second). Decodes in one pass and keeps only
    the frames it takes."""
    import av
    from PIL import Image
    n = int(min(max_frames, max(4, round(duration * FPS))))
    n -= n % 2
    want = (np.arange(n) + 0.5) * duration / n            # centre of each 1/n of the video
    out, last = [], None
    with av.open(str(path)) as c:
        for fr in c.decode(c.streams.video[0]):
            if fr.time is None:
                continue
            last = fr
            while len(out) < n and fr.time >= want[len(out)]:
                out.append(fr.to_image())
            if len(out) == n:
                break
    require(last is not None, f"{path.name}: decodable frames", expected=">0", found=0)
    while len(out) < n:                                    # stream shorter than the stated duration
        out.append(last.to_image())
    arr = []
    for img in out:
        scale = min(1.0, (max_pixels / (img.width * img.height)) ** 0.5)
        if scale < 1:
            img = img.resize((max(28, int(img.width * scale)), max(28, int(img.height * scale))), Image.BICUBIC)
        arr.append(np.asarray(img.convert("RGB")))
    return np.stack(arr), n / duration


# ------------------------------------------------------------------ omni
class Omni:
    def __init__(self, a):
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        self.torch, self.a = torch, a
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODELS["omni"], torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        self.model.disable_talker()
        self.proc = Qwen3OmniMoeProcessor.from_pretrained(MODELS["omni"])

    def __call__(self, video: Path, duration: float, text: str, sample: bool, seed: int) -> tuple[str, dict]:
        frames, fps = decode_frames(video, duration, self.a.max_frames, self.a.max_pixels)
        audio = decode_audio(video)
        require(len(audio) > 0, f"{video.name}: an audio track (use_audio_in_video)", expected=">0 samples", found=0)
        conv = [{"role": "system", "content": [{"type": "text", "text": w3.SYSTEM}]},
                {"role": "user", "content": [{"type": "video", "video": str(video)}, {"type": "text", "text": text}]}]
        chat = self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = self.proc(text=chat, audio=[audio], videos=[frames], fps=fps, return_tensors="pt",
                           padding=True, use_audio_in_video=True).to("cuda").to(self.model.dtype)
        self.torch.manual_seed(seed)
        gen = dict(thinker_max_new_tokens=MAX_TOKENS, thinker_do_sample=sample)
        if sample:
            gen.update(thinker_temperature=0.3, thinker_top_p=0.9, thinker_top_k=20)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, thinker_return_dict_in_generate=True, use_audio_in_video=True,
                                      return_audio=False, **gen)
        out = out[0] if isinstance(out, tuple) else out
        seq = getattr(out, "sequences", out)
        n_in = inputs["input_ids"].shape[1]
        txt = self.proc.batch_decode(seq[:, n_in:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return txt, {"n_frames": int(len(frames)), "fps": round(fps, 3), "frame_hw": list(frames.shape[1:3]),
                     "audio_s": round(len(audio) / SR, 2), "n_prompt_tokens": int(n_in),
                     "n_output_tokens": int(seq.shape[1] - n_in)}


# ------------------------------------------------------------------ video-SALMONN 2+
def torchcodec_shim() -> None:
    """The authors' dataset.py imports torchcodec, which needs FFmpeg shared
    libraries. This stands in for it: VideoDecoder fails, so their code falls
    back to decord (its own reader, same sampling settings); AudioDecoder decodes
    with PyAV to the rate and channels they ask for."""
    import torch

    class VideoDecoder:
        def __init__(self, *a, **k):
            raise RuntimeError("torchcodec shim: video goes through decord")

    class AudioDecoder:
        def __init__(self, file, sample_rate: int, num_channels: int):
            require(sample_rate == SR and num_channels == 1, "SALMONN asks for 16 kHz mono",
                    expected=(SR, 1), found=(sample_rate, num_channels))
            self.audio = decode_audio(Path(file))

        def get_all_samples(self):
            return types.SimpleNamespace(data=torch.from_numpy(self.audio)[None])

    tc, dec = types.ModuleType("torchcodec"), types.ModuleType("torchcodec.decoders")
    dec.VideoDecoder, dec.AudioDecoder, tc.decoders = VideoDecoder, AudioDecoder, dec
    sys.modules["torchcodec"], sys.modules["torchcodec.decoders"] = tc, dec


class Salmonn:
    def __init__(self, a):
        import torch
        root = Path(a.salmonn_repo) / "video_SALMONN2_plus"
        require((root / "qwenvl").is_dir(), "--salmonn-repo is a clone of bytedance/video-SALMONN-2",
                expected=str(root / "qwenvl"), found="missing")
        sys.path.insert(0, str(root))
        torchcodec_shim()
        from transformers import AutoTokenizer, WhisperFeatureExtractor
        from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
        from qwenvl.data.dataset import make_supervised_data_module
        from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
        from qwenvl.train.argument import DataArguments
        path = a.salmonn_model
        d = DataArguments()                      # inference.py's prepare_dataset(), verbatim settings
        d.video_max_frames, d.video_min_frames, d.base_interval = 768, 16, 0.1
        d.max_pixels = d.video_max_frame_pixels = 61250
        d.run_test = True
        d.image_processor = Qwen2VLImageProcessorFast.from_pretrained(path)
        d.audio_processor = WhisperFeatureExtractor(feature_size=d.feature_size, sampling_rate=d.sampling_rate,
                                                    hop_length=d.hop_length, chunk_length=d.chunk_length)
        d.model_type = "qwen2.5vl"
        self.tok = AutoTokenizer.from_pretrained(path, model_max_length=131072, padding_side="right", use_fast=False)
        self.model = video_SALMONN2_plus.from_pretrained(path, attn_implementation="sdpa",
                                                         torch_dtype=torch.bfloat16, device_map="cpu").cuda().eval()
        self.data = make_supervised_data_module(tokenizer=self.tok, data_args=d)["train_dataset"]
        self.torch = torch

    def __call__(self, video: Path, duration: float, text: str, sample: bool, seed: int) -> tuple[str, dict]:
        item = self.data._get_item({"video": str(video), "use_audio": True, "conversations": [
            {"from": "human", "value": f"<video>\n{w3.SYSTEM}\n\n{text}"}, {"from": "gpt", "value": ""}]})
        # their process_audio returns None on any error (bare except): fail loudly instead
        require(item.get("audio_feature") is not None, f"{video.name}: audio decoded", expected="features", found=None)
        n_frames = int(item["video_grid_thw"][0][0]) * 2 if "video_grid_thw" in item else 0
        inputs = {k: v.to("cuda") for k, v in item.items() if isinstance(v, self.torch.Tensor)}
        self.torch.manual_seed(seed)
        gen = dict(max_new_tokens=MAX_TOKENS, do_sample=sample)
        if sample:
            gen.update(temperature=0.3, top_p=0.9, top_k=20)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **gen)
        n_in = inputs["input_ids"].shape[1]
        txt = self.tok.decode(out[0, n_in:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return txt, {"n_frames": n_frames, "n_prompt_tokens": int(n_in), "n_output_tokens": int(out.shape[1] - n_in)}


# ------------------------------------------------------------------ run
def agreement(path: Path, ref: Path) -> None:
    """Per field and part: Jaccard of the linked moments, and how often each file links all 4."""
    a = {d["id_EXIST"]: d for d in map(json.loads, open(path, encoding="utf-8")) if d.get("edges")}
    b = {d["id_EXIST"]: d for d in map(json.loads, open(ref, encoding="utf-8")) if d.get("edges")}
    common = sorted(set(a) & set(b))
    require(len(common) > 0, f"videos in both {path.name} and {ref.name}", expected=">0", found=0)
    print(f"\nagreement with {ref.name} on {len(common)} videos (Jaccard of linked moments; "
          f"both empty counts as 1)")
    for part in ("frames", "speech"):
        jac, all4 = [], [0, 0]
        for k in common:
            for f in w3.FIELDS:
                x, y = set(a[k]["edges"][f][part]), set(b[k]["edges"][f][part])
                jac.append(1.0 if not (x | y) else len(x & y) / len(x | y))
                all4[0] += len(x) == w3.N_WIN
                all4[1] += len(y) == w3.N_WIN
        n = len(common) * len(w3.FIELDS)
        print(f"  {part:6s} Jaccard {np.mean(jac):.2f}; fields linked to all 4: this {all4[0] / n:.0%}, "
              f"{ref.stem} {all4[1] / n:.0%}")


def generate(cfg, a) -> Path:
    inp = w3.Inputs(cfg)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    version = f"av_{a.model}_v4"
    done = set()
    if out.is_file():
        prev = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
        require({d["prompt_version"] for d in prev} <= {version}, f"{out.name} holds only {version} (resume)",
                expected=version, found=sorted({d["prompt_version"] for d in prev}))
        done = {d["id_EXIST"] for d in prev}
    ids = w3.pick_sample(inp, a.sample) if a.sample else sorted(inp.videos.index)
    todo = [k for k in ids if k not in done]
    print(f"{len(ids)} videos, {len(ids) - len(todo)} done, {len(todo)} to go -> {out}", flush=True)
    if not todo:
        return out
    t0 = time.time()
    run = Omni(a) if a.model == "omni" else Salmonn(a)
    print(f"model {a.model} loaded in {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    with open(out, "a", encoding="utf-8", newline="\n") as f:
        for n, k in enumerate(todo, 1):
            video, text, hs = video_path(cfg, inp, k), prompt_text(inp, k), inp.has_speech(k)
            for attempt in range(a.max_retries + 1):
                seed = SEED + 1000 * attempt + int(k)
                raw, info = run(video, inp.moments[k]["duration_s"], text, sample=attempt > 0, seed=seed)
                status, edges, err = w3.parse(raw, hs, w3.N_WIN, "scores")
                if status == "ok":
                    break
            if status in ("cites_empty_speech", "too_many"):
                edges, status = w3.clean(edges, hs, w3.N_WIN), "ok_cleaned"
            f.write(json.dumps({"id_EXIST": k, "status": status, "error": err, "attempt": attempt, "edges": edges,
                                "raw": raw, "has_speech": hs, "model": MODELS[a.model] if a.model == "omni"
                                else a.salmonn_model, "prompt_version": version, "inputs": info,
                                "generation": {"sampled": attempt > 0, "seed": seed, "max_tokens": MAX_TOKENS}},
                               ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"  {n}/{len(todo)} {k} {status} (attempt {attempt}) {info}  {el / n:.1f} s/video", flush=True)
    return out


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", choices=list(MODELS), default="omni")
    ap.add_argument("--out", default=None, help="default: outputs/w3/edges_smoke_av_<model>.jsonl")
    ap.add_argument("--sample", type=int, default=20, help="W3's balanced train sample; 0 = every video")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=64, help="omni: frames per video (2 fps up to this)")
    ap.add_argument("--max-pixels", type=int, default=360 * 640, help="omni: pixels per frame")
    ap.add_argument("--salmonn-repo", default=None, help="clone of github.com/bytedance/video-SALMONN-2")
    ap.add_argument("--salmonn-model", default=MODELS["salmonn"])
    ap.add_argument("--compare", default=None, help="edges jsonl to measure agreement with (W3)")
    ap.add_argument("--dry-run", default=None, metavar="ID", help="print the prompt, decode the inputs; no model")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.dry_run:
        inp = w3.Inputs(cfg)
        v = video_path(cfg, inp, a.dry_run)
        print("SYSTEM:", w3.SYSTEM, "\n\nUSER: <video with audio>\n" + prompt_text(inp, a.dry_run))
        fr, fps = decode_frames(v, inp.moments[a.dry_run]["duration_s"], a.max_frames, a.max_pixels)
        au = decode_audio(v)
        print(f"\nframes {fr.shape} at {fps:.2f} fps; audio {len(au) / SR:.2f} s")
        return
    require(a.model != "salmonn" or a.salmonn_repo, "--salmonn-repo for --model salmonn", expected="a path", found=None)
    out = generate(cfg, argparse.Namespace(**{**vars(a), "out": a.out or str(
        cfg.out(f"w3/edges_smoke_av_{a.model}.jsonl"))}))
    w3.summarise(out)
    if a.compare:
        agreement(out, Path(a.compare))


if __name__ == "__main__":
    main()
