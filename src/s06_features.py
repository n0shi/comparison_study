"""Step 6: frozen-encoder features for every video, cached once for all graphs.

One h5 per encoder in outputs/features/, keyed by id_EXIST, so a stage can be
re-run or resumed alone (videos already in the file are skipped). Float16.

  xlmr      XLM-R-large, mean of the last layer over real tokens
              fields  (4, 1024)  TARGET, MESSAGE, SITUATION, PEOPLE (new descriptions)
              speech  (4, 1024)  ASR speech of each moment (zeros where empty)
              speech_mask (4,)   1 where the moment has speech
              org     (1, 1024)  the organisers' cleaned text, whole
  videomae  VideoMAE-base with its attention biases loaded (videomae_fix.py)
              clips   (4, 768)   mean over tokens of each H5 clip (16 frames)
  whisper   Whisper large-v3 encoder
              audio   (4, 1280)  mean of the encoder frames inside each moment
              audio_mask (4,)    1 where the moment has audio frames
  bge       BGE-M3 dense, L2-normalised (edges and neighbour search)
              fields  (4, 1024), speech (4, 1024), speech_mask (4,)
              desc    (1024,)    TARGET + MESSAGE      (video vector)
              transcript (1024,) the VLM transcript    (video vector)
  siglip    SigLIP 2 so400m, L2-normalised
              clips   (4, 1152)  middle frame of each H5 clip
              fields  (4, 1152)  first sentence of each field, lower-cased
                                 (SigLIP 2 reads 64 tokens, trained on lower case)
              video   (1152,)    mean of the clip vectors (video vector)
  bge_text / e5 / e5_small / m2v   text encoders in xlmr's layout (text_stage)
  pca64     label-free PCA to 64 dimensions, fitted on every video (train and
            test, no labels): e5_pca64.h5 (e5's layout) and videomae_pca64.h5
            (clips); the low-dimensional inputs of choices.md's encoder check

Moments come from s05_moments.py; texts from the v3 JSONs (vlm_describe.py).

Usage:
    python src/s06_features.py --config config.perun.yaml --stage xlmr
    python src/s06_features.py --config config.perun.yaml --stage all --limit 4 --out-dir outputs/features_smoke
"""
from __future__ import annotations

import argparse
import io
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from common import load_config, load_moments, load_records, require, setup_stdout

FIELDS = ["TARGET", "MESSAGE", "SITUATION", "PEOPLE"]
N_WIN = 4
STAGES = ["xlmr", "videomae", "whisper", "bge", "siglip", "bge_text", "e5",
          "e5_small", "m2v", "pca64"]                                   # new stages go last
MODELS = {
    "xlmr": "FacebookAI/xlm-roberta-large",
    "videomae": "MCG-NJU/videomae-base",
    "whisper": "openai/whisper-large-v3",
    "bge": "BAAI/bge-m3",
    "siglip": "google/siglip2-so400m-patch14-384",
    "bge_text": "BAAI/bge-m3",
    "e5": "intfloat/multilingual-e5-large",
    "e5_small": "intfloat/multilingual-e5-small",
    "m2v": "minishlab/potion-multilingual-128M",          # Model2Vec static embeddings (from BGE-M3)
    "pca64": "PCA of e5.h5 and videomae.h5 (sklearn)",
}
PCA_DIMS = 64
SR = 16000


# ------------------------------------------------------------------ inputs
class Inputs:
    def __init__(self, cfg, limit: int):
        self.cfg = cfg
        recs = {}
        for k in ("train_json_v3", "test_json_v3"):
            for r in load_records(cfg.path(k)):
                recs[str(r["id_EXIST"])] = r
        self.recs = recs
        self.moments = load_moments(cfg)
        v = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str, "id_Tiktok": str})
        v = v.sort_values("id_EXIST")
        if limit:
            v = pd.concat([v[v.split == "train"].head(limit // 2 or 1), v[v.split == "test"].head(limit - limit // 2)])
        missing = [i for i in v.id_EXIST if i not in recs or i not in self.moments]
        require(not missing, "every video has a v3 record and moments", expected=len(v),
                found=f"{len(missing)} missing, e.g. {missing[:5]}")
        self.videos = v.set_index("id_EXIST")

    def fields(self, k: str) -> list[str]:
        sec = self.recs[k].get("vlm_sections") or {}
        return [(sec.get(f) or "").strip() for f in FIELDS]


def open_out(path: Path) -> tuple[h5py.File, set]:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = h5py.File(path, "a")
    h.attrs["created_by"] = "comparison_study/src/s06_features.py"
    return h, set(h.keys())


def put(h: h5py.File, k: str, arrays: dict) -> None:
    g = h.create_group(k)
    for name, arr in arrays.items():
        g.create_dataset(name, data=np.asarray(arr, dtype=np.float16 if np.asarray(arr).dtype.kind == "f"
                                               else np.asarray(arr).dtype))


def progress(n: int, total: int, t0: float, every: int = 200) -> None:
    if n % every == 0 or n == total:
        el = time.time() - t0
        print(f"  {n}/{total}  {el / n:.2f} s/video  eta {el / n * (total - n) / 60:.0f} min", flush=True)


# ------------------------------------------------------------------ stages
def stage_xlmr(inp: Inputs, out: Path, device: str) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS["xlmr"])
    model = AutoModel.from_pretrained(MODELS["xlmr"], torch_dtype=torch.float16).to(device).eval()

    @torch.no_grad()
    def enc(texts: list[str]) -> np.ndarray:
        b = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        return ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()

    h5, done = open_out(out / "xlmr.h5")
    todo = [k for k in inp.videos.index if k not in done]
    t0 = time.time()
    for n, k in enumerate(todo, 1):
        m = inp.moments[k]
        speech = m["speech"]
        mask = np.array([bool(s.strip()) for s in speech], dtype=np.uint8)
        texts = inp.fields(k) + [s for s in speech if s.strip()] + [m["org_text"] or " "]
        e = enc(texts)
        sp = np.zeros((N_WIN, e.shape[1]), np.float32)
        sp[mask.astype(bool)] = e[4:4 + mask.sum()]
        put(h5, k, {"fields": e[:4], "speech": sp, "speech_mask": mask, "org": e[-1:]})
        progress(n, len(todo), t0)
    h5.close()


def stage_videomae(inp: Inputs, out: Path, device: str) -> None:
    import torch
    from PIL import Image
    from transformers import VideoMAEImageProcessor
    from videomae_fix import load_videomae
    proc = VideoMAEImageProcessor.from_pretrained(MODELS["videomae"])
    model = load_videomae(MODELS["videomae"]).to(device).eval().half()
    h5in = {s: h5py.File(inp.cfg.path(f"{s}_h5"), "r") for s in ("train", "test")}
    h5, done = open_out(out / "videomae.h5")
    todo = [k for k in inp.videos.index if k not in done]
    t0 = time.time()
    for n, k in enumerate(todo, 1):
        v = inp.videos.loc[k]
        clips = h5in[v.split]["clips"][v.id_Tiktok]
        require(clips.shape[0] == N_WIN, f"{k}: 4 H5 clips", expected=N_WIN, found=clips.shape)
        videos = [[np.asarray(Image.open(io.BytesIO(bytes(f))).convert("RGB")) for f in clips[c]]
                  for c in range(N_WIN)]
        px = proc(videos, return_tensors="pt")["pixel_values"].to(device).half()
        with torch.no_grad():
            emb = model(pixel_values=px).last_hidden_state.mean(1).float().cpu().numpy()
        put(h5, k, {"clips": emb})
        progress(n, len(todo), t0)
    h5.close()


def stage_whisper(inp: Inputs, out: Path, device: str) -> None:
    import torch
    from transformers import WhisperFeatureExtractor, WhisperModel
    from s04_asr import decode_audio
    fe = WhisperFeatureExtractor.from_pretrained(MODELS["whisper"])
    enc = WhisperModel.from_pretrained(MODELS["whisper"], torch_dtype=torch.float16).get_encoder()
    enc = enc.to(device).eval()
    dirs = {"train": inp.cfg.path("train_videos_dir"), "test": inp.cfg.path("test_videos_dir")}
    h5, done = open_out(out / "whisper.h5")
    todo = [k for k in inp.videos.index if k not in done]
    chunk = 30 * SR
    frame_s = 0.02  # one encoder frame per 20 ms (1500 frames per 30 s)
    t0 = time.time()
    for n, k in enumerate(todo, 1):
        v = inp.videos.loc[k]
        audio = decode_audio(dirs[v.split] / f"{v.id_Tiktok}.mp4")
        frames = []
        for s in range(0, max(len(audio), 1), chunk):
            piece = audio[s:s + chunk]
            if len(piece) == 0:
                break
            feats = fe(piece, sampling_rate=SR, return_tensors="pt").input_features.to(device).half()
            with torch.no_grad():
                hid = enc(feats).last_hidden_state[0].float().cpu().numpy()
            n_real = int(np.ceil(len(piece) / SR / frame_s))  # drop the padding frames
            frames.append(hid[:n_real])
        frames = np.concatenate(frames) if frames else np.zeros((0, 1280), np.float32)
        times = np.arange(len(frames)) * frame_s
        audio_emb, mask = np.zeros((N_WIN, 1280), np.float32), np.zeros(N_WIN, np.uint8)
        for w, (a, b) in enumerate(inp.moments[k]["windows"]):
            sel = (times >= a) & (times < b)
            if sel.any():
                audio_emb[w], mask[w] = frames[sel].mean(0), 1
        put(h5, k, {"audio": audio_emb, "audio_mask": mask})
        progress(n, len(todo), t0)
    h5.close()


def stage_bge(inp: Inputs, out: Path, device: str) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer
    # BGE-M3 dense = the CLS vector of the last layer, L2-normalised
    tok = AutoTokenizer.from_pretrained(MODELS["bge"])
    model = AutoModel.from_pretrained(MODELS["bge"], torch_dtype=torch.float16).to(device).eval()

    @torch.no_grad()
    def enc(texts: list[str]) -> np.ndarray:
        b = tok(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
        cls = model(**b).last_hidden_state[:, 0].float()
        return torch.nn.functional.normalize(cls, dim=-1).cpu().numpy()

    h5, done = open_out(out / "bge.h5")
    todo = [k for k in inp.videos.index if k not in done]
    t0 = time.time()
    for n, k in enumerate(todo, 1):
        m = inp.moments[k]
        f = inp.fields(k)
        mask = np.array([bool(s.strip()) for s in m["speech"]], dtype=np.uint8)
        texts = f + [f"{f[0]} {f[1]}", m["vlm_transcript"] or " "] + [s for s in m["speech"] if s.strip()]
        e = enc(texts)
        sp = np.zeros((N_WIN, e.shape[1]), np.float32)
        sp[mask.astype(bool)] = e[6:]
        put(h5, k, {"fields": e[:4], "desc": e[4], "transcript": e[5], "speech": sp, "speech_mask": mask})
        progress(n, len(todo), t0)
    h5.close()


def first_sentence(t: str) -> str:
    s = re.split(r"(?<=[.!?])\s+", (t or "").strip())[0]
    return s.lower() or " "


def stage_siglip(inp: Inputs, out: Path, device: str) -> None:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODELS["siglip"])
    model = AutoModel.from_pretrained(MODELS["siglip"], torch_dtype=torch.float16).to(device).eval()
    h5in = {s: h5py.File(inp.cfg.path(f"{s}_h5"), "r") for s in ("train", "test")}
    h5, done = open_out(out / "siglip.h5")
    todo = [k for k in inp.videos.index if k not in done]
    def norm(x):
        # transformers 5 returns BaseModelOutputWithPooling (features in pooler_output);
        # 4.x returns the feature tensor itself
        x = x if torch.is_tensor(x) else x.pooler_output
        return x / x.norm(dim=-1, keepdim=True)
    t0 = time.time()
    for n, k in enumerate(todo, 1):
        v = inp.videos.loc[k]
        clips = h5in[v.split]["clips"][v.id_Tiktok]
        imgs = [Image.open(io.BytesIO(bytes(clips[c, clips.shape[1] // 2]))).convert("RGB")
                for c in range(N_WIN)]
        texts = [first_sentence(t) for t in inp.fields(k)]
        with torch.no_grad():
            ib = proc(images=imgs, return_tensors="pt").to(device)
            img = norm(model.get_image_features(pixel_values=ib["pixel_values"].half())).float()
            tb = proc(text=texts, padding="max_length", max_length=64, truncation=True,
                      return_tensors="pt").to(device)
            txt = norm(model.get_text_features(**tb)).float()
        img = img.cpu().numpy()
        put(h5, k, {"clips": img, "fields": txt.cpu().numpy(),
                    "video": img.mean(0) / np.linalg.norm(img.mean(0))})
        progress(n, len(todo), t0)
    h5.close()


SHAPES = {
    "xlmr": {"fields": (4, 1024), "speech": (4, 1024), "speech_mask": (4,), "org": (1, 1024)},
    "videomae": {"clips": (4, 768)},
    "whisper": {"audio": (4, 1280), "audio_mask": (4,)},
    "bge": {"fields": (4, 1024), "speech": (4, 1024), "speech_mask": (4,), "desc": (1024,),
            "transcript": (1024,)},
    "siglip": {"clips": (4, 1152), "fields": (4, 1152), "video": (1152,)},
    "bge_text": {"fields": (4, 1024), "speech": (4, 1024), "speech_mask": (4,), "org": (1, 1024)},
    "e5": {"fields": (4, 1024), "speech": (4, 1024), "speech_mask": (4,), "org": (1, 1024)},
    "e5_small": {"fields": (4, 384), "speech": (4, 384), "speech_mask": (4,), "org": (1, 384)},
    "m2v": {"fields": (4, 256), "speech": (4, 256), "speech_mask": (4,), "org": (1, 256)},
    "e5_pca64": {"fields": (4, PCA_DIMS), "speech": (4, PCA_DIMS), "speech_mask": (4,), "org": (1, PCA_DIMS)},
    "videomae_pca64": {"clips": (4, PCA_DIMS)},
}


def check(inp: Inputs, out: Path) -> None:
    """Every stage file holds every video, with the right shapes and no NaN."""
    for st, shapes in SHAPES.items():
        p = out / f"{st}.h5"
        if not p.is_file():
            print(f"  {st:9s} missing")
            continue
        with h5py.File(p, "r") as h:
            ids = set(h.keys())
            want = set(inp.videos.index)
            bad, zero = [], 0
            for k in sorted(ids & want):
                for name, shape in shapes.items():
                    x = h[k][name][()]
                    if x.shape != shape or not np.isfinite(x).all():
                        bad.append(f"{k}/{name} {x.shape}")
                if "speech_mask" in shapes:
                    sm = h[k]["speech_mask"][()]
                    exp = [bool(s.strip()) for s in inp.moments[k]["speech"]]
                    if list(sm.astype(bool)) != exp:
                        bad.append(f"{k}/speech_mask")
                if "audio_mask" in shapes:
                    zero += int(h[k]["audio_mask"][()].sum() < N_WIN)
            print(f"  {st:9s} {len(ids & want)}/{len(want)} videos, {len(bad)} bad {bad[:3]}"
                  + (f", {zero} with a window without audio" if "audio_mask" in shapes else ""))
            require(not bad and want <= ids, f"{st}.h5 complete and valid",
                    expected=len(want), found=f"{len(ids & want)} present, {len(bad)} bad")
            if st == "siglip":
                # meaning, not just shape: a field must match its own video's frames
                # better than another video's (wrong text / image heads give ~0)
                ks = sorted(want)
                f = np.stack([h[k]["fields"][()].astype(np.float32) for k in ks]).mean(1)   # (n, d)
                c = np.stack([h[k]["clips"][()].astype(np.float32) for k in ks]).mean(1)
                s = f @ c.T                                                  # video i's fields vs video j's frames
                own = np.diag(s)
                off = ~np.eye(len(ks), dtype=bool)
                pct = ((s < own[:, None]) & off).sum(1) / off.sum(1)         # own frames beat this share of others
                print(f"  siglip    field-frame cosine: own video {own.mean():.4f}, other videos "
                      f"{s[off].mean():.4f}; own frames rank above {pct.mean():.0%} of other videos' (random: 50 %)")
                require(pct.mean() > 0.7, "SigLIP fields match their own frames best",
                        expected="own frames above > 70 % of other videos'", found=f"{pct.mean():.0%}")


def text_stage(name: str, inp: Inputs, out: Path, device: str) -> None:
    """Alternative text encoders for the node features, in xlmr.h5's layout (fields,
    speech, speech_mask, org), so graphs and probes can swap the encoder by name.
      bge_text  BGE-M3 dense: CLS of the last layer, L2-normalised (bge.h5 has
                no organisers' text)
      e5        multilingual-e5-large: mean over real tokens, L2-normalised, with
                the "query: " prefix its authors use for classification features
      e5_small  multilingual-e5-small, the same recipe (384 dimensions)
      m2v       Model2Vec potion-multilingual-128M: static token vectors averaged
                (no transformer at inference), L2-normalised, 256 dimensions"""
    if name == "m2v":
        from model2vec import StaticModel
        static = StaticModel.from_pretrained(MODELS[name])

        def enc(texts: list[str]) -> np.ndarray:
            v = np.asarray(static.encode(texts), np.float32)
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    else:
        import torch
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODELS[name])
        model = AutoModel.from_pretrained(MODELS[name], torch_dtype=torch.float16).to(device).eval()

        @torch.no_grad()
        def enc(texts: list[str]) -> np.ndarray:
            if name in ("e5", "e5_small"):
                texts = [f"query: {t}" for t in texts]
            b = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            h = model(**b).last_hidden_state
            if name == "bge_text":
                v = h[:, 0]
            else:
                m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            return torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()

    h5, done = open_out(out / f"{name}.h5")
    todo = [k for k in inp.videos.index if k not in done]
    t0 = time.time()
    for n, k in enumerate(todo, 1):
        m = inp.moments[k]
        mask = np.array([bool(s.strip()) for s in m["speech"]], dtype=np.uint8)
        e = enc(inp.fields(k) + [s for s in m["speech"] if s.strip()] + [m["org_text"] or " "])
        sp = np.zeros((N_WIN, e.shape[1]), np.float32)
        sp[mask.astype(bool)] = e[4:4 + mask.sum()]
        put(h5, k, {"fields": e[:4], "speech": sp, "speech_mask": mask, "org": e[-1:]})
        progress(n, len(todo), t0)
    h5.close()


def stage_pca(inp: Inputs, out: Path, device: str) -> None:
    """Label-free PCA to PCA_DIMS of e5's text vectors (one basis for fields, speech
    and organisers' text) and of VideoMAE's clips, fitted on every video's vectors
    (train and test; no labels are read). Rewrites both files whole."""
    from sklearn.decomposition import PCA
    ids = list(inp.videos.index)

    def load(stage, key):
        with h5py.File(out / f"{stage}.h5", "r") as h:
            miss = [k for k in ids if k not in h]
            require(not miss, f"{stage}.h5 has every video", expected=len(ids), found=f"{len(miss)} missing")
            return np.stack([h[k][key][()] for k in ids]).astype(np.float32)
    f, sp, o = load("e5", "fields"), load("e5", "speech"), load("e5", "org")
    m = load("e5", "speech_mask").astype(bool)
    pool = np.concatenate([f.reshape(-1, f.shape[-1]), sp[m], o.reshape(-1, o.shape[-1])])
    pt = PCA(PCA_DIMS, random_state=0).fit(pool)
    c = load("videomae", "clips")
    pc = PCA(PCA_DIMS, random_state=0).fit(c.reshape(-1, c.shape[-1]))
    print(f"  e5: {len(pool)} vectors, {PCA_DIMS} components keep {pt.explained_variance_ratio_.sum():.1%} "
          f"of the variance; videomae: {pc.explained_variance_ratio_.sum():.1%}", flush=True)

    def tr(p, x):
        return p.transform(x.reshape(-1, x.shape[-1])).reshape(*x.shape[:-1], PCA_DIMS)
    ft, st, ot, ct = tr(pt, f), tr(pt, sp) * m[..., None], tr(pt, o), tr(pc, c)
    for name, rows in (("e5_pca64", lambda i: {"fields": ft[i], "speech": st[i],
                                                "speech_mask": m[i].astype(np.uint8), "org": ot[i]}),
                       ("videomae_pca64", lambda i: {"clips": ct[i]})):
        path = out / f"{name}.h5"
        if path.exists():
            path.unlink()
        h5, _ = open_out(path)
        h5.attrs["pca"] = f"{PCA_DIMS} components, fitted on {len(ids)} videos (train and test, label-free)"
        for i, k in enumerate(ids):
            put(h5, k, rows(i))
        h5.close()
        print(f"  wrote {path}", flush=True)


RUN = {"xlmr": stage_xlmr, "videomae": stage_videomae, "whisper": stage_whisper,
       "bge": stage_bge, "siglip": stage_siglip,
       "bge_text": lambda *x: text_stage("bge_text", *x), "e5": lambda *x: text_stage("e5", *x),
       "e5_small": lambda *x: text_stage("e5_small", *x), "m2v": lambda *x: text_stage("m2v", *x),
       "pca64": stage_pca}


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", required=True, choices=STAGES + ["all", "check"])
    ap.add_argument("--out-dir", default=None, help="default: outputs/features")
    ap.add_argument("--limit", type=int, default=0, help="first N videos, half train / half test (smoke)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out_dir) if a.out_dir else cfg.path("out_dir") / "features"
    inp = Inputs(cfg, a.limit)
    if a.stage == "check":
        check(inp, out)
        return
    for st in (STAGES if a.stage == "all" else [a.stage]):
        print(f"== {st}: {MODELS[st]} -> {out / (st + '.h5')} ({len(inp.videos)} videos)", flush=True)
        t = time.time()
        RUN[st](inp, out, a.device)
        print(f"   done in {(time.time() - t) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
