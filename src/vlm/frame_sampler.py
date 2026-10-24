# Copied verbatim from vlm_provider/frame_sampler.py (sha256 515e1c222621ac63...), the code that
# produced the part-1 descriptions. Do not edit: changes here change the inputs.
# frame_sampler.py
from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import imagehash  # type: ignore
    _HAS_PHASH = True
except ImportError:
    _HAS_PHASH = False

try:
    from scenedetect import detect, ContentDetector  # type: ignore
    _HAS_SCENEDETECT = True
except ImportError:
    _HAS_SCENEDETECT = False


@dataclass
class SampledFrame:
    image: Image.Image
    timestamp: float
    source: str  # "scene" | "uniform"


def _resize_long_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _cv2_frame_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def _grab_frame_at(cap: cv2.VideoCapture, t_sec: float, fps: float) -> Image.Image | None:
    target_idx = max(int(round(t_sec * fps)), 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return _cv2_frame_to_pil(frame)


def _video_info(path: str) -> tuple[float, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    duration = n_frames / fps if fps > 0 else 0.0
    return fps, duration


def _scene_midpoints(video_path: str, threshold: float) -> list[float]:
    if not _HAS_SCENEDETECT:
        return []
    try:
        scenes = detect(video_path, ContentDetector(threshold=threshold))
    except Exception:
        return []
    return [
        (start.get_seconds() + end.get_seconds()) / 2
        for start, end in scenes
    ]


def sample_frames(
    video_path: str | Path,
    min_frames: int = 6,
    max_frames: int = 14,
    uniform_interval_sec: float = 2.0,
    use_scene_detect: bool = True,
    scene_detect_threshold: float = 27.0,
    dedupe_hamming_threshold: int = 4,
    max_frame_side: int = 720,
) -> list[SampledFrame]:
    """
    Content-aware sampling:
      1. (optional) scene midpoints via PySceneDetect
      2. uniform samples every `uniform_interval_sec`
      3. dedupe near-duplicates via perceptual hash
      4. clamp to [min_frames, max_frames]
    """
    video_path = str(video_path)
    fps, duration = _video_info(video_path)
    if duration <= 0:
        duration = 1.0

    candidate: list[tuple[float, str]] = []

    if use_scene_detect:
        for t in _scene_midpoints(video_path, scene_detect_threshold):
            candidate.append((t, "scene"))

    n_uniform = max(int(duration / uniform_interval_sec), 1)
    for i in range(n_uniform):
        t = (i + 0.5) * duration / n_uniform
        candidate.append((t, "uniform"))

    candidate.sort(key=lambda x: x[0])
    merged: list[tuple[float, str]] = []
    for t, src in candidate:
        if merged and abs(t - merged[-1][0]) < 0.4:
            continue
        merged.append((t, src))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: list[SampledFrame] = []
    try:
        for t, src in merged:
            img = _grab_frame_at(cap, t, fps)
            if img is None:
                continue
            img = _resize_long_side(img, max_frame_side)
            frames.append(SampledFrame(image=img, timestamp=t, source=src))
    finally:
        cap.release()

    # dedupe
    deduped: list[SampledFrame] = []
    if _HAS_PHASH and dedupe_hamming_threshold > 0:
        hashes = []
        for f in frames:
            h = imagehash.phash(f.image)
            if any(h - prev <= dedupe_hamming_threshold for prev in hashes):
                continue
            deduped.append(f)
            hashes.append(h)
    else:
        deduped = frames

    # clamp down
    if len(deduped) > max_frames:
        idx = np.linspace(0, len(deduped) - 1, max_frames).astype(int)
        deduped = [deduped[i] for i in idx]

    # top up
    if len(deduped) < min_frames:
        cap = cv2.VideoCapture(video_path)
        try:
            need = min_frames - len(deduped)
            extra_times = np.linspace(0, max(duration - 0.1, 0.1), need + 2)[1:-1]
            for t in extra_times:
                img = _grab_frame_at(cap, float(t), fps)
                if img is None:
                    continue
                img = _resize_long_side(img, max_frame_side)
                deduped.append(SampledFrame(img, float(t), "uniform"))
        finally:
            cap.release()
        deduped.sort(key=lambda f: f.timestamp)

    return deduped


def frame_to_base64_jpeg(img: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")