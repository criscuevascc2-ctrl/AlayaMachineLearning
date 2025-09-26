from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import requests
from huggingface_hub import snapshot_download
from moviepy.editor import VideoFileClip
from PIL import Image

import audio_embed
import vision_embed

VIDEO_HF_REPO_ID = os.getenv("VIDEO_HF_REPO_ID", "krizzcs2/video-av-lite").strip()
VIDEO_HF_REVISION = os.getenv("VIDEO_HF_REVISION", "").strip() or None
VIDEO_LOCAL_DIR = os.getenv("VIDEO_LOCAL_DIR", "").strip() or None
VIDEO_ONNX_FILENAME = os.getenv("VIDEO_ONNX_FILENAME", "model.onnx").strip()
VIDEO_ONNX_PATH = os.getenv("VIDEO_ONNX_PATH", "").strip() or None
VIDEO_ONNX_PROVIDER = os.getenv("VIDEO_ONNX_PROVIDER", "CPUExecutionProvider").strip()
VIDEO_FRAME_SAMPLES = int(os.getenv("VIDEO_FRAME_SAMPLES", "16"))

VISION_DIM = int(os.getenv("VIDEO_VISION_DIM", "768"))
AUDIO_DIM = int(os.getenv("VIDEO_AUDIO_DIM", "768"))
VECTOR_DIM = VISION_DIM + AUDIO_DIM

_session: ort.InferenceSession | None = None
_input_names: Sequence[str] | None = None
_model_dir: Path | None = None


def _token() -> str | None:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )


def _ensure_local_dir() -> Path:
    global _model_dir
    if _model_dir is not None:
        return _model_dir
    if VIDEO_LOCAL_DIR:
        local_dir = Path(VIDEO_LOCAL_DIR)
        if local_dir.exists():
            _model_dir = local_dir
            return _model_dir
    local_path = snapshot_download(
        repo_id=VIDEO_HF_REPO_ID,
        repo_type="model",
        revision=VIDEO_HF_REVISION,
        token=_token(),
        allow_patterns=["*.onnx"],
    )
    _model_dir = Path(local_path)
    return _model_dir


def _resolve_model_path() -> Path:
    if VIDEO_ONNX_PATH:
        path = Path(VIDEO_ONNX_PATH)
        if path.exists():
            return path
    model_dir = _ensure_local_dir()
    return model_dir / VIDEO_ONNX_FILENAME


def _ensure_loaded() -> None:
    global _session, _input_names
    if _session is not None:
        return
    model_path = _resolve_model_path()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    _session = ort.InferenceSession(str(model_path), providers=[VIDEO_ONNX_PROVIDER], sess_options=so)
    _input_names = [inp.name for inp in _session.get_inputs()]


def _download_remote(url: str) -> Tuple[Path, bool]:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    suffix = Path(url).suffix or ".mp4"
    handle, temp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(handle, "wb") as fh:
        fh.write(response.content)
    return Path(temp_path), True


def _ensure_local_video(item: str | Path) -> Tuple[Path, bool]:
    path = Path(str(item))
    if path.exists():
        return path, False
    value = str(item)
    if value.startswith("http://") or value.startswith("https://"):
        return _download_remote(value)
    raise FileNotFoundError(f"Video source not found: {item}")


def _extract_frames_and_audio(path: Path) -> Tuple[List[Image.Image], np.ndarray | None, int]:
    frames: List[Image.Image] = []
    waveform: np.ndarray | None = None
    sample_rate = audio_embed.AUDIO_SAMPLE_RATE

    clip = VideoFileClip(str(path))
    try:
        duration = float(clip.duration or 0.0)
        count = max(VIDEO_FRAME_SAMPLES, 1)
        if duration <= 0:
            iterator = clip.iter_frames(dtype="uint8")
            for idx, frame in enumerate(iterator):
                frames.append(Image.fromarray(frame).convert("RGB"))
                if idx + 1 >= count:
                    break
        else:
            times = np.linspace(0.0, max(duration - 1e-3, 0.0), num=count)
            for t in times:
                frame = clip.get_frame(float(t))
                frames.append(Image.fromarray(frame).convert("RGB"))
        if clip.audio is not None:
            audio_array = clip.audio.to_soundarray(fps=sample_rate)
            if audio_array.size:
                if audio_array.ndim == 2:
                    audio_array = audio_array.mean(axis=1)
                waveform = audio_array.astype(np.float32)
    finally:
        try:
            clip.close()
        except Exception:
            pass
    return frames, waveform, sample_rate


def _aggregate(vision_vec: np.ndarray, audio_vec: np.ndarray) -> np.ndarray:
    _ensure_loaded()
    assert _session is not None and _input_names is not None
    feeds = {}
    vision_arr = np.asarray(vision_vec, dtype=np.float32).reshape(1, -1)
    audio_arr = np.asarray(audio_vec, dtype=np.float32).reshape(1, -1)
    for name in _input_names:
        if name == "vision":
            feeds[name] = vision_arr
        elif name == "audio":
            feeds[name] = audio_arr
    outputs = _session.run(None, feeds)
    vector = outputs[0][0]
    norm = np.linalg.norm(vector)
    if np.isfinite(norm) and norm > 0:
        vector = vector / norm
    return vector


def _embed_single(item: str | Path) -> np.ndarray:
    try:
        path, is_temp = _ensure_local_video(item)
    except Exception:
        return np.zeros(VECTOR_DIM, dtype=np.float32)
    try:
        frames, waveform, sample_rate = _extract_frames_and_audio(path)
        if not frames:
            return np.zeros(VECTOR_DIM, dtype=np.float32)
        vision_batch = vision_embed.embed_images(frames)
        if vision_batch.size == 0:
            return np.zeros(VECTOR_DIM, dtype=np.float32)
        vision_vec = vision_batch.mean(axis=0)
        if waveform is not None and waveform.size > 0:
            audio_vec = audio_embed.embed_audio([waveform], sample_rate=sample_rate)[0]
        else:
            audio_vec = np.zeros(AUDIO_DIM, dtype=np.float32)
        return _aggregate(vision_vec, audio_vec)
    except Exception:
        return np.zeros(VECTOR_DIM, dtype=np.float32)
    finally:
        if 'is_temp' in locals() and is_temp:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


def embed_videos(items: Iterable[str | Path]) -> np.ndarray:
    vectors = [np.asarray(_embed_single(item), dtype=np.float32) for item in items]
    if not vectors:
        return np.zeros((0, VECTOR_DIM), dtype=np.float32)
    return np.vstack(vectors)


def embed_video(item: str | Path) -> np.ndarray:
    return embed_videos([item])[0]


if __name__ == "__main__":
    sample = os.getenv("VIDEO_SAMPLE_PATH")
    if sample:
        vec = embed_video(sample)
        print(vec.shape)
    else:
        print("Set VIDEO_SAMPLE_PATH to test video embedding.")
