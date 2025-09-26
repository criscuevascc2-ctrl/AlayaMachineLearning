from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoImageProcessor

VISION_HF_REPO_ID = os.getenv("VISION_HF_REPO_ID", "krizzcs2/vision-vit-base-patch16-224-fe").strip()
VISION_HF_REVISION = os.getenv("VISION_HF_REVISION", "").strip() or None
VISION_LOCAL_DIR = os.getenv("VISION_LOCAL_DIR", "").strip() or None
VISION_ONNX_FILENAME = os.getenv("VISION_ONNX_FILENAME", "model.onnx").strip()
VISION_ONNX_PATH = os.getenv("VISION_ONNX_PATH", "").strip() or None
VISION_ONNX_PROVIDER = os.getenv("VISION_ONNX_PROVIDER", "CPUExecutionProvider").strip()
VISION_IMAGE_SIZE = int(os.getenv("VISION_IMAGE_SIZE", "224"))

_session: ort.InferenceSession | None = None
_processor: AutoImageProcessor | None = None
_input_names: Sequence[str] | None = None
_model_dir: Path | None = None


def _token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _ensure_local_dir() -> Path:
    global _model_dir
    if _model_dir is not None:
        return _model_dir
    if VISION_LOCAL_DIR and Path(VISION_LOCAL_DIR).exists():
        _model_dir = Path(VISION_LOCAL_DIR)
        return _model_dir
    local_path = snapshot_download(
        repo_id=VISION_HF_REPO_ID,
        repo_type="model",
        revision=VISION_HF_REVISION,
        token=_token(),
        allow_patterns=["*.onnx", "*.json"],
    )
    _model_dir = Path(local_path)
    return _model_dir


def _resolve_model_path() -> Path:
    if VISION_ONNX_PATH and Path(VISION_ONNX_PATH).exists():
        return Path(VISION_ONNX_PATH)
    model_dir = _ensure_local_dir()
    return model_dir / VISION_ONNX_FILENAME


def _ensure_loaded() -> None:
    global _session, _processor, _input_names
    if _session is not None and _processor is not None:
        return

    model_path = _resolve_model_path()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    _session = ort.InferenceSession(str(model_path), providers=[VISION_ONNX_PROVIDER], sess_options=so)
    _input_names = [inp.name for inp in _session.get_inputs()]

    model_dir = _ensure_local_dir()
    _processor = AutoImageProcessor.from_pretrained(model_dir)


def _load_image(item: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, np.ndarray):
        if item.ndim == 2:
            return Image.fromarray(item).convert("RGB")
        return Image.fromarray(item.astype(np.uint8)).convert("RGB")
    path = Path(item)
    return Image.open(path).convert("RGB")


def _preprocess(images: List[Image.Image]) -> np.ndarray:
    assert _processor is not None
    processed = _processor(images, return_tensors="np", size=VISION_IMAGE_SIZE)
    return processed["pixel_values"].astype(np.float32)


def embed_images(items: Iterable[str | Path | Image.Image | np.ndarray]) -> np.ndarray:
    _ensure_loaded()
    imgs = [_load_image(it) for it in items]
    if not imgs:
        return np.zeros((0, 0), dtype=np.float32)
    pixel_values = _preprocess(imgs)
    feeds = {}
    assert _session is not None and _input_names is not None
    for name in _input_names:
        feeds[name] = pixel_values
    outputs = _session.run(None, feeds)
    hidden_states = outputs[0]
    if hidden_states.ndim == 3:
        cls_tokens = hidden_states[:, 0, :]
    else:
        cls_tokens = hidden_states
    return cls_tokens


def embed_image(item: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    return embed_images([item])[0]


if __name__ == "__main__":
    dummy = Image.new("RGB", (VISION_IMAGE_SIZE, VISION_IMAGE_SIZE), color=(255, 0, 0))
    vec = embed_image(dummy)
    print(vec.shape)
