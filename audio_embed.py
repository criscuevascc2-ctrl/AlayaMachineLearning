from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import Wav2Vec2FeatureExtractor

AUDIO_HF_REPO_ID = os.getenv("AUDIO_HF_REPO_ID", "krizzcs2/audio-wav2vec2-base-960h").strip()
AUDIO_HF_REVISION = os.getenv("AUDIO_HF_REVISION", "").strip() or None
AUDIO_LOCAL_DIR = os.getenv("AUDIO_LOCAL_DIR", "").strip() or None
AUDIO_ONNX_FILENAME = os.getenv("AUDIO_ONNX_FILENAME", "model.onnx").strip()
AUDIO_ONNX_PATH = os.getenv("AUDIO_ONNX_PATH", "").strip() or None
AUDIO_ONNX_PROVIDER = os.getenv("AUDIO_ONNX_PROVIDER", "CPUExecutionProvider").strip()
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))

_session: ort.InferenceSession | None = None
_feature_extractor: Wav2Vec2FeatureExtractor | None = None
_input_names: Sequence[str] | None = None
_model_dir: Path | None = None


def _token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _ensure_local_dir() -> Path:
    global _model_dir
    if _model_dir is not None:
        return _model_dir
    if AUDIO_LOCAL_DIR and Path(AUDIO_LOCAL_DIR).exists():
        _model_dir = Path(AUDIO_LOCAL_DIR)
        return _model_dir
    local_path = snapshot_download(
        repo_id=AUDIO_HF_REPO_ID,
        repo_type="model",
        revision=AUDIO_HF_REVISION,
        token=_token(),
        allow_patterns=["*.onnx", "*.json"],
    )
    _model_dir = Path(local_path)
    return _model_dir


def _resolve_model_path() -> Path:
    if AUDIO_ONNX_PATH and Path(AUDIO_ONNX_PATH).exists():
        return Path(AUDIO_ONNX_PATH)
    model_dir = _ensure_local_dir()
    return model_dir / AUDIO_ONNX_FILENAME


def _ensure_loaded() -> None:
    global _session, _feature_extractor, _input_names
    if _session is not None and _feature_extractor is not None:
        return

    model_path = _resolve_model_path()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    _session = ort.InferenceSession(str(model_path), providers=[AUDIO_ONNX_PROVIDER], sess_options=so)
    _input_names = [inp.name for inp in _session.get_inputs()]

    model_dir = _ensure_local_dir()
    _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)


def _prepare_waveform(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    wf = np.asarray(waveform, dtype=np.float32)
    if wf.ndim == 2:
        wf = wf.mean(axis=0)
    if sample_rate != AUDIO_SAMPLE_RATE and sample_rate > 0:
        try:
            import librosa

            wf = librosa.resample(wf, orig_sr=sample_rate, target_sr=AUDIO_SAMPLE_RATE)
        except Exception:
            # fallback: simple interpolation
            target_len = int(round(len(wf) * AUDIO_SAMPLE_RATE / sample_rate))
            wf = np.interp(np.linspace(0, len(wf), target_len, endpoint=False), np.arange(len(wf)), wf)
    return wf


def _preprocess(batch: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray | None]:
    assert _feature_extractor is not None
    processed = _feature_extractor(batch, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="np")
    input_values = processed["input_values"].astype(np.float32)
    attention_mask = processed.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.astype(np.int64)
    return input_values, attention_mask


def embed_audio(waveforms: Iterable[np.ndarray], sample_rate: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    _ensure_loaded()
    batch = [_prepare_waveform(wf, sample_rate) for wf in waveforms]
    if not batch:
        return np.zeros((0, 0), dtype=np.float32)
    input_values, attention_mask = _preprocess(batch)
    feeds = {}
    assert _session is not None and _input_names is not None
    for name in _input_names:
        if name == "input_values":
            feeds[name] = input_values
        elif name == "attention_mask" and attention_mask is not None:
            feeds[name] = attention_mask
    outputs = _session.run(None, feeds)
    hidden_states = outputs[0]
    if hidden_states.ndim == 3:
        mask = None
        if attention_mask is not None:
            mask = attention_mask.astype(bool)
        if mask is not None:
            lengths = mask.sum(axis=1, keepdims=True).clip(min=1)
            summed = (hidden_states * mask[:, :, None]).sum(axis=1)
            pooled = summed / lengths
        else:
            pooled = hidden_states.mean(axis=1)
        return pooled
    return hidden_states


def embed_one(waveform: np.ndarray, sample_rate: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    return embed_audio([waveform], sample_rate=sample_rate)[0]


if __name__ == "__main__":
    dummy = np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)
    vec = embed_one(dummy)
    print(vec.shape)
