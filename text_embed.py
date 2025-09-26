from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

# ---- Configurable via entorno ----
TEXT_HF_REPO_ID = os.getenv("TEXT_HF_REPO_ID", "krizzcs2/text-paraphrase-minilm").strip()
TEXT_HF_REVISION = os.getenv("TEXT_HF_REVISION", "").strip() or None
TEXT_ONNX_FILENAME = os.getenv("TEXT_ONNX_FILENAME", "model.onnx").strip()
TEXT_TOKENIZER_FILE = os.getenv("TEXT_TOKENIZER_FILE", "tokenizer.json").strip()
TEXT_LOCAL_ONNX_PATH = os.getenv("TEXT_ONNX_PATH", "").strip() or None
TEXT_LOCAL_TOKENIZER_PATH = os.getenv("TEXT_TOKENIZER_PATH", "").strip() or None
TEXT_ONNX_PROVIDER = os.getenv("TEXT_ONNX_PROVIDER", "CPUExecutionProvider").strip()
TEXT_MAX_LENGTH = int(os.getenv("TEXT_MAX_LEN", "256"))

# ---- Estado global ----
_session: ort.InferenceSession | None = None
_tokenizer: Tokenizer | None = None
_input_names: set[str] | None = None


def _hf_download(filename: str) -> str:
    """Descarga un archivo del repo configurado en el hub."""
    return hf_hub_download(
        repo_id=TEXT_HF_REPO_ID,
        filename=filename,
        token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        repo_type="model",
        revision=TEXT_HF_REVISION,
        local_dir_use_symlinks=False,
    )


def _resolve_model_path() -> str:
    if TEXT_LOCAL_ONNX_PATH and Path(TEXT_LOCAL_ONNX_PATH).exists():
        return TEXT_LOCAL_ONNX_PATH
    return _hf_download(TEXT_ONNX_FILENAME)


def _resolve_tokenizer_path() -> str:
    if TEXT_LOCAL_TOKENIZER_PATH and Path(TEXT_LOCAL_TOKENIZER_PATH).exists():
        return TEXT_LOCAL_TOKENIZER_PATH
    return _hf_download(TEXT_TOKENIZER_FILE)


def _ensure_loaded() -> None:
    global _session, _tokenizer, _input_names
    if _session is not None and _tokenizer is not None:
        return

    model_path = _resolve_model_path()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    _session = ort.InferenceSession(model_path, providers=[TEXT_ONNX_PROVIDER], sess_options=so)
    _input_names = {inp.name for inp in _session.get_inputs()}

    tok_path = _resolve_tokenizer_path()
    tokenizer = Tokenizer.from_file(tok_path)
    tokenizer.enable_truncation(max_length=TEXT_MAX_LENGTH)
    pad_id = tokenizer.token_to_id("[PAD]") or tokenizer.token_to_id("<pad>") or 0
    tokenizer.enable_padding(length=TEXT_MAX_LENGTH, pad_id=pad_id, pad_token="[PAD]")
    _tokenizer = tokenizer


def _build_inputs(texts: List[str]) -> dict[str, np.ndarray]:
    assert _tokenizer is not None and _input_names is not None
    encodings = _tokenizer.encode_batch(texts)
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    feeds: dict[str, np.ndarray] = {}
    for name in _input_names:
        if name == "input_ids":
            feeds[name] = input_ids
        elif name == "attention_mask":
            feeds[name] = attention_mask
        elif name == "token_type_ids":
            type_ids = np.array([e.type_ids or [0] * len(e.ids) for e in encodings], dtype=np.int64)
            feeds[name] = type_ids
    return feeds


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    """Obtiene embeddings para una lista de textos."""
    _ensure_loaded()
    cleaned: List[str] = [(t or "").strip() for t in texts]
    if not cleaned:
        return np.zeros((0, 0), dtype=np.float32)
    feeds = _build_inputs(cleaned)
    assert _session is not None
    outputs = _session.run(None, feeds)
    return outputs[0]


def embed_one(text: str) -> np.ndarray:
    """Embedding de un solo texto (vector 1D)."""
    emb = embed_texts([text])
    return emb[0]


if __name__ == "__main__":
    vec = embed_one("Hola, probando embeddings")
    print(vec.shape)
