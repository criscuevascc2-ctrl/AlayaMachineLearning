from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Tuple

from online_media_model import (
    DEFAULT_HORIZON_CONFIG,
    bootstrap_from_jsonl,
    normalize_offset,
    offset_to_hours,
)

EMBEDDER_PATHS = {
    "text_onnx": Path("models/text-paraphrase-minilm/model.onnx"),
    "tokenizer_json": Path("models/text-paraphrase-minilm/tokenizer.json"),
    "vision_dir": Path("models/vision-vit-base-patch16-224-fe"),
    "audio_dir": Path("models/audio-wav2vec2-base-960h"),
}

TRAINING_DATA = Path("_debug/label_exports/ig_media_enriched.jsonl")


def parse_horizon(media_type: str, label: str) -> Tuple[int | None, float | None]:
    media_type = (media_type or "").upper() or "FEED"
    offsets = DEFAULT_HORIZON_CONFIG.get(media_type) or DEFAULT_HORIZON_CONFIG["FEED"]
    for idx, offset in enumerate(offsets):
        if normalize_offset(offset) == label:
            return idx, offset_to_hours(offset)
    # fallback: parse suffix
    value = label.strip().lower()
    try:
        if value.endswith("h"):
            return None, float(value[:-1])
        if value.endswith("d"):
            return None, float(value[:-1]) * 24.0
    except ValueError:
        pass
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI para probar predicciones del modelo online")
    parser.add_argument("sample", type=Path, help="Ruta al JSON con el snapshot a evaluar")
    parser.add_argument(
        "--horizon",
        required=True,
        help="Horizonte a evaluar (ej. 12h, 3d, 7d). Debe existir en DEFAULT_HORIZON_CONFIG para el tipo de media",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=TRAINING_DATA,
        help="JSONL usado para warm-up (por defecto: _debug/label_exports/ig_media_enriched.jsonl)",
    )
    args = parser.parse_args()

    if not args.sample.exists():
        raise SystemExit(f"No se encuentra el archivo de entrada: {args.sample}")
    if not args.train_data.exists():
        raise SystemExit(f"No se encuentra el dataset de entrenamiento: {args.train_data}")

    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    media_type = sample.get("media_product_type", "FEED")

    print("Entrenando modelo online con datos hist?ricos...")
    model = bootstrap_from_jsonl(
        stream_path=args.train_data,
        embedder_paths=EMBEDDER_PATHS,
        horizon_config=DEFAULT_HORIZON_CONFIG,
    )

    idx, hours = parse_horizon(media_type, args.horizon)
    pred = model.predict_for_horizon(
        sample,
        horizon_label=args.horizon,
        horizon_idx=idx,
        horizon_hours=hours,
        elapsed_hours=None,
    )
    print("\nPredicci?n para", args.horizon)
    print(json.dumps(pred, indent=2, ensure_ascii=False))

    print("\nPredicciones para todos los horizontes disponibles:")
    all_preds = model.predict_all_horizons(sample, media_type)
    print(json.dumps(all_preds, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
