from pathlib import Path
from online_media_model import bootstrap_from_jsonl, DEFAULT_HORIZON_CONFIG

DATA_PATH = Path("_debug/label_exports/ig_media_enriched.jsonl")
CHECKPOINT_PATH = Path("checkpoints")
CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

embedder_paths = {
    "text_onnx": Path("models/text-paraphrase-minilm/model.onnx"),
    "tokenizer_json": Path("models/text-paraphrase-minilm/tokenizer.json"),
    "vision_dir": Path("models/vision-vit-base-patch16-224-fe"),
    "video_dir": Path("models/video-av-lite"),
    "video_onnx": Path("models/video-av-lite/model.onnx"),
    "video_repo": "krizzcs2/video-av-lite",
    "audio_dir": Path("models/audio-wav2vec2-base-960h"),
}

model = bootstrap_from_jsonl(
    stream_path=DATA_PATH,
    embedder_paths=embedder_paths,
    horizon_config=DEFAULT_HORIZON_CONFIG,
)

print("MAE por objetivo y horizonte:")
for target, scores in model.metrics().items():
    print(target, scores)

print("Entrenamiento inicial completado. Guarda el objeto 'model' con la estrategia que prefieras (pickle, dill, etc.) si necesitas persistencia.")
