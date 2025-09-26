from __future__ import annotations

"""Online learning pipeline for Instagram media metrics with multimodal features."""

import io
import json
import math
import os
import pathlib
import typing as tp

import numpy as np
import requests
from PIL import Image
from river import base, compose, linear_model, metrics, preprocessing, stream

try:
    import AlayaMachineLearning.text_embed as text_embed_module
except ImportError:  # pragma: no cover
    text_embed_module = None

try:
    import vision_embed
except ImportError:  # pragma: no cover
    vision_embed = None

try:
    import audio_embed
except ImportError:  # pragma: no cover
    audio_embed = None

try:
    import video_embed
except ImportError:  # pragma: no cover
    video_embed = None

TARGETS = [
    "reach",
    "likes",
    "comments",
    "saved",
    "shares",
    "ig_reels_avg_watch_time_ms",
]

NUMERIC_FEATURES = [
    "media_width",
    "media_height",
    "video_duration_sec",
    "followers_count",
    "follows_count",
    "media_count",
    "reach_today",
    "impressions_today",
    "profile_views_today",
    "stories_active_count",
    "followers_delta_20m",
    "plays",
    "target_horizon_idx",
    "target_horizon_hours",
    "snapshot_elapsed_hours",
]

CATEGORICAL_FEATURES = [
    "media_type",
    "media_product_type",
    "is_collab_post",
    "is_paid_partnership",
    "is_comment_enabled",
    "is_active",
    "live_active",
    "target_horizon",
]

DEFAULT_HORIZON_CONFIG = {
    "FEED": ("01:00:00", "03:00:00", "06:00:00", "12:00:00", "1 day", "3 days", "7 days", "28 days"),
    "REELS": ("01:00:00", "03:00:00", "06:00:00", "12:00:00", "1 day", "3 days", "7 days", "28 days"),
    "STORIES": tuple(f"{i:02d}:00:00" for i in range(1, 24)),
}

ImageSource = tp.Union[str, pathlib.Path]


def _is_nan(value: tp.Any) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def parse_datetime(value: tp.Any) -> np.datetime64 | None:
    if value in (None, "", "NaT"):
        return None
    try:
        return np.datetime64(value)
    except (TypeError, ValueError):
        return None


def offset_to_hours(offset: str | None) -> float | None:
    if offset is None:
        return None
    raw = str(offset).strip()
    if not raw:
        return None
    low = raw.lower()
    try:
        if "day" in low:
            return float(low.split()[0]) * 24.0
        if "hour" in low:
            return float(low.split()[0])
        if "minute" in low:
            return float(low.split()[0]) / 60.0
        if ":" in raw:
            hh, mm, ss = raw.split(":")
            return int(hh) + int(mm) / 60.0 + int(ss) / 3600.0
    except (TypeError, ValueError):
        return None
    return None


def normalize_offset(offset: str | None) -> str:
    hours = offset_to_hours(offset)
    if hours is None:
        return (str(offset) if offset is not None else "unknown").replace(" ", "_")
    if hours >= 24 and math.isclose(hours % 24, 0.0, abs_tol=1e-6):
        return f"{int(round(hours / 24))}d"
    if hours >= 1 and math.isclose(hours, round(hours), abs_tol=1e-6):
        return f"{int(round(hours))}h"
    minutes = int(round(hours * 60))
    if minutes >= 1:
        return f"{minutes}m"
    seconds = max(int(round(hours * 3600)), 1)
    return f"{seconds}s"


def compute_elapsed_hours(record: dict) -> float | None:
    posted = parse_datetime(record.get("timestamp_utc"))
    snapshot = parse_datetime(
        record.get("taken_at") or record.get("last_snapshot_at") or record.get("detected_at")
    )
    if posted is None or snapshot is None:
        return None
    delta = (snapshot - posted) / np.timedelta64(1, "h")
    try:
        value = float(delta)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return max(value, 0.0)


def infer_horizon_info(
    record: dict,
    horizon_config: dict[str, tp.Sequence[str]] | None = None,
) -> tuple[str, int, float | None] | None:
    horizon_config = horizon_config or DEFAULT_HORIZON_CONFIG
    offsets = record.get("schedule_offsets") or ()
    if offsets:
        raw_idx = record.get("next_due_idx")
        try:
            idx_val = int(raw_idx)
        except (TypeError, ValueError):
            idx_val = None
        if idx_val is None:
            idx = len(offsets) - 1
        elif idx_val <= 0:
            idx = 0
        elif idx_val >= len(offsets):
            idx = len(offsets) - 1
        else:
            idx = idx_val - 1
        offset_value = offsets[idx]
        hours = offset_to_hours(offset_value)
        if hours is None:
            product = (record.get("media_product_type") or "").upper()
            template = horizon_config.get(product) or horizon_config.get("FEED", ())
            if idx < len(template):
                hours = offset_to_hours(template[idx])
        return normalize_offset(offset_value), idx, hours
    product = (record.get("media_product_type") or "").upper()
    template = horizon_config.get(product) or horizon_config.get("FEED", ())
    elapsed = compute_elapsed_hours(record)
    if not template and elapsed is None:
        return None
    if template:
        hours_list = [offset_to_hours(x) for x in template]
        candidates = []
        if elapsed is not None:
            for i, hours in enumerate(hours_list):
                if hours is None:
                    continue
                candidates.append((abs(elapsed - hours), i))
        idx = min(candidates, key=lambda pair: pair[0])[1] if candidates else 0
        chosen = template[idx] if idx < len(template) else None
        hours = hours_list[idx] if idx < len(hours_list) else elapsed
        return normalize_offset(chosen), idx, hours
    if elapsed is None:
        return None
    return normalize_offset(None), 0, elapsed


def extract_targets(record: dict, names: tp.Iterable[str] = TARGETS) -> dict[str, float]:
    targets: dict[str, float] = {}
    for name in names:
        value = record.get(name)
        if value is None or _is_nan(value):
            continue
        try:
            targets[name] = float(value)
        except (TypeError, ValueError):
            continue
    return targets


def prepare_features_for_model(
    record: dict,
    horizon_label: str,
    horizon_idx: int,
    horizon_hours: float | None,
    elapsed_hours: float | None,
) -> dict:
    features = dict(record)
    features["target_horizon"] = horizon_label
    features["target_horizon_idx"] = int(horizon_idx)
    if horizon_hours is not None:
        features["target_horizon_hours"] = horizon_hours
    if elapsed_hours is not None:
        features["snapshot_elapsed_hours"] = elapsed_hours
    else:
        features.setdefault("snapshot_elapsed_hours", horizon_hours)
    return features


def _text_from_record(record: dict) -> str:
    caption = record.get("caption") or ""
    hashtags = " ".join(record.get("hashtags", []) or [])
    mentions = " ".join(record.get("mentions", []) or [])
    return " ".join(filter(None, [caption, hashtags, mentions])).strip()


def _vector_to_features(vector: np.ndarray, prefix: str) -> dict[str, float]:
    flat = np.asarray(vector).reshape(-1)
    return {f"{prefix}_{i}": float(value) for i, value in enumerate(flat)}

def _is_video_record(record: dict) -> bool:
    media_type = (record.get("media_type") or "").upper()
    if "VIDEO" in media_type:
        return True
    product = (record.get("media_product_type") or "").upper()
    return product in {"VIDEO", "REELS"}


def _load_image_from_record(record: dict) -> Image.Image | None:
    source: ImageSource | None = record.get("storage_path") or record.get("media_url")
    if not source:
        return None
    path = pathlib.Path(str(source))
    try:
        if path.exists():
            return Image.open(path).convert("RGB")
    except OSError:
        pass
    try:
        response = requests.get(str(source), timeout=5)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


def _load_audio_from_record(record: dict) -> tuple[np.ndarray | None, int | None]:
    waveform = record.get("audio_waveform")
    if waveform is not None:
        try:
            arr = np.asarray(waveform, dtype=np.float32)
        except Exception:
            arr = None
        if arr is not None:
            sample_rate = int(record.get("audio_sample_rate") or 16000)
            return arr, sample_rate
    audio_path = record.get("audio_path") or record.get("audio_storage_path")
    if audio_path:
        path = pathlib.Path(str(audio_path))
        if path.exists():
            try:
                import soundfile as sf  # type: ignore
                data, sample_rate = sf.read(path)
                if data.ndim == 2:
                    data = data.mean(axis=1)
                return data.astype(np.float32), int(sample_rate)
            except Exception:
                return None, None
    return None, None


class TimestampFeatures(base.Transformer):
    """Adds cyclical encodings for time of day and day of week."""

    def transform_one(self, x: dict) -> dict:
        ts = x.get("timestamp_utc")
        if ts is None:
            return {}
        if isinstance(ts, str):
            ts = np.datetime64(ts)
        timestamp = np.datetime64(ts).astype("datetime64[s]").astype("int64")
        day = 24 * 3600
        week = 7 * day
        return {
            "ts_hour_sin": math.sin(2 * math.pi * (timestamp % day) / day),
            "ts_hour_cos": math.cos(2 * math.pi * (timestamp % day) / day),
            "ts_week_sin": math.sin(2 * math.pi * (timestamp % week) / week),
            "ts_week_cos": math.cos(2 * math.pi * (timestamp % week) / week),
        }

    def learn_one(self, x: dict, y: dict | None = None) -> TimestampFeatures:
        return self


class TextEmbeddingTransformer(base.Transformer):
    def __init__(self, prefix: str = "text_emb") -> None:
        self.prefix = prefix
        self.available = text_embed_module is not None

    def transform_one(self, x: dict) -> dict:
        if not self.available or text_embed_module is None:
            return {}
        text = _text_from_record(x)
        if not text:
            return {}
        try:
            vector = text_embed_module.embed_texts([text])[0]
        except Exception:
            return {}
        return _vector_to_features(vector, self.prefix)

    def learn_one(self, x: dict, y: dict | None = None) -> TextEmbeddingTransformer:
        return self


class VisionEmbeddingTransformer(base.Transformer):
    def __init__(self, prefix: str = "vision_emb") -> None:
        self.prefix = prefix
        self.available = vision_embed is not None

    def transform_one(self, x: dict) -> dict:
        if not self.available or vision_embed is None:
            return {}
        image = _load_image_from_record(x)
        if image is None:
            return {}
        try:
            vector = vision_embed.embed_image(image)
        except Exception:
            return {}
        return _vector_to_features(vector, self.prefix)

    def learn_one(self, x: dict, y: dict | None = None) -> VisionEmbeddingTransformer:
        return self


class VideoEmbeddingTransformer(base.Transformer):
    def __init__(self, prefix: str = "video_emb") -> None:
        self.prefix = prefix
        self.available = video_embed is not None

    def transform_one(self, x: dict) -> dict:
        if not self.available or video_embed is None:
            return {}
        if not _is_video_record(x):
            return {}
        source = x.get("storage_path") or x.get("media_url")
        if not source:
            return {}
        try:
            vector = video_embed.embed_video(source)
        except Exception:
            return {}
        return _vector_to_features(vector, self.prefix)

    def learn_one(self, x: dict, y: dict | None = None) -> VideoEmbeddingTransformer:
        return self


class AudioEmbeddingTransformer(base.Transformer):
    def __init__(self, prefix: str = "audio_emb") -> None:
        self.prefix = prefix
        self.available = audio_embed is not None

    def transform_one(self, x: dict) -> dict:
        if not self.available or audio_embed is None:
            return {}
        waveform, sample_rate = _load_audio_from_record(x)
        if waveform is None or sample_rate is None:
            return {}
        try:
            vector = audio_embed.embed_one(waveform, sample_rate=sample_rate)
        except Exception:
            return {}
        return _vector_to_features(vector, self.prefix)

    def learn_one(self, x: dict, y: dict | None = None) -> AudioEmbeddingTransformer:
        return self


def build_feature_builder() -> compose.Transformer:
    union_parts: list[tuple[str, base.Transformer]] = [
        (
            "categorical",
            compose.Select(*CATEGORICAL_FEATURES)
            | preprocessing.OrdinalEncoder()
            | preprocessing.OneHotEncoder(),
        ),
        (
            "numeric",
            compose.Select(*NUMERIC_FEATURES)
            | preprocessing.StatImputer()
            | preprocessing.StandardScaler(),
        ),
        ("timestamp", TimestampFeatures()),
    ]
    text_transformer = TextEmbeddingTransformer()
    if text_transformer.available:
        union_parts.append(("text", text_transformer))
    vision_transformer = VisionEmbeddingTransformer()
    if vision_transformer.available:
        union_parts.append(("vision", vision_transformer))
    video_transformer = VideoEmbeddingTransformer()
    if video_transformer.available:
        union_parts.append(("video", video_transformer))
    audio_transformer = AudioEmbeddingTransformer()
    if audio_transformer.available:
        union_parts.append(("audio", audio_transformer))
    return compose.TransformerUnion(*union_parts)


class OnlineMediaRegressor:
    def __init__(
        self,
        feature_builder: compose.Transformer,
        targets: list[str] = TARGETS,
        base_regressor: linear_model.LinearRegression | None = None,
        horizon_feature: str = "target_horizon",
    ) -> None:
        self.feature_builder = feature_builder
        self.targets = targets
        self.horizon_feature = horizon_feature
        base_regressor = base_regressor or linear_model.PARegressor(mode=1, C=0.5)
        self.models = {t: preprocessing.StandardScaler() | base_regressor.clone() for t in targets}
        self.metric_tracker: dict[str, dict[str, metrics.MAE]] = {}

    def _sanitize_input(self, x: dict) -> dict:
        cleaned: dict = {}
        for key, value in x.items():
            if value is None or _is_nan(value):
                continue
            if isinstance(value, list):
                cleaned[key] = [item for item in value if not _is_nan(item)]
                continue
            cleaned[key] = value
        for feature in NUMERIC_FEATURES:
            cleaned.setdefault(feature, 0.0)
        for feature in CATEGORICAL_FEATURES:
            cleaned.setdefault(feature, 'missing')
        return cleaned

    def _featurize(self, x: dict, update: bool = True) -> tuple[dict, dict]:
        cleaned = self._sanitize_input(x)
        features = self.feature_builder.transform_one(cleaned)
        if update:
            self.feature_builder.learn_one(cleaned)
        return features, cleaned

    def predict_one(self, x: dict) -> dict[str, float]:
        features, _ = self._featurize(x, update=False)
        return {t: self.models[t].predict_one(features) for t in self.targets}

    def learn_one(self, x: dict, y: dict[str, float]) -> dict[str, float]:
        features, cleaned = self._featurize(x, update=True)
        horizon_label = str(cleaned.get(self.horizon_feature, "unknown"))
        preds: dict[str, float] = {}
        for target in self.targets:
            if target not in y:
                continue
            pred = self.models[target].predict_one(features)
            self.models[target].learn_one(features, y[target])
            horizon_metrics = self.metric_tracker.setdefault(target, {})
            metric = horizon_metrics.setdefault(horizon_label, metrics.MAE())
            metric.update(y[target], pred)
            overall = horizon_metrics.setdefault("__overall__", metrics.MAE())
            overall.update(y[target], pred)
            preds[target] = pred
        return preds

    def metrics(self) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for target, horizon_metrics in self.metric_tracker.items():
            summary[target] = {name: float(metric.get()) for name, metric in horizon_metrics.items()}
        return summary

    def predict_for_horizon(
        self,
        x: dict,
        horizon_label: str,
        horizon_idx: int | None = None,
        horizon_hours: float | None = None,
        elapsed_hours: float | None = None,
    ) -> dict[str, float]:
        sample = dict(x)
        sample[self.horizon_feature] = horizon_label
        if horizon_idx is not None:
            sample["target_horizon_idx"] = int(horizon_idx)
        if horizon_hours is not None:
            sample["target_horizon_hours"] = horizon_hours
            sample.setdefault("snapshot_elapsed_hours", horizon_hours)
        if elapsed_hours is not None:
            sample["snapshot_elapsed_hours"] = elapsed_hours
        return self.predict_one(sample)

    def predict_all_horizons(
        self,
        x: dict,
        media_product_type: str,
        horizon_config: dict[str, tp.Sequence[str]] | None = None,
    ) -> dict[str, dict[str, float]]:
        horizon_config = horizon_config or DEFAULT_HORIZON_CONFIG
        product = (media_product_type or "").upper()
        offsets = horizon_config.get(product) or horizon_config.get("FEED", ())
        outputs: dict[str, dict[str, float]] = {}
        for idx, offset in enumerate(offsets):
            label = normalize_offset(offset)
            hours = offset_to_hours(offset)
            outputs[label] = self.predict_for_horizon(
                x,
                horizon_label=label,
                horizon_idx=idx,
                horizon_hours=hours,
                elapsed_hours=hours,
            )
        return outputs


def iter_training_events(
    stream_path: pathlib.Path | str,
    horizon_config: dict[str, tp.Sequence[str]] | None = None,
):
    horizon_config = horizon_config or DEFAULT_HORIZON_CONFIG
    path_obj = pathlib.Path(stream_path)
    with path_obj.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            horizon_info = infer_horizon_info(event, horizon_config=horizon_config)
            if horizon_info is None:
                continue
            horizon_label, horizon_idx, horizon_hours = horizon_info
            targets = extract_targets(event)
            if not targets:
                continue
            elapsed = compute_elapsed_hours(event)
            features = prepare_features_for_model(
                event, horizon_label, horizon_idx, horizon_hours, elapsed
            )
            yield features, targets


def _configure_embedding_environment(embedder_paths: dict[str, tp.Any] | None) -> None:
    if not embedder_paths:
        return
    mapping = {
        "text_onnx": "TEXT_ONNX_PATH",
        "tokenizer_json": "TEXT_TOKENIZER_PATH",
        "text_repo": "TEXT_HF_REPO_ID",
        "vision_dir": "VISION_LOCAL_DIR",
        "vision_onnx": "VISION_ONNX_PATH",
        "vision_repo": "VISION_HF_REPO_ID",
        "video_dir": "VIDEO_LOCAL_DIR",
        "video_onnx": "VIDEO_ONNX_PATH",
        "video_repo": "VIDEO_HF_REPO_ID",
        "audio_dir": "AUDIO_LOCAL_DIR",
        "audio_onnx": "AUDIO_ONNX_PATH",
        "audio_repo": "AUDIO_HF_REPO_ID",
    }
    for key, env_var in mapping.items():
        value = embedder_paths.get(key)
        if value is not None:
            os.environ.setdefault(env_var, str(value))


def build_model(embedder_paths: dict[str, tp.Any] | None = None) -> OnlineMediaRegressor:
    _configure_embedding_environment(embedder_paths)
    feature_builder = build_feature_builder()
    return OnlineMediaRegressor(feature_builder)


def bootstrap_from_jsonl(
    stream_path: pathlib.Path | str,
    embedder_paths: dict[str, tp.Any] | None = None,
    horizon_config: dict[str, tp.Sequence[str]] | None = None,
) -> OnlineMediaRegressor:
    model = build_model(embedder_paths)
    horizon_config = horizon_config or DEFAULT_HORIZON_CONFIG
    for features, targets in iter_training_events(stream_path, horizon_config=horizon_config):
        model.learn_one(features, targets)
    return model


def mini_example(
    stream_path: pathlib.Path | str,
    embedder_paths: dict[str, tp.Any] | None = None,
) -> None:
    model = bootstrap_from_jsonl(stream_path, embedder_paths)
    print("MAE:", json.dumps(model.metrics(), indent=2))
