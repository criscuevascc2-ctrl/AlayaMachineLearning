# -*- coding: utf-8 -*-
"""
label.py
Carga datos desde Supabase, reorganiza los DataFrames requeridos y genera un dataset
enriquecido combinando snapshots de media, insights y metricas de cuenta.

Tablas necesarias:
  - ig_media
  - ig_media_insights
  - ig_account_insights_20m
  - ig_account_insights_daily

Salida:
  - ig_media_enriched.jsonl (unico archivo exportado)

Requisitos previos:
  - Definir SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY en .env.local
  - pip install supabase pandas python-dotenv (ver requirements.txt)

Variables opcionales:
  LABEL_OUTPUT_DIR -> directorio donde guardar exportaciones (default _debug/label_exports)
  LABEL_CHUNK_SIZE -> numero de filas por paginacion (default 1000)
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from supabase import Client, create_client

TABLES = [
    "ig_media",
    "ig_media_insights",
    "ig_account_insights_20m",
    "ig_account_insights_daily",
]

MEDIA_COLUMNS = [
    "media_id",
    "ig_user_id",
    "media_type",
    "media_product_type",
    "caption",
    "media_url",
    "like_count_init",
    "comments_count_init",
    "is_comment_enabled",
    "children_count",
    "children_ids",
    "children_media_types",
    "children_media_urls",
    "video_duration_sec",
    "media_width",
    "media_height",
    "is_collab_post",
    "collab_usernames",
    "is_paid_partnership",
    "hashtags",
    "mentions",
    "timestamp_utc",
    "detected_at",
    "last_seen_at",
    "schedule_offsets",
    "next_due_idx",
    "next_due_at",
    "last_snapshot_at",
    "done_all_snapshots",
    "is_active",
    "storage_path",
]

INSIGHTS_COLUMNS = [
    "media_id",
    "taken_at",
    "media_product_type",
    "reach",
    "views",
    "total_interactions",
    "likes",
    "comments",
    "saved",
    "shares",
    "replies",
    "taps_forward",
    "taps_back",
    "exits",
    "ig_reels_avg_watch_time_ms",
    "ig_reels_video_view_total_time_ms",
    "plays",
    "video_duration_sec",
    "engagement",
    "engagement_rate_pct",
    "retention_pct",
    "comments_total",
    "comments_pos",
    "comments_neu",
    "comments_neg",
    "sentiment_avg_score",
]

ACCOUNT20M_COLUMNS = [
    "taken_bucket_20m",
    "followers_count",
    "follows_count",
    "media_count",
    "reach_today",
    "impressions_today",
    "profile_views_today",
    "website_clicks_today",
    "email_contacts_today",
    "phone_call_clicks_today",
    "get_directions_clicks_today",
    "stories_active_count",
    "live_active",
    "followers_delta_20m",
]

DAILY_COLUMNS = [
    "date_utc",
    "reach_day",
    "impressions_day",
    "profile_views_day",
    "website_clicks_day",
    "email_contacts_day",
    "phone_call_clicks_day",
    "get_directions_clicks_day",
    "followers_end_day",
    "followers_delta_day",
    "media_published_day",
    "content_mix_day",
    "reels_watch_time_sum_ms_day",
    "story_interactions_day",
    "hashtags_used_day",
    "collab_posts_day",
    "paid_partnerships_day",
    "link_clicks_day",
    "audience_country",
    "audience_city",
    "audience_gender_age",
]

TABLE_EXPORT_COLUMNS: Dict[str, Optional[List[str]]] = {
    "ig_media": MEDIA_COLUMNS,
    "ig_media_insights": INSIGHTS_COLUMNS,
    "ig_account_insights_20m": ACCOUNT20M_COLUMNS,
    "ig_account_insights_daily": DAILY_COLUMNS,
}

COMBINED_COLUMNS: List[str] = []
for block in (MEDIA_COLUMNS, INSIGHTS_COLUMNS, ACCOUNT20M_COLUMNS, DAILY_COLUMNS):
    for col in block:
        if col not in COMBINED_COLUMNS:
            COMBINED_COLUMNS.append(col)


def load_env_vars(path: Path = Path(".env.local")) -> None:
    if not path.exists():
        return
    in_doc_block = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith(("'''", '"""')):
            in_doc_block = not in_doc_block
            continue
        if in_doc_block or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        value = value.split("#", 1)[0].strip()
        if (value.startswith("\"") and value.endswith("\"")) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key.strip()] = value


def get_supabase_client() -> Client:
    load_env_vars()
    url = (os.getenv("SUPABASE_URL") or "").strip()
    key = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url or not key:
        raise RuntimeError("Faltan SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY en .env.local")
    return create_client(url, key)


def ensure_output_dir() -> Path:
    default_dir = "_debug/label_exports"
    target = Path(os.getenv("LABEL_OUTPUT_DIR", default_dir)).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    return target


def fetch_table_rows(client: Client, table: str, chunk_size: int) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    all_rows: List[Dict[str, Any]] = []
    offset = 0
    total_count: Optional[int] = None

    while True:
        upper = offset + chunk_size - 1
        response = (
            client
            .table(table)
            .select("*", count="exact")
            .range(offset, upper)
            .execute()
        )
        batch = response.data or []
        if total_count is None:
            total_count = response.count
        all_rows.extend(batch)
        offset += len(batch)
        if len(batch) < chunk_size:
            break
    return all_rows, total_count


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({col: [] for col in columns})
    df = df.copy()
    missing = [col for col in columns if col not in df.columns]
    for col in missing:
        df[col] = None
    return df[columns]


def export_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def parse_datetime(value: Any) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if ts is pd.NaT or pd.isna(ts):
        return None
    return ts


def prepare_20m_groups(df: pd.DataFrame, default_user: Optional[Any]) -> Dict[Any, List[Dict[str, Any]]]:
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    if df.empty:
        return groups
    for original in df.to_dict(orient="records"):
        row = dict(original)
        uid = row.get("ig_user_id", default_user)
        if uid is None:
            continue
        row["_taken_dt"] = parse_datetime(row.get("taken_bucket_20m") or row.get("taken_at"))
        groups[uid].append(row)
    for uid, items in groups.items():
        items.sort(key=lambda r: r["_taken_dt"] if r["_taken_dt"] is not None else pd.Timestamp.max)
    return groups


def prepare_daily_lookup(df: pd.DataFrame, default_user: Optional[Any]) -> Dict[Any, Dict[Any, Dict[str, Any]]]:
    lookup: Dict[Any, Dict[Any, Dict[str, Any]]] = defaultdict(dict)
    if df.empty:
        return lookup
    for original in df.to_dict(orient="records"):
        row = dict(original)
        uid = row.get("ig_user_id", default_user)
        if uid is None:
            continue
        dt = parse_datetime(row.get("date_utc"))
        if dt is None:
            continue
        lookup[uid][dt.date()] = row
    return lookup


def pick_nearest_snapshot(entries: Optional[List[Dict[str, Any]]], target: Optional[pd.Timestamp]) -> Optional[Dict[str, Any]]:
    if not entries or target is None:
        return None
    best: Optional[Dict[str, Any]] = None
    best_delta: Optional[float] = None
    for row in entries:
        dt = row.get("_taken_dt")
        if dt is None:
            continue
        try:
            delta = abs((target - dt).total_seconds())
        except Exception:
            continue
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best = row
    return best


def build_enriched_media_rows(
    df_media: pd.DataFrame,
    df_insights: pd.DataFrame,
    df_20m_raw: pd.DataFrame,
    df_daily_raw: pd.DataFrame,
) -> pd.DataFrame:
    if df_media.empty or df_insights.empty:
        return pd.DataFrame({col: [] for col in COMBINED_COLUMNS})

    media_lookup = {row["media_id"]: row for row in df_media.to_dict(orient="records") if row.get("media_id")}
    default_user = None
    if not df_media["ig_user_id"].dropna().empty:
        default_user = df_media["ig_user_id"].dropna().iloc[0]

    groups_20m = prepare_20m_groups(df_20m_raw, default_user)
    daily_lookup = prepare_daily_lookup(df_daily_raw, default_user)

    combined_rows: List[Dict[str, Any]] = []
    insight_records = df_insights.to_dict(orient="records")

    for insight in insight_records:
        media_id = insight.get("media_id")
        media_row = media_lookup.get(media_id)
        if not media_row:
            continue

        taken_at_ts = parse_datetime(insight.get("taken_at"))
        ig_user_id = media_row.get("ig_user_id")
        nearest_snapshot = pick_nearest_snapshot(groups_20m.get(ig_user_id), taken_at_ts)

        daily_row: Optional[Dict[str, Any]] = None
        date_key = taken_at_ts.date() if taken_at_ts is not None else None
        if date_key is None:
            media_ts = parse_datetime(media_row.get("timestamp_utc"))
            if media_ts is not None:
                date_key = media_ts.date()
        if ig_user_id is not None and date_key is not None:
            daily_row = daily_lookup.get(ig_user_id, {}).get(date_key)

        record: Dict[str, Any] = {}
        for col in MEDIA_COLUMNS:
            record[col] = media_row.get(col)
        for col in INSIGHTS_COLUMNS:
            record[col] = insight.get(col)

        if nearest_snapshot:
            for col in ACCOUNT20M_COLUMNS:
                record[col] = nearest_snapshot.get(col)
        else:
            for col in ACCOUNT20M_COLUMNS:
                record.setdefault(col, None)

        if daily_row:
            for col in DAILY_COLUMNS:
                record[col] = daily_row.get(col)
        else:
            for col in DAILY_COLUMNS:
                record.setdefault(col, None)

        combined_rows.append(record)

    if not combined_rows:
        return pd.DataFrame({col: [] for col in COMBINED_COLUMNS})

    combined_df = pd.DataFrame(combined_rows)
    combined_df = ensure_columns(combined_df, COMBINED_COLUMNS)
    if {"media_id", "taken_at"}.issubset(combined_df.columns):
        combined_df = combined_df.sort_values(["media_id", "taken_at"], kind="mergesort")
    return combined_df


def main() -> None:
    chunk_size = int(os.getenv("LABEL_CHUNK_SIZE", "1000"))
    client = get_supabase_client()
    output_dir = ensure_output_dir()

    raw_dfs: Dict[str, pd.DataFrame] = {}
    export_dfs: Dict[str, pd.DataFrame] = {}
    summary: List[str] = []

    for table in TABLES:
        print(f"[INFO] Descargando tabla {table}...")
        rows, total = fetch_table_rows(client, table, chunk_size)
        raw_df = pd.DataFrame(rows)
        raw_dfs[table] = raw_df

        export_cols = TABLE_EXPORT_COLUMNS.get(table)
        if export_cols:
            export_df = ensure_columns(raw_df, export_cols)
        else:
            export_df = raw_df.copy()
        export_dfs[table] = export_df

        summary.append(
            f"{table}: {len(export_df)} filas (total reportado={total})"
        )
        print(f"[OK] {table}: filas={len(export_df)}, total={total}")

    df_media = export_dfs.get("ig_media", pd.DataFrame())
    df_insights = export_dfs.get("ig_media_insights", pd.DataFrame())
    df_20m_raw = raw_dfs.get("ig_account_insights_20m", pd.DataFrame())
    df_daily_raw = raw_dfs.get("ig_account_insights_daily", pd.DataFrame())

    enriched_df = build_enriched_media_rows(df_media, df_insights, df_20m_raw, df_daily_raw)
    enriched_json = output_dir / "ig_media_enriched.jsonl"
    enriched_records = enriched_df.to_dict(orient="records")
    export_jsonl(enriched_records, enriched_json)
    summary.append(
        f"ig_media_enriched: {len(enriched_df)} filas -> {enriched_json}"
    )
    print(
        f"[OK] ig_media_enriched: filas={len(enriched_df)}, archivo={enriched_json.name}"
    )

    print("\nResumen:")
    for line in summary:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
