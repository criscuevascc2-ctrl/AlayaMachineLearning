# -*- coding: utf-8 -*-
"""
Snapshot de Insights (Instagram Graph API v23.0) con Batch + fallback.
- Selecciona candidatos desde la VISTA v_ig_media_due (vencidos + >28d sin snapshot).
- Pide insights por BATCH (máx 50 sub-requests por POST).
- Si un sub-request falla, hace 1 GET individual con set mínimo por superficie.
- Guarda en public.ig_media_insights (PK: media_id, taken_at).
- Al final, ejecuta el RPC ig_media_sunset_28d() para “apagar” >28d con snapshot.

Requisitos:
  pip install python-dotenv requests supabase python-dateutil
Permisos mínimos:
  instagram_basic, instagram_manage_insights
"""

import os, json, time, re, math
from datetime import datetime, timezone
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from supabase import create_client, Client
from postgrest.exceptions import APIError

# ================== CONFIG ==================
load_dotenv(".env.local")

IG_USER_ID = (os.getenv("IG_USER_ID") or "").strip()
IG_TOKEN   = (os.getenv("IG_ACCESS_TOKEN") or "").strip()
SB_URL     = (os.getenv("SUPABASE_URL") or "").strip()
SB_KEY     = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()

# Ventana histórica solo se usa para otras rutas; aquí trabajamos con la vista due
CUTOFF_DAYS = int(os.getenv("INSIGHTS_CUTOFF_DAYS", "30"))

# Presupuesto de llamadas HTTP (POST batch cuenta 1; GET fallback cuenta 1)
API_BUDGET  = int(os.getenv("INSIGHTS_API_BUDGET", "180"))

# Máximo de medios a procesar en esta corrida
MAX_MEDIA   = int(os.getenv("INSIGHTS_MAX_MEDIA", "1200"))

# Tamaño de batch Graph (máx 50)
BATCH_SIZE  = min(int(os.getenv("GRAPH_BATCH_SIZE", "50")), 50)

# Graph API v23.0
BASE = os.getenv("GRAPH_BASE", "https://graph.facebook.com/v23.0")

assert IG_USER_ID and IG_TOKEN and SB_URL and SB_KEY, "Faltan IG_USER_ID / IG_ACCESS_TOKEN / SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY"

sb: Client = create_client(SB_URL, SB_KEY)
API_CALLS = 0

# ================== MÉTRICAS por superficie (SIN 'impressions' en v22+) ==================
# FEED (imagen o video no-reels)
METRICS_FEED    = "views,reach,total_interactions,saved,likes,comments,shares"

# REELS (video)
METRICS_REELS   = "views,reach,total_interactions,saved,likes,comments,shares,ig_reels_avg_watch_time,ig_reels_video_view_total_time,plays"

# STORIES (activas) -> SOLO válidas en metric=..., no incluir taps_* ni exits aquí
METRICS_STORIES = "views,reach,replies"

# Fallbacks mínimos (si falla el batch) — también SIN impressions
FALLBACK_FEED     = "views,reach,total_interactions"
FALLBACK_REELS    = "views,reach,total_interactions"
FALLBACK_STORIES  = "views,reach,replies"

# Mapeo métrica API -> columna tabla
FIELD_MAP = {
    "reach": "reach",
    "views": "views",
    "video_views": "views",  # alias si la API devuelve este nombre
    "total_interactions": "total_interactions",
    "saved": "saved",
    "likes": "likes",
    "comments": "comments",
    "shares": "shares",
    "replies": "replies",
    "plays": "plays",

    # Story navigation → columnas propias (NO shares)
    "tap_forward": "taps_forward",
    "taps_forward": "taps_forward",
    "swipe_forward": "taps_forward",  # Next story también a taps_forward
    "tap_back": "taps_back",
    "taps_back": "taps_back",
    "tap_exit": "exits",
    "exits": "exits",

    # Reels watch-time (API devuelve ms; guardamos en *_ms)
    "ig_reels_avg_watch_time": "ig_reels_avg_watch_time_ms",
    "ig_reels_video_view_total_time": "ig_reels_video_view_total_time_ms",
}

# Columnas permitidas para upsert (sanitización anti-PGRST204)
ALLOWED_COLUMNS = {
    "media_id","taken_at","media_product_type",
    "reach","impressions","views","total_interactions",
    "likes","comments","saved","shares","replies",
    "taps_forward","taps_back","exits",
    "ig_reels_avg_watch_time_ms","ig_reels_video_view_total_time_ms",
    "plays","video_duration_sec",
    "comments_total","comments_pos","comments_neu","comments_neg","sentiment_avg_score",
    "snapshot_horizon",
}

def _refine_metrics_on_error(metrics_str: str | None, body_text: str | None) -> str | None:
    if not metrics_str or not body_text:
        return None
    lowered = body_text.lower()
    metrics = [m.strip() for m in metrics_str.split(',') if m.strip()]
    if not metrics:
        return None
    new_metrics = list(metrics)

    if any(m.lower() == 'plays' for m in new_metrics):
        if 'plays metric is no longer supported' in lowered:
            new_metrics = [m for m in new_metrics if m.lower() != 'plays']
        elif 'metric[0]' in lowered and 'plays' in lowered and 'must be one of the following' in lowered:
            new_metrics = [m for m in new_metrics if m.lower() != 'plays']

    if new_metrics == metrics:
        return None
    return ','.join(new_metrics) if new_metrics else None

def sanitize_row_for_upsert(row: dict) -> dict:
    """Elimina llaves que no existan en la tabla (evita PGRST204)."""
    return {k: v for k, v in row.items() if k in ALLOWED_COLUMNS}


HORIZON_CONFIG = {
    'FEED': ('01:00:00', '03:00:00', '06:00:00', '12:00:00', '1 day', '3 days', '7 days', '28 days'),
    'REELS': ('01:00:00', '03:00:00', '06:00:00', '12:00:00', '1 day', '3 days', '7 days', '28 days'),
    'STORIES': tuple(f"{i:02d}:00:00" for i in range(1, 24)),
}


def _parse_iso_datetime(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except Exception:
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None


def _offset_to_hours(offset: str | None) -> float | None:
    if not offset:
        return None
    raw = offset.strip().lower()
    if not raw:
        return None
    try:
        if 'day' in raw:
            return float(raw.split()[0]) * 24.0
        if 'hour' in raw:
            return float(raw.split()[0])
        if 'minute' in raw:
            return float(raw.split()[0]) / 60.0
        if ':' in raw:
            parts = raw.split(':')
            if len(parts) == 3:
                hh, mm, ss = parts
                return int(hh) + int(mm) / 60.0 + int(ss) / 3600.0
    except Exception:
        return None
    return None


def _infer_snapshot_horizon(surface: str, media_meta: dict | None, taken_at_iso: str) -> dict | None:
    if not media_meta:
        return None
    posted_iso = media_meta.get('timestamp_utc')
    posted = _parse_iso_datetime(posted_iso)
    taken_at = _parse_iso_datetime(taken_at_iso) or datetime.now(timezone.utc)
    if not posted or posted > taken_at:
        return None

    elapsed_hours = (taken_at - posted).total_seconds() / 3600.0
    if not math.isfinite(elapsed_hours):
        return None

    config = HORIZON_CONFIG.get((surface or 'FEED').upper(), HORIZON_CONFIG['FEED'])
    hours_list = [_offset_to_hours(val) for val in config]
    candidates = [
        (abs(elapsed_hours - h), idx)
        for idx, h in enumerate(hours_list)
        if h is not None
    ]

    if not candidates:
        return None

    _, idx = min(candidates, key=lambda pair: pair[0])
    label = config[idx] if idx < len(config) else None
    chosen_hours = hours_list[idx] if idx < len(hours_list) else None

    info = {
        'label': label,
        'index': idx,
        'elapsed_hours': round(elapsed_hours, 3),
    }
    if chosen_hours is not None:
        info['hours'] = round(chosen_hours, 3)
    return info


# ================== HTTP session (keep-alive + retry) ==================
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.2,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(pool_connections=40, pool_maxsize=80, max_retries=retries)
    s.mount("https://", adapter); s.mount("http://", adapter)
    return s

SESSION = make_session()

# ================== IG Helpers ==================
def ig_get(path, params=None, raw_url=False):
    """GET simple; si raw_url=True usamos la URL completa (paginación 'next')."""
    if raw_url:
        r = SESSION.get(path, timeout=30)
    else:
        params = dict(params or {})
        params["access_token"] = IG_TOKEN
        r = SESSION.get(f"{BASE}/{path}", params=params, timeout=30)
    if r.status_code == 429:
        time.sleep(60); r = SESSION.get(r.url, timeout=30)
    if r.status_code >= 400:
        raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}")
    return r.json()

def ig_post(path, data=None):
    """POST; si path == '' usa /v23.0/ (endpoint batch). Cuenta para budget."""
    global API_CALLS
    if API_CALLS >= API_BUDGET:
        raise RuntimeError("Budget de llamadas HTTP agotado.")
    data = dict(data or {})
    data["access_token"] = IG_TOKEN
    url = f"{BASE}/{path}" if path else f"{BASE}/"
    r = SESSION.post(url, data=data, timeout=60)
    API_CALLS += 1
    if r.status_code == 429:
        time.sleep(60); r = SESSION.post(url, data=data, timeout=60); API_CALLS += 1
    if r.status_code >= 400:
        raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}")
    return r.json()

# ================== Util — order con nulls first ==================
def order_nulls_first(req, column: str):
    try:
        return req.order(column, desc=False, nullsfirst=True)
    except TypeError:
        try:
            return req.order(column, desc=False, nulls_first=True)
        except TypeError:
            return req.order(column, desc=False)

# ================== Candidatos ==================
def fetch_candidates_from_db(limit: int) -> list[dict]:
    q = sb.table("v_ig_media_due").select(
        "media_id,media_product_type,video_duration_sec,next_due_at"
    )
    q = order_nulls_first(q, "next_due_at").limit(limit).execute()
    rows = q.data or []
    return [
        {
            "media_id": r["media_id"],
            "surface": (r["media_product_type"] or "FEED").upper(),
            "video_duration_sec": r.get("video_duration_sec"),
        }
        for r in rows
    ]

def fetch_active_stories() -> list[dict]:
    """Trae STORIES activas (no histórico)."""
    items = []
    params = {"fields": "id,timestamp", "limit": 100}
    try:
        data = ig_get(f"{IG_USER_ID}/stories", params)
    except Exception as e:
        print(f"[WARN] fetch_active_stories GET failed: {e}")
        return items

    while True:
        for s in data.get("data", []):
            items.append({"media_id": s["id"], "surface": "STORIES", "video_duration_sec": None})
        next_url = (data.get("paging") or {}).get("next")
        if not next_url: break
        data = ig_get(next_url, raw_url=True)
    return items

# ================== Selector de métricas ==================
def pick_metrics(surface: str) -> tuple[str, str]:
    s = (surface or "FEED").upper()
    if s == "REELS":   return METRICS_REELS,   FALLBACK_REELS
    if s == "STORIES": return METRICS_STORIES, FALLBACK_STORIES
    return METRICS_FEED, FALLBACK_FEED

# ================== Story navigation parsing ==================
def parse_story_navigation_metrics(payload: dict | None) -> dict[str, int]:
    """Convierte el payload de metric=navigation a claves normalizadas.
    Soporta:
      - values[].breakdowns[].dimension_values/value
      - total_value.breakdowns[].results[].dimension_values/value
    """
    metrics: dict[str, int] = {}
    if not isinstance(payload, dict):
        return metrics

    def norm(key: str) -> str:
        k = (key or "").strip().lower()
        # Meta suele devolver: SWIPE_FORWARD, TAP_FORWARD, TAP_BACK, TAP_EXIT
        if k in ("swipe_forward", "tap_forward", "tap_back", "tap_exit"):
            return k
        return k

    data = payload.get("data") or []

    for entry in data:
        # ---- 1) Formato clásico: values[].breakdowns[] ----
        values = entry.get("values") or []
        for val_entry in values:
            # values[].value puede ser dict agregando métricas
            val_obj = val_entry.get("value")
            if isinstance(val_obj, dict):
                for key, number in val_obj.items():
                    if isinstance(number, (int, float)):
                        metrics[norm(str(key))] = number
            # o bien venir en 'breakdowns' dentro de cada values[]
            for br in (val_entry.get("breakdowns") or []):
                dims = br.get("dimension_values") or []
                if dims:
                    key = norm(str(dims[0]))
                    val = br.get("value")
                    if isinstance(val, (int, float)):
                        metrics[key] = val

        # ---- 2) Formato que estás recibiendo: total_value.breakdowns.results[] ----
        total_value = entry.get("total_value") or {}
        for br in (total_value.get("breakdowns") or []):
            for res in (br.get("results") or []):
                dims = res.get("dimension_values") or []
                if dims:
                    key = norm(str(dims[0]))
                    val = res.get("value")
                    if isinstance(val, (int, float)):
                        metrics[key] = val

    return metrics


def fetch_story_navigation_single(media_id: str) -> dict[str, int]:
    global API_CALLS
    if API_CALLS >= API_BUDGET:
        return {}

    params = {
        "metric": "navigation",
        "breakdown": "story_navigation_action_type",
        "access_token": IG_TOKEN,
    }

    try:
        r = SESSION.get(f"{BASE}/{media_id}/insights", params=params, timeout=25)
    except requests.RequestException as ex:
        print(f"[WARN] story navigation single request failed media={media_id}: {ex}")
        return {}

    API_CALLS += 1
    if r.status_code == 429:
        time.sleep(60)
        try:
            r = SESSION.get(r.url, timeout=25)
        except requests.RequestException as ex:
            print(f"[WARN] story navigation retry failed media={media_id}: {ex}")
            return {}
        API_CALLS += 1

    if r.status_code != 200:
        print(f"[WARN] story navigation single HTTP {r.status_code} media={media_id}: {r.text[:180]}")
        return {}

    try:
        payload = r.json()
    except Exception as ex:
        print(f"[WARN] story navigation single parse media={media_id}: {ex}")
        return {}

    return parse_story_navigation_metrics(payload)

def fetch_story_navigation_metrics(media_ids: list[str]) -> dict[str, dict]:
    nav_results: dict[str, dict] = {}
    if not media_ids:
        return nav_results

    dedup: list[str] = []
    seen: set[str] = set()
    for mid in media_ids:
        smid = str(mid)
        if smid not in seen:
            seen.add(smid)
            dedup.append(smid)

    for i in range(0, len(dedup), BATCH_SIZE):
        batch_ids = dedup[i:i+BATCH_SIZE]
        subreqs = [
            {
                "method": "GET",
                "relative_url": f"{mid}/insights?metric=navigation&breakdown=story_navigation_action_type"
            }
            for mid in batch_ids
        ]
        try:
            resp = ig_post("", {"batch": json.dumps(subreqs, ensure_ascii=False)})
        except requests.HTTPError as e:
            print(f"[WARN] story navigation batch HTTP error: {e}")
            for mid in batch_ids:
                single = fetch_story_navigation_single(mid)
                if single:
                    nav_results[mid] = single
            continue

        for idx, mid in enumerate(batch_ids):
            entry = resp[idx] if idx < len(resp) else {}
            code = entry.get("code")
            body = entry.get("body")
            payload = None
            if code == 200 and body:
                try:
                    payload = json.loads(body)
                except Exception as ex:
                    print(f"[WARN] story navigation parse body media={mid}: {ex} | body[:160]={str(body)[:160]}")
            nav_map = parse_story_navigation_metrics(payload)
            if nav_map:
                nav_results[mid] = nav_map
            else:
                single = fetch_story_navigation_single(mid)
                if single:
                    nav_results[mid] = single

        time.sleep(0.05)

    return nav_results


def fetch_media_metadata_map(media_ids: list[str]) -> dict[str, dict]:
    if not media_ids:
        return {}

    out: dict[str, dict] = {}
    chunk_size = 200
    for i in range(0, len(media_ids), chunk_size):
        chunk = media_ids[i:i + chunk_size]
        try:
            res = sb.table("ig_media").select("media_id,timestamp_utc,detected_at").in_("media_id", chunk).execute()
        except Exception as e:
            print(f"[WARN] fetch media metadata failed: {e}")
            continue
        for row in (res.data or []):
            mid = row.get("media_id")
            if mid:
                out[str(mid)] = row
    return out

# ================== Batch de insights ==================
def fetch_insights_batch(items: list[dict]) -> dict:
    """
    items: [{media_id, surface, video_duration_sec}]
    Devuelve: dict { media_id: {metric_name: value, ...}, ... }
    - POST / (batch) en chunks de 50.
    - Si sub-respuesta falla, hace fallback individual con set minimo.
    """
    results: dict[str, dict] = {}

    for i in range(0, len(items), BATCH_SIZE):
        chunk = items[i:i+BATCH_SIZE]
        subreqs, mids, fallbacks = [], [], []
        chunk_story_ids: list[str] = []

        for m in chunk:
            metrics_str, fb_str = pick_metrics(m["surface"])
            rel_url = f"{m['media_id']}/insights?metric={metrics_str}"
            if m["surface"] == "STORIES":
                rel_url += "&period=day"
                chunk_story_ids.append(str(m["media_id"]))
            subreqs.append({
                "method": "GET",
                "relative_url": rel_url
            })
            mids.append(m["media_id"])
            fallbacks.append((m["surface"], metrics_str, fb_str))

        try:
            resp = ig_post("", {"batch": json.dumps(subreqs, ensure_ascii=False)})
        except requests.HTTPError as e:
            print(f"[WARN] batch HTTP error: {e}")
            for mid, (surf, primary, fb) in zip(mids, fallbacks):
                results[mid] = fetch_insights_single(mid, surf, primary, fb)
            if chunk_story_ids:
                nav_map = fetch_story_navigation_metrics(chunk_story_ids)
                for sid, nav in nav_map.items():
                    results.setdefault(sid, {}).update(nav)
            continue

        for k, mid in enumerate(mids):
            r = resp[k] if k < len(resp) else {}
            code = r.get("code")
            body = r.get("body")
            if code == 200 and body:
                try:
                    data = json.loads(body).get("data", [])
                    met = {}
                    for e in data:
                        name = e.get("name")
                        val  = (e.get("values") or [{}])[0].get("value")
                        if name == "video_views":
                            name = "views"   # normalizacion
                        met[name] = val
                    results[mid] = met
                    continue
                except Exception as ex:
                    print(f"[WARN] parse body media={mid}: {ex} | body[:160]={str(body)[:160]}")

            # fallback si falla
            surf, primary, fb = fallbacks[k]
            results[mid] = fetch_insights_single(mid, surf, primary, fb)

        time.sleep(0.1)  # respiro corto

        if chunk_story_ids:
            nav_map = fetch_story_navigation_metrics(chunk_story_ids)
            for sid, nav in nav_map.items():
                results.setdefault(sid, {}).update(nav)

    return results

def fetch_insights_single(media_id: str, surface: str, primary_metrics: str | None, fallback_str: str | None) -> dict:
    """Fallback individual (1 request) priorizando el set completo disponible."""
    global API_CALLS

    queue: list[str] = []
    seen: set[str] = set()
    for candidate in (primary_metrics, fallback_str):
        if not candidate:
            continue
        norm = candidate.strip()
        if norm and norm not in seen:
            queue.append(norm)
            seen.add(norm)

    while queue:
        metrics = queue.pop(0)
        if API_CALLS >= API_BUDGET:
            return {}
        try:
            params = {"metric": metrics, "access_token": IG_TOKEN}
            if surface == "STORIES":
                params["period"] = "day"
            r = SESSION.get(f"{BASE}/{media_id}/insights", params=params, timeout=25)
            API_CALLS += 1
            if r.status_code == 429:
                time.sleep(60)
                r = SESSION.get(r.url, timeout=25)
                API_CALLS += 1

            if r.status_code != 200:
                body_text = ""
                try:
                    body_text = r.text or ""
                except Exception:
                    body_text = ""
                refined = _refine_metrics_on_error(metrics, body_text)
                if refined and refined not in seen:
                    seen.add(refined)
                    queue.insert(0, refined)
                    continue
                print(f"[WARN] single HTTP {r.status_code} media={media_id} metrics={metrics}: {body_text[:180]}")
                continue

            data = r.json().get("data", [])
            met = {}
            for e in data:
                name = e.get("name")
                val  = (e.get("values") or [{}])[0].get("value")
                if name == "video_views":
                    name = "views"
                met[name] = val
            if met:
                return met
        except Exception as ex:
            print(f"[WARN] single exception media={media_id} metrics={metrics}: {ex}")
    return {}

# ================== Transform → fila ig_media_insights ==================
def build_insights_row(media_id: str, surface: str, metrics_dict: dict, video_duration_sec, media_meta: dict | None):
    now_ts = datetime.now(timezone.utc).isoformat()
    row = {
        "media_id": media_id,
        "taken_at": now_ts,
        "media_product_type": surface,
        "reach": None, "impressions": None, "views": None, "total_interactions": None,
        "likes": None, "comments": None, "saved": None, "shares": None, "replies": None,
        "taps_forward": None, "taps_back": None, "exits": None,
        "ig_reels_avg_watch_time_ms": None, "ig_reels_video_view_total_time_ms": None,
        "plays": None,
        "video_duration_sec": None,
        "comments_total": None, "comments_pos": None, "comments_neu": None,
        "comments_neg": None, "sentiment_avg_score": None,
        "snapshot_horizon": None,
    }
    # Mapeo directo
    for name, val in (metrics_dict or {}).items():
        dest = FIELD_MAP.get(name)
        if dest is not None:
            row[dest] = val

    horizon_info = _infer_snapshot_horizon(surface, media_meta, now_ts)
    if horizon_info:
        row["snapshot_horizon"] = horizon_info

    # NO mezclar navegación en shares (semánticamente no corresponde)

    if video_duration_sec:
        try: row["video_duration_sec"] = float(video_duration_sec)
        except: pass
    return row

def upsert_insights_rows(rows: list[dict]):
    if not rows:
        return

    remaining = [sanitize_row_for_upsert(r) for r in rows]  # <-- sanitiza antes
    while remaining:
        try:
            sb.table("ig_media_insights").upsert(remaining, on_conflict="media_id,taken_at").execute()
            return
        except APIError as e:
            payload = getattr(e, "response", None) or (e.args[0] if e.args else {})
            message = ""
            code = ""
            if isinstance(payload, dict):
                message = (payload.get("message") or "")
                code = payload.get("code") or ""
            else:
                message = str(payload or e)
            message_lower = message.lower()
            if "muy temprano" in message_lower:
                match = re.search(r"media\s+(\d+)", message)
                media_id = match.group(1) if match else None
                if not media_id and remaining:
                    media_id = str(remaining[0].get("media_id"))
                if media_id:
                    print(f"[WARN] skip snapshot media={media_id}: {message}")
                    remaining = [r for r in remaining if str(r.get("media_id")) != str(media_id)]
                    if not remaining:
                        return
                    continue
            if "ya complet" in message_lower and "snapshot" in message_lower:
                match = re.search(r"media\s+(\d+)", message)
                media_id = match.group(1) if match else None
                if not media_id and remaining:
                    media_id = str(remaining[0].get("media_id"))
                if media_id:
                    print(f"[WARN] skip snapshot media={media_id}: {message}")
                    remaining = [r for r in remaining if str(r.get("media_id")) != str(media_id)]
                    if not remaining:
                        return
                    continue
            print(f"[ERROR] upsert failed (code={code}): {message}")
            raise

# ================== MAIN ==================
def snapshot_insights(max_media: int | None = None) -> int:
    global API_CALLS
    max_media = max_media or MAX_MEDIA

    # 1) candidatos (desde la vista due)
    candidates = fetch_candidates_from_db(limit=max_media)

    # 2) stories activas (opcionales, fuera de schedule de 28d)
    try:
        stories_active = fetch_active_stories()
    except Exception as e:
        print(f"[WARN] fetch_active_stories: {e}")
        stories_active = []

    # Deduplicate when stories appear in both sources
    combined = candidates + stories_active
    unique_items, seen_media = [], set()
    for it in combined:
        mid = str(it["media_id"])
        if mid in seen_media:
            continue
        seen_media.add(mid)
        unique_items.append(it)
    items = unique_items[:max_media]
    if not items:
        print("[INFO] Sin candidatos en ventana temporal.")
        return 0

    # 3) pedir insights (batch + fallback)
    media_meta_map = fetch_media_metadata_map([it["media_id"] for it in items])
    insights_map = fetch_insights_batch(items)

    # 4) construir filas y upsert en bloques
    buffer, processed, empties = [], 0, 0
    for it in items:
        mid, surf, dur = it["media_id"], it["surface"], it.get("video_duration_sec")
        mdict = insights_map.get(mid, {})
        if not mdict:
            empties += 1
        row = build_insights_row(mid, surf, mdict, dur, media_meta_map.get(str(mid)))
        row = sanitize_row_for_upsert(row)  # seguridad extra
        buffer.append(row); processed += 1

        if len(buffer) >= 200:
            upsert_insights_rows(buffer)
            print(f"[DB] Insertados {len(buffer)} snapshots. HTTP calls: {API_CALLS}")
            buffer = []

    if buffer:
        upsert_insights_rows(buffer)
        print(f"[DB] Insertados {len(buffer)} snapshots. HTTP calls: {API_CALLS}")

    if empties:
        print(f"[WARN] {empties} medios devolvieron métricas vacías (revisa permisos/superficie).")

    return processed

if __name__ == "__main__":
    t0 = time.time()
    try:
        print(f"[INFO] Snapshot insights (batch) | cutoff={CUTOFF_DAYS}d | budget={API_BUDGET} | max_media={MAX_MEDIA} | v=23.0")
        n = snapshot_insights()
        dt = time.time() - t0
        print("="*60)
        print(f"OK: {n} snapshots en {dt:.2f}s | Llamadas HTTP: {API_CALLS}")
        print("="*60)
    except Exception as e:
        print("[ERROR]", e)
        raise

    # Sunset masivo >28d con al menos un snapshot (idempotente)
    try:
        r = sb.rpc("ig_media_sunset_28d").execute()
        print(f"[DB] Sunset 28d: {r.data} medios apagados")
    except Exception as e:
        print(f"[WARN] Sunset 28d falló: {e}")


