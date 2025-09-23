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

import os, json, time, re
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
METRICS_REELS   = "views,reach,total_interactions,saved,likes,comments,shares,ig_reels_avg_watch_time,ig_reels_video_view_total_time"

# STORIES (activas)
METRICS_STORIES = "views,reach,replies,taps_forward,taps_back,exits"

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
    "taps_forward": "taps_forward",
    "taps_back": "taps_back",
    "exits": "exits",
    "plays": "plays",
    # Watch-time (API devuelve ms; guardamos en *_ms)
    "ig_reels_avg_watch_time": "ig_reels_avg_watch_time_ms",
    "ig_reels_video_view_total_time": "ig_reels_video_view_total_time_ms",
}

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

# ================== Util — order con nulls first (compat varias versiones) ==================
def order_nulls_first(req, column: str):
    """
    Intenta .order(..., nullsfirst=True); si la versión no soporta, intenta nulls_first; si no, sin flag.
    """
    try:
        return req.order(column, desc=False, nullsfirst=True)
    except TypeError:
        try:
            return req.order(column, desc=False, nulls_first=True)
        except TypeError:
            return req.order(column, desc=False)

# ================== Candidatos ==================
def fetch_candidates_from_db(limit: int) -> list[dict]:
    """
    Lee candidatos desde la vista v_ig_media_due:
      - Vencidos (next_due_at <= now())
      - >28 días SIN snapshot (para 1 captura y apagar)
    Devuelve: [{media_id, surface, video_duration_sec}]
    """
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
        # No rompas la corrida si no tienes permisos para stories
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

# ================== Batch de insights ==================
def fetch_insights_batch(items: list[dict]) -> dict:
    """
    items: [{media_id, surface, video_duration_sec}]
    Devuelve: dict { media_id: {metric_name: value, ...}, ... }
    - POST / (batch) en chunks de 50.
    - Si sub-respuesta falla, hace fallback individual con set mínimo.
    """
    results: dict[str, dict] = {}

    for i in range(0, len(items), BATCH_SIZE):
        chunk = items[i:i+BATCH_SIZE]
        subreqs, mids, fallbacks = [], [], []

        for m in chunk:
            metrics_str, fb_str = pick_metrics(m["surface"])
            rel_url = f"{m['media_id']}/insights?metric={metrics_str}"
            if m["surface"] == "STORIES":
                rel_url += "&period=day"
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
                            name = "views"   # normalización
                        met[name] = val
                    results[mid] = met
                    continue
                except Exception as ex:
                    print(f"[WARN] parse body media={mid}: {ex} | body[:160]={str(body)[:160]}")

            # fallback si falla
            surf, primary, fb = fallbacks[k]
            results[mid] = fetch_insights_single(mid, surf, primary, fb)

        time.sleep(0.1)  # respiro corto

    return results

def fetch_insights_single(media_id: str, surface: str, primary_metrics: str | None, fallback_str: str | None) -> dict:
    """Fallback individual (1 request) priorizando el set completo disponible."""
    global API_CALLS

    attempts: list[str] = []
    for candidate in (primary_metrics, fallback_str):
        if candidate and candidate not in attempts:
            attempts.append(candidate)

    for metrics in attempts:
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
                print(f"[WARN] single HTTP {r.status_code} media={media_id} metrics={metrics}: {r.text[:180]}")
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
def build_insights_row(media_id: str, surface: str, metrics_dict: dict, video_duration_sec):
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
    }
    for name, val in (metrics_dict or {}).items():
        dest = FIELD_MAP.get(name)
        if dest is not None:
            row[dest] = val
    if video_duration_sec:
        try: row["video_duration_sec"] = float(video_duration_sec)
        except: pass
    return row

def upsert_insights_rows(rows: list[dict]):
    if not rows:
        return

    remaining = list(rows)
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
    insights_map = fetch_insights_batch(items)

    # 4) construir filas y upsert en bloques
    buffer, processed, empties = [], 0, 0
    for it in items:
        mid, surf, dur = it["media_id"], it["surface"], it.get("video_duration_sec")
        mdict = insights_map.get(mid, {})
        if not mdict:
            empties += 1
        row = build_insights_row(mid, surf, mdict, dur)
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


