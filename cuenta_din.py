# -*- coding: utf-8 -*-
"""
Snapshot 20m (IG Account) — Instagram Graph API v23.0
- Inserta/actualiza en public.ig_account_insights_20m (PK: ig_user_id, taken_bucket_20m).
- 'Instantáneo': followers_count, follows_count, media_count, stories activas, live activo.
- 'Día': reach (time series), profile_views (total_value), content_views (total_value),
         website_clicks (total_value), y taps por botón desde profile_links_taps (total_value + breakdown).
- Desglose followers/non-followers: reach_breakdown_today (primero con reach; si no aplica, intenta views/content_views).
- online_followers: bucket de la hora local (histórico últimos 30 días; puede venir vacío {}).

Requisitos:
  pip install python-dotenv requests supabase
  (opcional) pip install redis
"""

import os, json, time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from supabase import create_client, Client

# ================== CONFIG ==================
load_dotenv(".env.local")

IG_USER_ID = (os.getenv("IG_USER_ID") or "").strip()
IG_TOKEN   = (os.getenv("IG_ACCESS_TOKEN") or "").strip()
SB_URL     = (os.getenv("SUPABASE_URL") or "").strip()
SB_KEY     = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
API_VERSION= (os.getenv("API_VERSION") or "v23.0").strip()
ACCOUNT_TZ = (os.getenv("ACCOUNT_TZ") or "America/Santiago").strip()
API_BUDGET = int(os.getenv("IG_API_BUDGET", "120"))
DEBUG_JSON = (os.getenv("DEBUG_JSON") or "0").strip() == "1"
USE_STORY_CACHE = int(os.getenv("USE_STORY_CACHE", "1"))

# Redis opcional para DMs (si lo usas)
REDIS_URL  = (os.getenv("REDIS_URL") or "").strip()
DM_KEY_RCVD   = (os.getenv("DM_KEY_RCVD")   or f"ig:{IG_USER_ID}:dm_received_20m").strip()
DM_KEY_SENT   = (os.getenv("DM_KEY_SENT")   or f"ig:{IG_USER_ID}:dm_sent_20m").strip()
DM_KEY_THREADS= (os.getenv("DM_KEY_THREADS")or f"ig:{IG_USER_ID}:dm_new_threads_20m").strip()
DM_KEY_RT     = (os.getenv("DM_KEY_RT")     or f"ig:{IG_USER_ID}:response_time_dm_sec").strip()
DM_KEY_QUEUE  = (os.getenv("DM_KEY_QUEUE")  or f"ig:{IG_USER_ID}:queued_dms").strip()

assert IG_USER_ID and IG_TOKEN and SB_URL and SB_KEY, "Faltan variables de entorno."

BASE = f"https://graph.facebook.com/{API_VERSION}"
sb: Client = create_client(SB_URL, SB_KEY)
API_CALLS = 0

def _pdbg(label, payload):
    if DEBUG_JSON:
        try:
            print(f"[DEBUG] {label}: {json.dumps(payload, ensure_ascii=False)[:2000]}")
        except Exception:
            pass

# ================== IG helpers ==================
def ig_get(path: str, params: dict | None = None) -> dict:
    """GET a Graph API con manejo de presupuesto y errores."""
    global API_CALLS
    if API_CALLS >= API_BUDGET:
        raise RuntimeError(f"Presupuesto IG agotado ({API_BUDGET}).")
    p = dict(params or {})
    p["access_token"] = IG_TOKEN
    url = f"{BASE}/{path}"
    r = requests.get(url, params=p, timeout=30)
    API_CALLS += 1
    if r.status_code == 200:
        data = r.json()
        _pdbg(f"GET {path}", data)
        return data
    try:
        payload = r.json()
    except Exception:
        payload = {"text": r.text}
    raise RuntimeError(f"IG GET {r.status_code}: {payload}")

def ig_paginate_items(path: str, params: dict | None = None):
    p = dict(params or {})
    while True:
        data = ig_get(path, p)
        for it in data.get("data", []):
            yield it
        next_url = (data.get("paging") or {}).get("next")
        if not next_url:
            break
        from urllib.parse import urlparse, parse_qs
        after = (parse_qs(urlparse(next_url).query).get("after") or [None])[0]
        if not after:
            break
        p["after"] = after

# ================== Fetchers ==================
def fetch_account_counts() -> dict:
    """followers_count, follows_count, media_count (instantáneo)."""
    try:
        f = "username,followers_count,follows_count,media_count"
        data = ig_get(f"{IG_USER_ID}", {"fields": f})
        return {
            "followers_count": data.get("followers_count"),
            "follows_count": data.get("follows_count"),
            "media_count": data.get("media_count"),
        }
    except Exception as e:
        print(f"[WARN] account counts: {e}")
        return {}

def _pick_last_value(item: dict):
    vals = (item or {}).get("values") or []
    if not vals: return None
    return vals[-1].get("value")

def fetch_reach_day() -> int | None:
    """reach (period=day) — time series, sin metric_type."""
    try:
        res = ig_get(f"{IG_USER_ID}/insights", {"metric": "reach", "period": "day"})
        data = res.get("data") or []
        if not data: return None
        v = _pick_last_value(data[0])
        return int(v) if isinstance(v, (int, float)) else None
    except Exception as e:
        print(f"[WARN] reach day: {e}")
        return None

def fetch_totalvalue_day() -> dict:
    """
    profile_views, content_views, website_clicks, profile_links_taps
    con metric_type=total_value (v23.0).
    """
    out = {
        "profile_views_today": None,
        "content_views_today": None,
        "website_clicks_today": None,
        "profile_links_taps_total_today": None,
    }
    try:
        res = ig_get(f"{IG_USER_ID}/insights", {
            "metric": "profile_views,content_views,website_clicks,profile_links_taps",
            "period": "day",
            "metric_type": "total_value",
        })
        for item in res.get("data", []):
            name = item.get("name")
            v = _pick_last_value(item)
            if name == "profile_views":
                out["profile_views_today"] = int(v) if isinstance(v, (int, float)) else None
            elif name in ("content_views","views"):
                out["content_views_today"] = int(v) if isinstance(v, (int, float)) else None
            elif name == "website_clicks":
                out["website_clicks_today"] = int(v) if isinstance(v, (int, float)) else None
            elif name == "profile_links_taps":
                out["profile_links_taps_total_today"] = int(v) if isinstance(v, (int, float)) else None
    except Exception as e:
        print(f"[WARN] total_value day: {e}")
    return out

def fetch_profile_link_taps_breakdown_day() -> dict:
    """
    Breakdown por tipo de botón (EMAIL/CALL/DIRECTIONS/WEBSITE...).
    """
    try:
        res = ig_get(f"{IG_USER_ID}/insights", {
            "metric": "profile_links_taps",
            "period": "day",
            "metric_type": "total_value",
            "breakdown": "contact_button_type",
        })
        data = res.get("data") or []
        if not data: return {}
        v = _pick_last_value(data[0])
        if isinstance(v, dict):
            out = {}
            for k, val in v.items():
                try: out[str(k).upper()] = int(val)
                except Exception: pass
            return out
    except Exception as e:
        print(f"[WARN] profile_links_taps breakdown: {e}")
    return {}

def fetch_followtype_breakdown_reach_day() -> dict | None:
    """FOLLOWER/NON_FOLLOWER para reach (si aplica)."""
    try:
        res = ig_get(f"{IG_USER_ID}/insights", {
            "metric": "reach",
            "period": "day",
            "breakdown": "follow_type",
        })
        data = res.get("data") or []
        if not data: return None
        v = _pick_last_value(data[0])
        if isinstance(v, dict):
            out = {}
            if "FOLLOWER" in v: out["FOLLOWER"] = int(v["FOLLOWER"])
            if "NON_FOLLOWER" in v: out["NON_FOLLOWER"] = int(v["NON_FOLLOWER"])
            return out if out else None
        return None
    except Exception as e:
        print(f"[WARN] reach follow_type: {e}")
        return None

def fetch_followtype_breakdown_views_day() -> dict | None:
    """FOLLOWER/NON_FOLLOWER para views/content_views (total_value) — fallback si reach no trae dict."""
    try:
        res = ig_get(f"{IG_USER_ID}/insights", {
            "metric": "views",  # probar primero 'views'
            "period": "day",
            "metric_type": "total_value",
            "breakdown": "follow_type",
        })
        data = res.get("data") or []
        if data:
            v = _pick_last_value(data[0])
            if isinstance(v, dict):
                out = {}
                if "FOLLOWER" in v: out["FOLLOWER"] = int(v["FOLLOWER"])
                if "NON_FOLLOWER" in v: out["NON_FOLLOWER"] = int(v["NON_FOLLOWER"])
                if out: return out
        # fallback a content_views
        res2 = ig_get(f"{IG_USER_ID}/insights", {
            "metric": "content_views",
            "period": "day",
            "metric_type": "total_value",
            "breakdown": "follow_type",
        })
        data2 = res2.get("data") or []
        if data2:
            v2 = _pick_last_value(data2[0])
            if isinstance(v2, dict):
                out = {}
                if "FOLLOWER" in v2: out["FOLLOWER"] = int(v2["FOLLOWER"])
                if "NON_FOLLOWER" in v2: out["NON_FOLLOWER"] = int(v2["NON_FOLLOWER"])
                if out: return out
    except Exception as e:
        print(f"[WARN] views/content_views follow_type: {e}")
    return None

def fetch_online_followers_now() -> tuple[int | None, int | None]:
    """
    'online_followers' = distribución 24h (histórico 30 días).
    Tomamos el bucket de la hora local de ACCOUNT_TZ para estimar 'ahora'.
    """
    try:
        res = ig_get(f"{IG_USER_ID}/insights", {"metric": "online_followers", "period": "lifetime"})
        values = (res.get("data") or [{}])[0].get("values") or []
        if not values:
            return None, None
        dist = values[-1].get("value")
        hour_local = datetime.now(ZoneInfo(ACCOUNT_TZ)).hour
        if isinstance(dist, dict):
            val = dist.get(str(hour_local)) or dist.get(hour_local)
        elif isinstance(dist, list) and len(dist) == 24:
            val = dist[hour_local]
        else:
            val = None
        return (int(val) if val is not None else None), hour_local
    except Exception as e:
        print(f"[WARN] online_followers: {e}")
        return None, None

def fetch_stories_active_count() -> int:
    if USE_STORY_CACHE:
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            query = (
                sb.table("ig_media")
                .select("media_id")
                .eq("ig_user_id", IG_USER_ID)
                .eq("media_product_type", "STORIES")
                .gte("last_seen_at", cutoff)
                .limit(500)
                .execute()
            )
            data = query.data or []
            if data:
                return len([row for row in data if row.get("media_id")])
        except Exception as exc:
            print(f"[WARN] stories cache fallback: {exc}")

    try:
        n = 0
        for _ in ig_paginate_items(f"{IG_USER_ID}/stories", params={"fields": "id", "limit": 100}):
            n += 1
        return n
    except Exception as e:
        print(f"[WARN] stories: {e}")
        return 0

def fetch_live_active_flag() -> bool:
    try:
        res = ig_get(f"{IG_USER_ID}/live_media", {"fields": "id", "limit": 5})
        return len(res.get("data") or []) > 0
    except Exception as e:
        print(f"[WARN] live_media: {e}")
        return False

# ================== Redis opcional (DMs) ==================
def fetch_dm_counters_from_redis() -> dict:
    out = {
        "dm_received_20m": None,
        "dm_sent_20m": None,
        "dm_new_threads_20m": None,
        "response_time_dm_sec": None,
        "queued_dms": None,
    }
    if not REDIS_URL:
        return out
    try:
        import redis
        r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        def get_int(key):
            v = r.get(key)
            try:
                return int(float(v)) if v is not None else None
            except Exception:
                return None
        out["dm_received_20m"]      = get_int(DM_KEY_RCVD)
        out["dm_sent_20m"]          = get_int(DM_KEY_SENT)
        out["dm_new_threads_20m"]   = get_int(DM_KEY_THREADS)
        out["response_time_dm_sec"] = get_int(DM_KEY_RT)
        out["queued_dms"]           = get_int(DM_KEY_QUEUE)
    except Exception as e:
        print(f"[WARN] Redis: {e}")
    return out

# ================== DB helpers ==================
def select_last_rows(n: int = 2) -> list[dict]:
    q = sb.table("ig_account_insights_20m") \
        .select("*") \
        .eq("ig_user_id", IG_USER_ID) \
        .order("taken_bucket_20m", desc=True) \
        .limit(n) \
        .execute()
    return q.data or []

def upsert_account_row(row: dict):
    sb.table("ig_account_insights_20m").upsert(row).execute()

# ================== Smoothing & deltas ==================
def compute_followers_delta(curr_followers: int | None, last_row: dict | None) -> int | None:
    try:
        if curr_followers is None or not last_row:
            return None
        prev = last_row.get("followers_count")
        return (curr_followers - prev) if isinstance(prev, int) else None
    except Exception:
        return None

def compute_online_followers_smooth(curr: int | None, last_rows: list[dict]) -> int | None:
    vals = []
    if curr is not None: vals.append(curr)
    for r in last_rows:
        v = r.get("online_followers")
        if isinstance(v, int): vals.append(v)
        if len(vals) >= 3: break
    if not vals: return None
    try:
        return int(round(sum(vals) / len(vals)))
    except Exception:
        return None

# ================== MAIN ==================
def snapshot_account_20m() -> dict:
    # Instantáneo
    base = fetch_account_counts()

    # Día (v23): reach separado; el resto con metric_type=total_value
    reach_today = fetch_reach_day()
    tv = fetch_totalvalue_day()

    # Taps detallados por tipo
    taps = fetch_profile_link_taps_breakdown_day()
    email = taps.get("EMAIL") or taps.get("EMAIL_CONTACT")
    phone = taps.get("CALL")  or taps.get("PHONE_CALL")
    dirs  = taps.get("DIRECTIONS") or taps.get("GET_DIRECTIONS")
    web_taps = taps.get("WEBSITE")

    # Breakdown followers/non-followers
    reach_ft = fetch_followtype_breakdown_reach_day()
    if reach_ft is None:
        reach_ft = fetch_followtype_breakdown_views_day()  # fallback a views/content_views

    # Otros “instantáneos”
    online_now, hour_local = fetch_online_followers_now()
    stories_n = fetch_stories_active_count()
    live_flag = fetch_live_active_flag()
    dms = fetch_dm_counters_from_redis()

    # Previos para delta/smoothing
    prev_rows = select_last_rows(1)
    prev = prev_rows[0] if prev_rows else None

    # Mapea content_views_today → impressions_today (para evitar NULL en tu esquema)
    impressions_today = tv.get("content_views_today")

    # website_clicks: usa website_clicks_today; si no viene, usa web_taps del breakdown
    website_clicks_today = tv.get("website_clicks_today")
    if website_clicks_today is None and isinstance(web_taps, int):
        website_clicks_today = web_taps

    row = {
        "ig_user_id": IG_USER_ID,
        "taken_at": datetime.now(timezone.utc).isoformat(),  # bucket_20min lo genera la DB

        # Instantáneo
        "followers_count": base.get("followers_count"),
        "follows_count": base.get("follows_count"),
        "media_count": base.get("media_count"),

        # Day
        "reach_today": reach_today,
        "impressions_today": impressions_today,                 # <- content_views mapeado
        "profile_views_today": tv.get("profile_views_today"),
        "website_clicks_today": website_clicks_today,
        "email_contacts_today": email if isinstance(email, int) else None,
        "phone_call_clicks_today": phone if isinstance(phone, int) else None,
        "get_directions_clicks_today": dirs if isinstance(dirs, int) else None,

        # Breakdown followers/non-followers (reach o fallback views)
        "reach_breakdown_today": reach_ft,

        # Otros
        "online_followers": online_now,
        "online_followers_hour": hour_local,
        "stories_active_count": stories_n,
        "live_active": live_flag,

        # DMs (si provees vía Redis o los dejas None)
        "dm_received_20m": dms["dm_received_20m"],
        "dm_sent_20m": dms["dm_sent_20m"],
        "dm_new_threads_20m": dms["dm_new_threads_20m"],
        "response_time_dm_sec": dms["response_time_dm_sec"],
        "queued_dms": dms["queued_dms"],
    }

    # Extras calculadas
    row["followers_delta_20m"]     = compute_followers_delta(row["followers_count"], prev)
    row["online_followers_smooth"] = compute_online_followers_smooth(row["online_followers"], prev_rows)

    # UPSERT
    upsert_account_row(row)
    return row

if __name__ == "__main__":
    try:
        t0 = time.time()
        print(f"[INFO] Snapshot cuenta 20m | IG_USER={IG_USER_ID} | api={API_VERSION} | tz={ACCOUNT_TZ}")
        row = snapshot_account_20m()
        print(json.dumps({"ok": True, "api_calls": API_CALLS, "elapsed_sec": round(time.time()-t0,2)}, ensure_ascii=False))
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
