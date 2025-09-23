# -*- coding: utf-8 -*-
"""
ETL diario (IG Account) — Instagram Graph API v23.0
Rellena public.ig_account_insights_daily.

Campos cubiertos:
- reach_day                         -> insights(metric=reach, period=day)
- impressions_day                   -> insights(metric=views|content_views, period=day, metric_type=total_value)
- profile_views_day                 -> insights(metric=profile_views, period=day, metric_type=total_value)
- website_clicks_day                -> insights(metric=website_clicks, period=day, metric_type=total_value) (fallback: taps WEBSITE)
- email/phone/directions (day)      -> profile_links_taps + breakdown=contact_button_type
- link_clicks_day                   -> total de profile_links_taps (total_value)
- followers_end_day                 -> insights(metric=follower_count, period=day)  [sin metric_type]
- followers_delta_day               -> (a) follows_and_unfollows (total_value) o (b) diff vs día anterior en DB
- media_published_day               -> conteo de /media creados ese día
- content_mix_day                   -> {"REELS":n, "FEED":m, ...} desde media_product_type del día
- hashtags_used_day                 -> total de hashtags en captions
- reels_watch_time_sum_ms_day       -> suma video_view_total_time (ms) de reels publicados ese dia
- story_interactions_day            -> total de interacciones (replies/taps/exits) en stories del dia
- audience_*                        -> follower_demographics (country/city/age/gender) con metric_type=total_value

Append-only si hay cambios:
- Si la tabla tiene columna snapshot_at (PK compuesta con snapshot_at), insert condicional.
- Si no existe snapshot_at, se hace upsert tradicional (compatible con tu PK original).

Requisitos:
  pip install python-dotenv requests supabase
"""

import os, re, json, time
from datetime import datetime, timedelta, timezone, date
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
API_BUDGET = int(os.getenv("IG_API_BUDGET", "200"))
DEBUG_JSON = (os.getenv("DEBUG_JSON") or "0").strip() == "1"
ALWAYS_INSERT = (os.getenv("ALWAYS_INSERT") or "0").strip() == "1"  # solo si hay snapshot_at

# Si quieres fijar la fecha (UTC): export DATE_UTC=YYYY-MM-DD
DATE_UTC_STR = (os.getenv("DATE_UTC") or "").strip()
DATE_UTC = date.fromisoformat(DATE_UTC_STR) if DATE_UTC_STR else datetime.now(timezone.utc).date()

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

# ================== Helpers de tiempo ==================
def day_bounds_utc(d: date):
    """(since_ts, until_ts) para [d 00:00Z, (d+1) 00:00Z)."""
    start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    end   = start + timedelta(days=1)
    return int(start.timestamp()), int(end.timestamp())

SINCE_TS, UNTIL_TS = day_bounds_utc(DATE_UTC)

# ================== IG helpers ==================
def ig_get(path: str, params: dict | None = None) -> dict:
    """GET a Graph API con manejo de presupuesto y errores."""
    global API_CALLS
    if API_CALLS >= API_BUDGET:
        raise RuntimeError(f"Presupuesto IG agotado ({API_BUDGET}).")
    p = dict(params or {})
    p["access_token"] = IG_TOKEN
    url = f"{BASE}/{path}"
    r = requests.get(url, params=p, timeout=40)
    API_CALLS += 1
    if r.status_code == 200:
        data = r.json()
        _pdbg(f"GET {path}", {"params": {k:v for k,v in p.items() if k!='access_token'}, "data": data})
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

def pick_last_value(item: dict):
    vals = (item or {}).get("values") or []
    if not vals: return None
    last = None
    for v in vals:
        et = v.get("end_time")
        if et:
            try:
                et_dt = datetime.fromisoformat(et.replace("Z", "+00:00"))
                if et_dt.timestamp() <= UNTIL_TS + 1:
                    last = v
            except Exception:
                last = v
        else:
            last = v
    return (last or {}).get("value")

# ================== Fetchers (insights) ==================
def fetch_reach_day() -> int | None:
    try:
        res = ig_get(f"{IG_USER_ID}/insights", {
            "metric": "reach",
            "period": "day",
            "since": SINCE_TS,
            "until": UNTIL_TS
        })
        data = res.get("data") or []
        if not data: return None
        v = pick_last_value(data[0])
        return int(v) if isinstance(v, (int, float)) else None
    except Exception as e:
        print(f"[WARN] reach day: {e}")
        return None

def fetch_total_value_metrics() -> dict:
    """
    profile_views, views/content_views, website_clicks, profile_links_taps (period=day, metric_type=total_value)
    """
    out = {
        "profile_views_day": None,
        "impressions_day": None,     # views/content_views
        "website_clicks_day": None,
        "link_clicks_day": None,     # total profile_links_taps
        "taps_breakdown": {},        # EMAIL/CALL/DIRECTIONS/WEBSITE
    }
    try:
        res = ig_get(f"{IG_USER_ID}/insights", {
            "metric": "profile_views,content_views,website_clicks,profile_links_taps,views",
            "period": "day",
            "metric_type": "total_value",
            "since": SINCE_TS,
            "until": UNTIL_TS
        })
        for item in res.get("data", []):
            name = item.get("name")
            v = pick_last_value(item)
            if name == "profile_views":
                out["profile_views_day"] = int(v) if isinstance(v, (int, float)) else None
            elif name in ("views", "content_views"):
                out["impressions_day"] = int(v) if isinstance(v, (int, float)) else out["impressions_day"]
            elif name == "website_clicks":
                out["website_clicks_day"] = int(v) if isinstance(v, (int, float)) else None
            elif name == "profile_links_taps":
                out["link_clicks_day"] = int(v) if isinstance(v, (int, float)) else None
    except Exception as e:
        print(f"[WARN] total_value day: {e}")

    # breakdown de taps por tipo (EMAIL/CALL/DIRECTIONS/WEBSITE)
    try:
        res2 = ig_get(f"{IG_USER_ID}/insights", {
            "metric": "profile_links_taps",
            "period": "day",
            "metric_type": "total_value",
            "breakdown": "contact_button_type",
            "since": SINCE_TS,
            "until": UNTIL_TS
        })
        data2 = res2.get("data") or []
        if data2:
            v2 = pick_last_value(data2[0])
            if isinstance(v2, dict):
                bk = {}
                for k, val in v2.items():
                    try:
                        bk[str(k).upper()] = int(val)
                    except Exception:
                        pass
                out["taps_breakdown"] = bk
    except Exception as e:
        print(f"[WARN] taps breakdown: {e}")

    # Fallback: usar tap WEBSITE si no vino website_clicks_day
    if out["website_clicks_day"] is None:
        w = out["taps_breakdown"].get("WEBSITE")
        if isinstance(w, int): out["website_clicks_day"] = w

    return out

def fetch_follower_count_day() -> int | None:
    """follower_count (period=day) — sin metric_type en v23."""
    try:
        res = ig_get(f"{IG_USER_ID}/insights", {
            "metric": "follower_count",
            "period": "day",
            "since": SINCE_TS,
            "until": UNTIL_TS
        })
        data = res.get("data") or []
        if not data:
            return None
        v = pick_last_value(data[0])
        return int(v) if isinstance(v, (int, float)) else None
    except Exception as e:
        print(f"[WARN] follower_count day: {e}")
        return None

def fetch_follow_unfollow_day() -> dict:
    """follows_and_unfollows (total_value) → dict {follows, unfollows} si está disponible."""
    try:
        res = ig_get(f"{IG_USER_ID}/insights", {
            "metric": "follows_and_unfollows",
            "period": "day",
            "metric_type": "total_value",
            "since": SINCE_TS,
            "until": UNTIL_TS
        })
        data = res.get("data") or []
        if not data: return {}
        v = pick_last_value(data[0])
        return v if isinstance(v, dict) else {}
    except Exception as e:
        print(f"[WARN] follows_and_unfollows: {e}")
        return {}

# ================== Demografía (snapshot) ==================
def fetch_audience_demographics() -> dict:
    """
    follower_demographics con metric_type=total_value y breakdown validos:
    country, city, age, gender. Devuelve dicts listos para guardar:
      - audience_country: {"CL": 100, "AR": 50, ...}
      - audience_city: {"Santiago": 80, "Buenos Aires": 20, ...}
      - audience_gender_age: {"age": {...}, "gender": {...}}
    """
    out = {
        "audience_country": None,
        "audience_city": None,
        "audience_gender_age": None,
        "audience_locale": None,  # ya no soportado en v23
    }

    def _demo(breakdown: str):
        try:
            res = ig_get(f"{IG_USER_ID}/insights", {
                "metric": "follower_demographics",
                "period": "lifetime",
                "metric_type": "total_value",
                "breakdown": breakdown,
            })
            data = res.get("data") or []
            if not data:
                return None

            tv = (data[0] or {}).get("total_value") or {}
            bks = tv.get("breakdowns") or []
            if not bks:
                return None
            results = bks[0].get("results") or []

            parsed: dict[str, int | None] = {}
            for r in results:
                keys = r.get("dimension_values") or []
                if not keys:
                    continue
                key = str(keys[0])
                try:
                    value = int(r.get("value", 0))
                except Exception:
                    value = None
                parsed[key] = value
            return parsed if parsed else None
        except Exception as e:
            print(f"[WARN] follower_demographics/{breakdown}: {e}")
            return None

    country = _demo("country")
    city = _demo("city")
    age = _demo("age")
    gender = _demo("gender")

    out["audience_country"] = country
    out["audience_city"] = city

    ga = {}
    if isinstance(age, dict):
        ga["age"] = age
    if isinstance(gender, dict):
        ga["gender"] = gender
    out["audience_gender_age"] = ga if ga else None

    return out

# ================== Media del día ==================
def _list_media_for_day() -> list[dict]:
    """
    Devuelve media publicados durante DATE_UTC (UTC).
    """
    fields = "id,caption,timestamp,media_type,media_product_type"
    params = {"fields": fields, "limit": 100}
    start_dt = datetime.fromtimestamp(SINCE_TS, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(UNTIL_TS, tz=timezone.utc)
    items: list[dict] = []
    for item in ig_paginate_items(f"{IG_USER_ID}/media", params=params):
        ts = item.get("timestamp")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue
        if dt >= end_dt:
            continue
        if dt < start_dt:
            break
        items.append(item)
    return items

HASH_RE = re.compile(r"#(\w+)", flags=re.UNICODE)

def fetch_media_counts_for_day(media_items: list[dict] | None = None) -> tuple[int, dict | None, int | None]:
    """
    Devuelve:
      - media_published_day (int)
      - content_mix_day (dict por media_product_type)
      - hashtags_used_day (int total en captions del dia)
    """
    items = media_items if media_items is not None else _list_media_for_day()
    count = len(items)
    mix: dict[str, int] = {}
    hashtags_total = 0

    for item in items:
        mpt = (item.get("media_product_type") or "UNKNOWN").upper()
        mix[mpt] = mix.get(mpt, 0) + 1

        cap = item.get("caption") or ""
        hashtags_total += len(HASH_RE.findall(cap))

    return count, (mix or None), (hashtags_total if hashtags_total > 0 else None)

def fetch_reels_watch_time_for_day(media_items: list[dict] | None = None) -> int | None:
    """
    Suma video_view_total_time (ms) de los reels publicados durante DATE_UTC.
    """
    items = media_items if media_items is not None else _list_media_for_day()
    total = 0
    has_value = False

    for item in items:
        product_type = (item.get("media_product_type") or "").upper()
        if product_type != "REELS":
            continue
        media_id = item.get("id")
        if not media_id:
            continue
        try:
            res = ig_get(f"{media_id}/insights", {"metric": "video_view_total_time"})
            data = res.get("data") or []
            if not data:
                continue
            v = pick_last_value(data[0])
            if isinstance(v, (int, float)):
                total += int(v)
                has_value = True
        except Exception as e:
            print(f"[WARN] reel watch time {media_id}: {e}")

    return total if has_value else None

def fetch_story_interactions_for_day(media_items: list[dict] | None = None) -> int | None:
    """
    Suma replies, taps_forward, taps_back, exits (y shares si aplica) para stories del dia.
    """
    items = media_items if media_items is not None else _list_media_for_day()
    total = 0
    has_value = False
    metrics_variants = [
        ["replies", "shares", "taps_forward", "taps_back", "exits"],
        ["replies", "taps_forward", "taps_back", "exits"],
    ]

    for item in items:
        product_type = (item.get("media_product_type") or "").upper()
        media_type = (item.get("media_type") or "").upper()
        if product_type not in ("STORIES",) and media_type != "STORY":
            continue
        media_id = item.get("id")
        if not media_id:
            continue

        res = None
        last_exc: Exception | None = None
        for metrics in metrics_variants:
            joined = ",".join(metrics)
            if not joined:
                continue
            try:
                res = ig_get(f"{media_id}/insights", {"metric": joined})
                break
            except Exception as exc:
                last_exc = exc
        if res is None:
            if last_exc is not None:
                print(f"[WARN] story interactions {media_id}: {last_exc}")
            continue

        for metric_item in res.get("data") or []:
            v = pick_last_value(metric_item)
            if isinstance(v, (int, float)):
                total += int(v)
                has_value = True

    return total if has_value else None

# ================== DB helpers ==================
def upsert_daily_row(row: dict):
    sb.table("ig_account_insights_daily").upsert(row).execute()

def insert_daily_row(row: dict):
    sb.table("ig_account_insights_daily").insert(row).execute()

def select_prev_daily(ig_user_id: str, d: date):
    q = sb.table("ig_account_insights_daily") \
        .select("followers_end_day") \
        .eq("ig_user_id", ig_user_id) \
        .lt("date_utc", d.isoformat()) \
        .order("date_utc", desc=True) \
        .limit(1) \
        .execute()
    data = q.data or []
    return (data[0] if data else None)

def table_supports_snapshot() -> bool:
    """Detecta si existe columna snapshot_at (para modo append-only)."""
    try:
        sb.table("ig_account_insights_daily").select("snapshot_at").limit(1).execute()
        return True
    except Exception:
        return False

def get_last_snapshot_for_day(ig_user_id: str, d_utc: date):
    cols = "ig_user_id,date_utc,snapshot_at," \
           "reach_day,impressions_day,profile_views_day,website_clicks_day," \
           "email_contacts_day,phone_call_clicks_day,get_directions_clicks_day," \
           "followers_end_day,followers_delta_day,media_published_day,content_mix_day," \
           "reels_watch_time_sum_ms_day,story_interactions_day,hashtags_used_day," \
           "collab_posts_day,paid_partnerships_day,link_clicks_day," \
           "audience_country,audience_city,audience_gender_age,audience_locale,audience_growth_quality"
    try:
        q = sb.table("ig_account_insights_daily") \
            .select(cols) \
            .eq("ig_user_id", ig_user_id) \
            .eq("date_utc", d_utc.isoformat()) \
            .order("snapshot_at", desc=True) \
            .limit(1).execute()
        data = q.data or []
        return data[0] if data else None
    except Exception:
        # si no existe snapshot_at, devolver la única fila del día (si existe)
        q = sb.table("ig_account_insights_daily") \
            .select(cols.replace("snapshot_at,","")) \
            .eq("ig_user_id", ig_user_id) \
            .eq("date_utc", d_utc.isoformat()) \
            .limit(1).execute()
        data = q.data or []
        return data[0] if data else None

COMPARE_FIELDS = [
    "reach_day","impressions_day","profile_views_day","website_clicks_day",
    "email_contacts_day","phone_call_clicks_day","get_directions_clicks_day",
    "followers_end_day","followers_delta_day","media_published_day",
    "content_mix_day","reels_watch_time_sum_ms_day","story_interactions_day",
    "hashtags_used_day","collab_posts_day","paid_partnerships_day",
    "link_clicks_day","audience_country","audience_city",
    "audience_gender_age","audience_locale","audience_growth_quality",
]

def _canon(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        try: return int(v)
        except Exception: return v
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    return v

def has_meaningful_change(new_row: dict, last_row: dict | None) -> bool:
    if not last_row:
        return True
    for k in COMPARE_FIELDS:
        if _canon(new_row.get(k)) != _canon(last_row.get(k)):
            return True
    return False

# ================== MAIN ==================
def run_daily_snapshot():
    # Métricas día
    reach_day = fetch_reach_day()
    tv = fetch_total_value_metrics()
    fu = fetch_follow_unfollow_day()

    # Follower count "end of day"
    followers_end_day = fetch_follower_count_day()
    if followers_end_day is None and DATE_UTC == datetime.now(timezone.utc).date():
        # fallback solo para HOY: usa el campo actual del usuario
        try:
            acc = ig_get(f"{IG_USER_ID}", {"fields": "followers_count"})
            followers_end_day = acc.get("followers_count")
        except Exception as e:
            print(f"[WARN] fallback followers_count: {e}")

    # Media del día
    media_items = _list_media_for_day()
    media_published_day, content_mix_day, hashtags_used_day = fetch_media_counts_for_day(media_items)
    reels_watch_time_sum_ms_day = fetch_reels_watch_time_for_day(media_items)
    story_interactions_day = fetch_story_interactions_for_day(media_items)

    # Demografía snapshot
    demo = fetch_audience_demographics()

    # Delta de followers
    followers_delta_day = None
    if isinstance(fu, dict) and ("follows" in fu or "unfollows" in fu):
        try:
            followers_delta_day = int(fu.get("follows", 0)) - int(fu.get("unfollows", 0))
        except Exception:
            followers_delta_day = None
    if followers_delta_day is None and followers_end_day is not None:
        prev = select_prev_daily(IG_USER_ID, DATE_UTC)
        if prev and isinstance(prev.get("followers_end_day"), int):
            followers_delta_day = followers_end_day - prev["followers_end_day"]

    # Taps por tipo
    taps = tv.get("taps_breakdown") or {}
    email_contacts_day        = taps.get("EMAIL") or taps.get("EMAIL_CONTACT")
    phone_call_clicks_day     = taps.get("CALL")  or taps.get("PHONE_CALL")
    get_directions_clicks_day = taps.get("DIRECTIONS") or taps.get("GET_DIRECTIONS")

    # Armar fila
    row = {
        "ig_user_id": IG_USER_ID,
        "date_utc": DATE_UTC.isoformat(),

        "reach_day": reach_day,
        "impressions_day": tv.get("impressions_day"),
        "profile_views_day": tv.get("profile_views_day"),
        "website_clicks_day": tv.get("website_clicks_day"),
        "email_contacts_day": int(email_contacts_day) if isinstance(email_contacts_day, int) else None,
        "phone_call_clicks_day": int(phone_call_clicks_day) if isinstance(phone_call_clicks_day, int) else None,
        "get_directions_clicks_day": int(get_directions_clicks_day) if isinstance(get_directions_clicks_day, int) else None,

        "followers_end_day": int(followers_end_day) if isinstance(followers_end_day, int) else None,
        "followers_delta_day": int(followers_delta_day) if isinstance(followers_delta_day, int) else None,

        "media_published_day": int(media_published_day) if isinstance(media_published_day, int) else None,
        "content_mix_day": content_mix_day,   # dict o None
        "reels_watch_time_sum_ms_day": int(reels_watch_time_sum_ms_day) if isinstance(reels_watch_time_sum_ms_day, int) else None,
        "story_interactions_day": int(story_interactions_day) if isinstance(story_interactions_day, int) else None,
        "hashtags_used_day": int(hashtags_used_day) if isinstance(hashtags_used_day, int) else None,
        "collab_posts_day": None,
        "paid_partnerships_day": None,
        "link_clicks_day": tv.get("link_clicks_day"),

        "audience_country": demo["audience_country"],
        "audience_city": demo["audience_city"],
        "audience_gender_age": demo["audience_gender_age"],
        "audience_locale": demo["audience_locale"],   # quedará None (no soportado en v23)
        "audience_growth_quality": fu if fu else None,  # {"follows": x, "unfollows": y} si vino
    }

    # Modo append-only si existe snapshot_at; si no, upsert
    supports_append = table_supports_snapshot()
    if supports_append:
        row["snapshot_at"] = datetime.now(timezone.utc).isoformat()
        last = get_last_snapshot_for_day(IG_USER_ID, DATE_UTC)
        if ALWAYS_INSERT or has_meaningful_change(row, last):
            insert_daily_row(row)
            print("[INFO] inserted new snapshot row (changes detected or ALWAYS_INSERT)")
        else:
            print("[INFO] no changes for the day; skipped insert")
    else:
        upsert_daily_row(row)
        print("[INFO] upserted row (table has no snapshot_at)")

    return row

if __name__ == "__main__":
    try:
        t0 = time.time()
        print(f"[INFO] Daily snapshot | IG_USER={IG_USER_ID} | api={API_VERSION} | date_utc={DATE_UTC.isoformat()} | tz={ACCOUNT_TZ}")
        row = run_daily_snapshot()
        print(json.dumps({"ok": True, "api_calls": API_CALLS, "elapsed_sec": round(time.time()-t0,2), "date_utc": DATE_UTC.isoformat()}, ensure_ascii=False))
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
