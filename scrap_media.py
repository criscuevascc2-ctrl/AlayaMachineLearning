# -*- coding: utf-8 -*-
"""
Backfill IG Media (rápido y liviano) + Stories cache en Supabase Storage (TTL 14 días).
Optimizado para:
- NO descargar assets salvo que FALTEN video_duration_sec / media_width / media_height.
- REELS/VIDEO: la duración SOLO se calcula desde un URL de video (jamás thumbnail).
- Carruseles: width/height del primer hijo con datos; duración del primer hijo de video (o máxima si CAROUSEL_DURATION_MODE=max).
- Patch parcial: no sobreescribe con NULL.

Requisitos:
  pip install python-dotenv supabase requests pillow

Permisos IG:
  instagram_basic, instagram_manage_insights (no imprescindible para listar/descargar stories)
"""

import os, time, json, re, unicodedata, struct, mimetypes
from io import BytesIO
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode, urlparse, parse_qs
from pathlib import Path

import requests
from dotenv import load_dotenv
from supabase import create_client, Client

# ================== CONFIG ==================
load_dotenv(".env.local")

IG_USER_ID  = (os.getenv("IG_USER_ID") or "").strip()
IG_TOKEN    = (os.getenv("IG_ACCESS_TOKEN") or "").strip()
SB_URL      = (os.getenv("SUPABASE_URL") or "").strip()
SB_KEY      = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()

CUTOFF_DAYS = int(os.getenv("IG_CUTOFF_DAYS", "30"))
API_BUDGET  = int(os.getenv("IG_API_BUDGET", "180"))
PAGE_LIMIT  = int(os.getenv("IG_PAGE_LIMIT", "100"))
BATCH_SIZE  = int(os.getenv("DB_BATCH_SIZE", "100"))
DEBUG_DIR   = (os.getenv("DEBUG_DIR") or "./_debug").strip()
DEFAULT_LANG = (os.getenv("DEFAULT_LANG") or "es").strip().lower()

# Stories cache
STORIES_BUCKET   = os.getenv("STORIES_BUCKET", "Historias")
STORIES_TTL_DAYS = int(os.getenv("STORIES_TTL_DAYS", "90"))
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "60"))

# Límite opcional para evitar descargar archivos gigantes (0 = sin límite)
PROBE_MAX_BYTES  = int(os.getenv("IG_PROBE_MAX_BYTES", "0"))

# Carrusel: duración rápida (primer video) o exacta (máxima) -> "first" | "max"
CAROUSEL_DURATION_MODE = (os.getenv("CAROUSEL_DURATION_MODE", "first") or "first").lower()

assert IG_USER_ID and IG_TOKEN and SB_URL and SB_KEY, "Faltan variables de entorno."

Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)
BASE = "https://graph.facebook.com/v21.0"

sb: Client = create_client(SB_URL, SB_KEY)
API_CALLS  = 0
INITIAL_COUNT_FIELDS = ("like_count_init", "comments_count_init", "saved_count_init", "shared_count_init")
FROZEN_FIELDS_ONCE = ("video_duration_sec", "media_width", "media_height", "detected_at")
CUTOFF_UTC = datetime.now(timezone.utc) - timedelta(days=CUTOFF_DAYS)

# ================== Campos del list endpoint (MEDIA) ==================
FIELDS_MEDIA = ",".join([
  "id","caption","media_type","media_product_type","timestamp",
  "permalink","media_url","thumbnail_url","like_count","comments_count",
  "is_comment_enabled","username","shortcode","width","height","video_duration",
  "children{id,media_type,media_url,thumbnail_url}",
  "owner{id,username}"
])

# ================== Regex & parse ==================
_WS = "\u00A0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u202F\u205F\u3000"
_PUNCT_STRIP = ".,!?:;…“”\"'»«()[]{}<>"
_RE_HASHTAG = re.compile(r"(?<!\w)#([A-Za-z0-9_ÁÉÍÓÚÜÑáéíóúüñ]+)", re.UNICODE)
_RE_MENTION = re.compile(r"(?<!\w)@([A-Za-z0-9_\.ÁÉÍÓÚÜÑáéíóúüñ]+)", re.UNICODE)

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    for ch in _WS:
        s = s.replace(ch, " ")
    return s

def uniq_clean(seq):
    seen, out = set(), []
    for x in seq:
        x = x.strip(_PUNCT_STRIP)
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def parse_caption_details(caption: str | None) -> dict:
    if not caption:
        return {"hashtags": None, "mentions": None}
    text = normalize_text(caption)
    tags  = uniq_clean(_RE_HASHTAG.findall(text))
    ments = uniq_clean(_RE_MENTION.findall(text))
    return {"hashtags": tags or None, "mentions": ments or None}

# ================== IG helpers ==================
def to_utc_iso(ts: str) -> str:
    try:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc).isoformat()

def ig_get(path: str, params: dict | None = None) -> dict:
    global API_CALLS
    if API_CALLS >= API_BUDGET:
        raise RuntimeError(f"Presupuesto de llamadas IG agotado ({API_BUDGET}).")
    params = dict(params or {})
    params["access_token"] = IG_TOKEN
    url = f"{BASE}/{path}?{urlencode(params, doseq=True)}"
    r = requests.get(url, timeout=30)
    API_CALLS += 1
    if r.status_code == 200:
        return r.json()
    try:
        payload = r.json()
    except Exception:
        payload = {"text": r.text}
    raise RuntimeError(f"IG GET {r.status_code}: {payload}")

def ig_paginate_items(path: str, params: dict | None = None):
    params = dict(params or {})
    while True:
        data = ig_get(path, params)
        for it in data.get("data", []):
            yield it
        next_url = (data.get("paging") or {}).get("next")
        if not next_url:
            break
        qs = parse_qs(urlparse(next_url).query)
        after = (qs.get("after") or [None])[0]
        if not after:
            break
        params["after"] = after

# STORIES activas (solo 24h)
STORY_FIELDS = "id,media_type,media_url,thumbnail_url,timestamp"

def list_active_stories():
    params = {"fields": STORY_FIELDS, "limit": 100}
    for s in ig_paginate_items(f"{IG_USER_ID}/stories", params=params):
        yield s

# ================== Supabase DB helpers ==================
def upsert_ig_media(rows: list[dict]):
    if not rows:
        return
    sb.table("ig_media").upsert(rows, on_conflict="media_id").execute()

def select_existing_subset(ids: list[str]) -> dict:
    if not ids:
        return {}
    res = sb.table("ig_media") \
        .select("media_id,hashtags,mentions,caption_lang,video_duration_sec,media_width,media_height,storage_path,media_product_type,timestamp_utc,media_url,media_url_refreshed_at") \
        .in_("media_id", ids).execute()
    out = {}
    for row in (res.data or []):
        out[row["media_id"]] = row
    return out

# ================== PROBE (RAM) Imágenes/Vídeos ==================
try:
    from PIL import Image
    _PIL_OK = True
except Exception:
    _PIL_OK = False

def _is_image_ctype(ct: str | None) -> bool:
    if not ct: return False
    ct = ct.split(";")[0].strip().lower()
    return ct in ("image/jpeg","image/jpg","image/png","image/webp")

def _is_video_ctype(ct: str | None) -> bool:
    if not ct: return False
    ct = ct.split(";")[0].strip().lower()
    return ct in ("video/mp4","video/quicktime")

def download_bytes_with_ctype(url: str, need_bytes: bool = True) -> tuple[bytes, str|None]:
    # HEAD para Content-Length (si falla, seguimos)
    try:
        hr = requests.head(url, timeout=10, allow_redirects=True)
        cl = hr.headers.get("Content-Length")
        if PROBE_MAX_BYTES > 0 and cl and cl.isdigit() and int(cl) > PROBE_MAX_BYTES:
            raise ValueError(f"Asset demasiado grande ({cl} bytes) > PROBE_MAX_BYTES={PROBE_MAX_BYTES}")
    except Exception:
        pass

    r = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
    r.raise_for_status()
    content = r.content if need_bytes else b""
    ctype = (r.headers.get("Content-Type") or None)
    if PROBE_MAX_BYTES > 0 and len(content) > PROBE_MAX_BYTES:
        raise ValueError(f"Asset descargado {len(content)} bytes > PROBE_MAX_BYTES={PROBE_MAX_BYTES}")
    return content, ctype

def _probe_image_dims_pillow(b: bytes) -> tuple[int|None,int|None]:
    with Image.open(BytesIO(b)) as im:
        w, h = im.size
    return int(w), int(h)

def _probe_png_dims(b: bytes) -> tuple[int|None,int|None]:
    if len(b) >= 24 and b[:8] == b"\x89PNG\r\n\x1a\n":
        w = int.from_bytes(b[16:20], "big", signed=False)
        h = int.from_bytes(b[20:24], "big", signed=False)
        if w > 0 and h > 0: return w, h
    return None, None

def _probe_jpeg_dims(b: bytes) -> tuple[int|None,int|None]:
    i, n = 0, len(b)
    if n < 2 or b[:2] != b"\xff\xd8":
        return None, None
    i = 2
    while i+4 <= n:
        if b[i] != 0xFF:
            i += 1; continue
        marker = b[i+1]
        i += 2
        if i+2 > n:
            break
        seglen = int.from_bytes(b[i:i+2], "big")
        if seglen < 2 or i+seglen > n:
            break
        if (0xC0 <= marker <= 0xC3) or (0xC5 <= marker <= 0xC7) or (0xC9 <= marker <= 0xCB) or (0xCD <= marker <= 0xCF):
            if seglen >= 7:
                h = int.from_bytes(b[i+3:i+5], "big")
                w = int.from_bytes(b[i+5:i+7], "big")
                return (w if w>0 else None, h if h>0 else None)
        i += seglen
    return None, None

def probe_image_dims(b: bytes) -> tuple[int|None,int|None]:
    if _PIL_OK:
        try:
            return _probe_image_dims_pillow(b)
        except Exception:
            pass
    w, h = _probe_png_dims(b)
    if w and h: return w, h
    w, h = _probe_jpeg_dims(b)
    if w and h: return w, h
    return None, None

def _u32(b: bytes, off: int) -> int:
    return struct.unpack_from(">I", b, off)[0]

def _u64(b: bytes, off: int) -> int:
    return struct.unpack_from(">Q", b, off)[0]

def _fixed_16_16_to_float(u32val: int) -> float:
    return u32val / 65536.0

def _iter_boxes(b: bytes, start: int, end: int):
    i = start
    n = end
    while i + 8 <= n:
        size = _u32(b, i)
        typ  = b[i+4:i+8]
        hdr  = 8
        if size == 1:
            if i + 16 > n: break
            size = _u64(b, i+8)
            hdr  = 16
        if size == 0:
            size = n - i
        if size < hdr or i + size > n:
            break
        yield i, size, typ, i + hdr
        i += size

def _find_child(b: bytes, start: int, end: int, want: bytes):
    for off, size, typ, payload in _iter_boxes(b, start, end):
        if typ == want:
            return (off, off+size, payload)
    return None

def _find_path(b: bytes, path: list[bytes]):
    cur = (0, len(b), 0)
    start, end, _ = cur
    idx = 0
    while idx < len(path):
        hit = _find_child(b, start, end, path[idx])
        if not hit:
            return None
        start, end, payload = hit
        start, end = payload, end
        idx += 1
    return (start, end, start)

def _parse_mvhd_duration(b: bytes) -> float|None:
    node = _find_path(b, [b"moov", b"mvhd"])
    if not node: return None
    start, end, payload = node
    if payload + 4 > end: return None
    version = b[payload]
    try:
        if version == 0:
            # mvhd v0 layout: version|flags (4) + creation (4) + modification (4) + timescale (4) + duration (4)
            if payload + 20 > end:
                return None
            timescale = _u32(b, payload + 12)
            duration  = _u32(b, payload + 16)
        else:
            # mvhd v1 layout: version|flags (4) + creation (8) + modification (8) + timescale (4) + duration (8)
            if payload + 32 > end:
                return None
            timescale = _u32(b, payload + 20)
            duration  = _u64(b, payload + 24)
        if timescale and duration:
            return float(duration) / float(timescale)
    except Exception:
        return None
    return None

def _parse_first_video_trak_dims(b: bytes) -> tuple[int|None,int|None]:
    moov = _find_path(b, [b"moov"])
    if not moov: return None, None
    mstart, mend, mpayload = moov
    off = mpayload
    while off < mend:
        hits = list(_iter_boxes(b, off, mend))
        if not hits: break
        progressed = False
        for boff, bsize, btyp, bpayload in hits:
            if btyp != b"trak":
                continue
            tkhd = _find_child(b, bpayload, boff+bsize, b"tkhd")
            if not tkhd:
                continue
            tstart, tend, tpayload = tkhd
            if tend - 8 >= tpayload:
                try:
                    w_raw = _u32(b, tend - 8)
                    h_raw = _u32(b, tend - 4)
                    w = int(round(_fixed_16_16_to_float(w_raw)))
                    h = int(round(_fixed_16_16_to_float(h_raw)))
                    if w > 0 and h > 0:
                        return w, h
                except Exception:
                    pass
            progressed = True
        if not progressed:
            break
        off = hits[-1][0] + hits[-1][1]
    return None, None

def probe_mp4_info(b: bytes) -> dict:
    try:
        dur = _parse_mvhd_duration(b)
    except Exception:
        dur = None
    try:
        w, h = _parse_first_video_trak_dims(b)
    except Exception:
        w, h = (None, None)
    return {"width": w, "height": h, "duration": (round(dur, 3) if dur is not None else None)}

def probe_url_dims_and_duration(url: str) -> dict:
    """
    Baja el asset a RAM SOLO si se invoca.
    Devuelve {'width','height','duration','ctype','bytes'}
    """
    b, ctype = download_bytes_with_ctype(url, need_bytes=True)
    info = {"width": None, "height": None, "duration": None, "ctype": ctype, "bytes": b}
    if _is_image_ctype(ctype):
        w, h = probe_image_dims(b)
        info["width"], info["height"] = w, h
    elif _is_video_ctype(ctype) or url.lower().endswith((".mp4", ".mov")):
        mp4 = probe_mp4_info(b)
        info.update({k: mp4.get(k) for k in ("width","height","duration")})
    else:
        # Heurística: intenta imagen; si no, mp4
        w, h = probe_image_dims(b)
        if not (w and h):
            mp4 = probe_mp4_info(b)
            info.update({k: mp4.get(k) for k in ("width","height","duration")})
        else:
            info["width"], info["height"] = w, h
    return info

# ================== Storage (Stories) ==================
def ensure_bucket(name: str):
    try:
        buckets = sb.storage.list_buckets()
        if not any(b.get("name") == name for b in (buckets or [])):
            sb.storage.create_bucket(name, public=False)
    except Exception:
        pass

def guess_ext_and_type(url: str, content_type_hdr: str | None) -> tuple[str, str]:
    if content_type_hdr:
        ct = content_type_hdr.split(";")[0].strip().lower()
        if ct == "image/jpeg": return ".jpg", ct
        if ct == "image/png":  return ".png", ct
        if ct == "image/webp": return ".webp", ct
        if ct == "video/mp4":  return ".mp4", ct
        ext = mimetypes.guess_extension(ct) or ""
        return (ext or ".bin"), ct
    path = urlparse(url).path.lower()
    for ext in (".jpg",".jpeg",".png",".webp",".mp4",".mov"):
        if path.endswith(ext):
            return ext, mimetypes.types_map.get(ext, "application/octet-stream")
    return ".bin", "application/octet-stream"

def storage_upload_bytes(bucket_name: str, path: str, content: bytes, content_type: str, upsert=True):
    b = sb.storage.from_(bucket_name)
    options = {"content-type": content_type or "application/octet-stream"}
    if upsert:
        options["upsert"] = "true"
    try:
        return b.upload(path, content, options)
    except TypeError:
        return b.upload(path=path, file=content, file_options=options)

def storage_remove_paths(bucket_name: str, paths: list[str]):
    if not paths: return
    b = sb.storage.from_(bucket_name)
    b.remove(paths)

def cache_story_asset_and_probe(media_id: str, media_url: str) -> tuple[str|None, dict|None]:
    try:
        ensure_bucket(STORIES_BUCKET)
        info = probe_url_dims_and_duration(media_url)
        content, ctype = info["bytes"], (info["ctype"] or "application/octet-stream")
        ext, _ = guess_ext_and_type(media_url, ctype)
        dt = datetime.now(timezone.utc)
        rel_path = f"{IG_USER_ID}/{dt.year:04d}/{dt.month:02d}/{media_id}{ext}"
        storage_upload_bytes(STORIES_BUCKET, rel_path, content, ctype, upsert=True)
        return rel_path, info
    except Exception as e:
        print(f"[WARN] falla al cachear story {media_id}: {e}")
        return None, None

def cleanup_old_story_assets(ttl_days: int = STORIES_TTL_DAYS, max_delete: int = 500) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=ttl_days)).isoformat()
    q = sb.table("ig_media") \
        .select("media_id,storage_path") \
        .eq("media_product_type", "STORIES") \
        .not_.is_("storage_path", "null") \
        .lt("timestamp_utc", cutoff) \
        .limit(max_delete) \
        .execute()
    rows = q.data or []
    paths = [r["storage_path"] for r in rows if r.get("storage_path")]
    if paths:
        storage_remove_paths(STORIES_BUCKET, paths)
        ids = [r["media_id"] for r in rows]
        sb.table("ig_media").update({"storage_path": None}).in_("media_id", ids).execute()
    return len(paths)

# ================== Debug ==================
def dump_debug_json(rows: list[dict], meta: dict | None = None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = Path(DEBUG_DIR) / f"ig_media_backfill_{ts}.json"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ig_user_id": IG_USER_ID,
        "count": len(rows),
        "meta": {
            "cutoff_utc": CUTOFF_UTC.isoformat(),
            "api_version": BASE,
            "api_calls": API_CALLS,
            "page_limit": PAGE_LIMIT,
            "batch_size": BATCH_SIZE,
            **(meta or {})
        },
        "rows": rows
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(out_path)

# ================== TRANSFORM & ENRICH ==================
def _coerce_duration(val) -> float | None:
    if val is None: return None
    try:
        return float(val)
    except Exception:
        return None

def build_row_from_media(m: dict) -> dict:
    ts_utc = to_utc_iso(m["timestamp"])
    children = (m.get("children") or {}).get("data", []) or []
    now_iso = datetime.now(timezone.utc).isoformat()
    parsed = parse_caption_details(m.get("caption"))

    return {
        "media_id": m["id"],
        "ig_user_id": IG_USER_ID,
        "media_type": m.get("media_type"),
        "media_product_type": (m.get("media_product_type") or "FEED"),
        "caption": m.get("caption"),
        "caption_lang": DEFAULT_LANG,
        "permalink": m.get("permalink"),
        "media_url": m.get("media_url"),
        "media_url_refreshed_at": now_iso,
        "thumbnail_url": m.get("thumbnail_url"),
        "like_count_init": m.get("like_count"),
        "comments_count_init": m.get("comments_count"),
        "saved_count_init": m.get("saved_count") or m.get("saved"),
        "shared_count_init": m.get("shared_count") or m.get("shares"),
        "username": m.get("username") or (m.get("owner") or {}).get("username"),
        "is_comment_enabled": m.get("is_comment_enabled"),
        "children_count": len(children) if children else None,
        "children_ids": [c.get("id") for c in children] or None,
        "children_media_types": [c.get("media_type") for c in children] or None,
        "children_media_urls": [c.get("media_url") for c in children] or None,
        "children_thumbnails": [c.get("thumbnail_url") for c in children] or None,
        "video_duration_sec": _coerce_duration(m.get("video_duration")),  # ← normaliza a float
        "media_width": m.get("width"),
        "media_height": m.get("height"),
        "storage_path": None,
        "alt_text": None, "location_id": None, "location_name": None,
        "is_collab_post": None, "collab_usernames": None, "is_paid_partnership": None,
        "hashtags": parsed["hashtags"],
        "mentions": parsed["mentions"],
        "timestamp_utc": ts_utc,
        "detected_at": now_iso,
        "last_seen_at": now_iso,
        "raw_payload": m
    }

def enrich_dims_duration_if_needed(row: dict, existing: dict | None, m: dict):
    """
    SOLO descarga si faltan datos.
    - Duración SOLO desde un URL de video (no thumbnail).
    - Carrusel: dims del primer hijo con datos; duración del primer hijo de video (o máxima).
    """
    e = existing or {}
    need_w = (row.get("media_width")  in (None, 0)) and not e.get("media_width")
    need_h = (row.get("media_height") in (None, 0)) and not e.get("media_height")
    need_d = (row.get("video_duration_sec") in (None, 0, 0.0)) and (e.get("video_duration_sec") in (None, 0, 0.0))

    if not (need_w or need_h or need_d):
        return row

    media_type  = (row.get("media_type") or "").upper()
    product     = (row.get("media_product_type") or "").upper()
    is_carousel = (media_type == "CAROUSEL_ALBUM")
    is_parent_video = (media_type == "VIDEO") or (product == "REELS")

    def apply_info(info, apply_w=True, apply_h=True, apply_d=True):
        if not info: return
        if apply_w and need_w and info.get("width"):  row["media_width"]  = int(info["width"])
        if apply_h and need_h and info.get("height"): row["media_height"] = int(info["height"])
        if apply_d and need_d and (info.get("duration") is not None):
            row["video_duration_sec"] = float(info["duration"])

    if is_carousel:
        children = (m.get("children") or {}).get("data", []) or []

        # 1) Width/Height: primer hijo con datos (imagen o video)
        if need_w or need_h:
            for c in children:
                url_c = c.get("media_url") or c.get("thumbnail_url")
                if not url_c:
                    continue
                try:
                    info = probe_url_dims_and_duration(url_c)
                    apply_info(info, apply_w=True, apply_h=True, apply_d=False)
                    if (not need_w or row.get("media_width")) and (not need_h or row.get("media_height")):
                        break
                except Exception:
                    continue

        # 2) Duración: primer video (o máximo) usando SOLO media_url de video
        if need_d:
            if CAROUSEL_DURATION_MODE == "max":
                max_d = None
                for c in children:
                    if (c.get("media_type") or "").upper() != "VIDEO":
                        continue
                    url_v = c.get("media_url")  # NO usar thumbnail
                    if not url_v:
                        continue
                    try:
                        info = probe_url_dims_and_duration(url_v)
                        d = info.get("duration")
                        if d is not None and (max_d is None or d > max_d):
                            max_d = d
                    except Exception:
                        continue
                if max_d is not None:
                    row["video_duration_sec"] = float(max_d)
            else:
                for c in children:
                    if (c.get("media_type") or "").upper() != "VIDEO":
                        continue
                    url_v = c.get("media_url")  # NO usar thumbnail
                    if not url_v:
                        continue
                    try:
                        info = probe_url_dims_and_duration(url_v)
                        apply_info(info, apply_w=False, apply_h=False, apply_d=True)
                        break
                    except Exception:
                        continue
        return row

    # No carrusel (FEED/REELS/VIDEO/IMAGE)
    # Para duración: solo desde media_url si el padre es video (VIDEO/REELS).
    video_url = (m.get("media_url") or row.get("media_url")) if is_parent_video else None

    # Si necesitamos duración y hay URL de video → una sola descarga y aplicamos todo lo posible
    if need_d and video_url:
        try:
            info = probe_url_dims_and_duration(video_url)
            apply_info(info, apply_w=True, apply_h=True, apply_d=True)
            return row
        except Exception:
            pass

    # Si aún faltan width/height, probamos con media_url o thumbnail (solo para dims)
    if (need_w or need_h):
        img_url = row.get("media_url") or m.get("media_url") or row.get("thumbnail_url") or m.get("thumbnail_url")
        if img_url:
            try:
                info = probe_url_dims_and_duration(img_url)
                apply_info(info, apply_w=True, apply_h=True, apply_d=False)  # NUNCA duración desde thumbnail
            except Exception:
                pass
    return row

def prune_nulls_for_patch(row: dict) -> dict:
    return {k: v for k, v in row.items() if v is not None}

def needs_update(existing: dict | None, row: dict) -> bool:
    if not existing:
        return True
    e = existing
    product = (row.get("media_product_type") or e.get("media_product_type") or "").upper()
    if product != "STORIES":
        return True
    if (e.get("hashtags") in (None, [], {})) and row.get("hashtags"): return True
    if (e.get("mentions") in (None, [], {})) and row.get("mentions"): return True
    if (e.get("caption_lang") in (None, "")) and row.get("caption_lang"): return True
    if (e.get("video_duration_sec") in (None, 0, 0.0)) and row.get("video_duration_sec"): return True
    if (e.get("media_width") in (None, 0)) and row.get("media_width"): return True
    if (e.get("media_height") in (None, 0)) and row.get("media_height"): return True
    return False


# ================== BACKFILL MEDIA (FEED/REELS + CAROUSEL) ==================
def backfill_media(max_items: int | None = None) -> int:
    inserted_total = 0
    debug_rows = []
    m_batch = []
    params = {"fields": FIELDS_MEDIA, "limit": PAGE_LIMIT}

    def process_batch(batch):
        nonlocal inserted_total, debug_rows
        if not batch:
            return
        ids = [m["id"] for m in batch]
        existing = select_existing_subset(ids)
        upserts = []

        for m in batch:
            ts_utc = to_utc_iso(m["timestamp"])
            if datetime.fromisoformat(ts_utc) < CUTOFF_UTC:
                continue
            base_row = build_row_from_media(m)
            ex = existing.get(base_row["media_id"])

            if ex:
                detected_at_prev = ex.get("detected_at")
                if detected_at_prev:
                    base_row["detected_at"] = detected_at_prev

            # Enriquecer SOLO si faltan datos (y el existente tampoco los tiene)
            base_row = enrich_dims_duration_if_needed(base_row, ex, m)

            if ex:
                for field in INITIAL_COUNT_FIELDS:
                    base_row.pop(field, None)
                for field in FROZEN_FIELDS_ONCE:
                    prev_val = ex.get(field) if ex else None
                    if field == "detected_at":
                        if prev_val:
                            base_row[field] = prev_val
                        else:
                            base_row.pop(field, None)
                        continue
                    if prev_val not in (None, 0, 0.0):
                        base_row[field] = prev_val
                    elif base_row.get(field) in (None, 0, 0.0):
                        base_row.pop(field, None)

            row_to_send = prune_nulls_for_patch(base_row) if ex else base_row
            if needs_update(ex, row_to_send):
                upserts.append(row_to_send)

        if upserts:
            upsert_ig_media(upserts)
            inserted_total += len(upserts)
            debug_rows.extend(upserts)

    for m in ig_paginate_items(f"{IG_USER_ID}/media", params=params):
        ts_utc = to_utc_iso(m["timestamp"])
        if datetime.fromisoformat(ts_utc) < CUTOFF_UTC:
            print("\n[INFO] Límite de fecha alcanzado.")
            break
        m_batch.append(m)
        if len(m_batch) >= BATCH_SIZE:
            process_batch(m_batch)
            print(f"[DB] Lote procesado. Nuevos/actualizados: {inserted_total}")
            m_batch = []
        if max_items and inserted_total >= max_items:
            break

    if m_batch and (not max_items or inserted_total < max_items):
        process_batch(m_batch)
        print(f"\n[DB] Lote final procesado. Nuevos/actualizados: {inserted_total}")

    if debug_rows:
        print(f"\n[DEBUG] Archivo generado -> {dump_debug_json(debug_rows, meta={'mode':'media'})}")
    else:
        print("\n[INFO] No hubo filas para upsert (todo ya estaba completo).")

    return inserted_total

# ================== STORIES: cache + upsert ==================
def build_row_from_story(s: dict, storage_path: str | None, info: dict | None = None) -> dict:
    ts_utc = to_utc_iso(s["timestamp"])
    now_iso = datetime.now(timezone.utc).isoformat()
    w = h = d = None
    if info:
        w = info.get("width"); h = info.get("height"); d = info.get("duration")
    return {
        "media_id": s["id"],
        "ig_user_id": IG_USER_ID,
        "media_type": s.get("media_type"),
        "media_product_type": "STORIES",
        "caption": None,
        "caption_lang": DEFAULT_LANG,
        "permalink": None,
        "media_url": s.get("media_url"),
        "media_url_refreshed_at": now_iso,
        "thumbnail_url": s.get("thumbnail_url"),
        "like_count_init": None, "comments_count_init": None,
        "saved_count_init": None, "shared_count_init": None,
        "username": None, "is_comment_enabled": None,
        "children_count": None, "children_ids": None, "children_media_types": None,
        "children_media_urls": None, "children_thumbnails": None,
        "video_duration_sec": (float(d) if d is not None else None),
        "media_width": (int(w) if w else None),
        "media_height": (int(h) if h else None),
        "storage_path": storage_path,
        "alt_text": None, "location_id": None, "location_name": None,
        "is_collab_post": None, "collab_usernames": None, "is_paid_partnership": None,
        "hashtags": None, "mentions": None,
        "timestamp_utc": ts_utc,
        "detected_at": now_iso,
        "last_seen_at": now_iso,
        "raw_payload": s
    }

def process_active_stories() -> int:
    added = 0
    active = list(list_active_stories())
    if not active:
        print("[INFO] No hay stories activas.")
        return 0

    existing = select_existing_subset([s["id"] for s in active])
    rows_to_upsert = []

    for s in active:
        prev = existing.get(s["id"])
        have_path = prev.get("storage_path") if prev else None

        info = None
        storage_path = have_path

        need_w = not prev or (prev.get("media_width") in (None, 0))
        need_h = not prev or (prev.get("media_height") in (None, 0))
        need_d = not prev or (prev.get("video_duration_sec") in (None, 0, 0.0))
        need_meta = need_w or need_h or need_d

        media_url = s.get("media_url")

        if not storage_path and media_url:
            try:
                storage_path, info = cache_story_asset_and_probe(s["id"], media_url)
            except Exception:
                info = None
            if info is None and need_meta and media_url:
                try:
                    info = probe_url_dims_and_duration(media_url)
                except Exception:
                    info = None
        elif need_meta and media_url:
            try:
                info = probe_url_dims_and_duration(media_url)
            except Exception: 
                info = None

        row = build_row_from_story(s, storage_path, info)

        if prev:
            for field in INITIAL_COUNT_FIELDS:
                row.pop(field, None)
            for field in FROZEN_FIELDS_ONCE:
                prev_val = prev.get(field) if prev else None
                if field == "detected_at":
                    if prev_val:
                        row[field] = prev_val
                    else:
                        row.pop(field, None)
                    continue
                if prev_val not in (None, 0, 0.0):
                    row[field] = prev_val
                elif row.get(field) in (None, 0, 0.0):
                    row.pop(field, None)

        rows_to_upsert.append(prune_nulls_for_patch(row) if prev else row)

    if rows_to_upsert:
        upsert_ig_media(rows_to_upsert)
        added = len(rows_to_upsert)
        print(f"[DB] Stories activas procesadas: {added}")
        print(f"[DEBUG] Archivo -> {dump_debug_json(rows_to_upsert, meta={'mode':'stories'})}")
    return added

# ================== MAIN ==================
if __name__ == "__main__":
    try:
        start_time = time.time()
        print(f"[INFO] Backfill MEDIA + STORIES cache | IG_USER={IG_USER_ID} | cutoff={CUTOFF_DAYS}d")

        n_media = backfill_media()
        n_sto   = process_active_stories()
        n_clean = cleanup_old_story_assets(STORIES_TTL_DAYS)

        duration = time.time() - start_time
        print("\n" + "="*60)
        print(f"OK: MEDIA upserts={n_media} | STORIES cached/updated={n_sto} | CLEANED={n_clean}")
        print(f"Tiempo total: {duration:.2f}s | Llamadas IG API: {API_CALLS}")
        print("="*60)

    except Exception as e:
        print(f"\n[ERROR] El script falló: {e}")
        raise

