# -*- coding: utf-8 -*-
"""
coment_media.py (ONNX)
- Extrae comentarios (y replies) del último mes (Instagram Graph API v23.0).
- Calcula sentimiento (ES) con ONNX (batch) SOLO cuando falta en DB.
- Inserta nuevos comentarios en public.ig_media_comments (UPSERT).
- Para existentes, solo actualiza columnas de sentimiento (UPDATE).
- Rollup a la fila de insights más reciente vía RPC (si hubo cambios):
    ig_rollup_comments_to_insights_media(_media_id)

Permisos recomendados:
  instagram_basic, instagram_manage_comments

Requisitos:
  pip install requests python-dotenv supabase onnxruntime transformers "optimum[onnxruntime]"

.env.local (una por línea, sin comillas):
  IG_USER_ID=1784...
  IG_ACCESS_TOKEN=EAAB...
  SUPABASE_URL=https://<tu-proyecto>.supabase.co
  SUPABASE_SERVICE_ROLE_KEY=eyJ...

Opcionales (.env.local):
  MAX_MEDIA=200
  COMMENTS_PAGE_LIMIT=50
  UPSERT_BATCH=500
  COMMENTS_API_BUDGET=2000
  ALWAYS_ROLLUP=0
  SENT_MODEL=onnx-robertuito-int8          # etiqueta para DB
"""

import os, sys, time, requests
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from supabase import create_client, Client

# ONNX wrapper (ya lo probaste)
from sentiment_onnx import sent_es

# =================== Carga .env ===================
load_dotenv(".env.local", override=True)

IG_USER_ID  = (os.getenv("IG_USER_ID") or "").strip()
IG_TOKEN    = (os.getenv("IG_ACCESS_TOKEN") or os.getenv("IG_TOKEN") or "").strip()
SB_URL      = (os.getenv("SUPABASE_URL") or "").strip()
SB_KEY      = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
assert IG_USER_ID and IG_TOKEN and SB_URL and SB_KEY, "Faltan variables base en .env.local"

GRAPH_VER   = "v23.0"
BASE        = f"https://graph.facebook.com/{GRAPH_VER}"

# =================== Parámetros ===================
MAX_MEDIA       = int(os.getenv("MAX_MEDIA", "200"))
PAGE_LIMIT       = int(os.getenv("COMMENTS_PAGE_LIMIT", "50"))
UPSERT_BATCH     = int(os.getenv("UPSERT_BATCH", "500"))
API_BUDGET       = int(os.getenv("COMMENTS_API_BUDGET", "2000"))
SENT_MODEL_NAME  = os.getenv("SENT_MODEL", "onnx-robertuito-int8").strip()
ALWAYS_ROLLUP    = int(os.getenv("ALWAYS_ROLLUP", "0"))

# Ventana temporal: últimos 30 días
NOW_UTC     = datetime.now(timezone.utc)
CUTOFF_UTC  = NOW_UTC - timedelta(days=30)

sb: Client = create_client(SB_URL, SB_KEY)
API_CALLS = 0

# =================== Utilidades IG ===================
def to_utc_iso(ts: str) -> str:
    """Normaliza timestamp ISO de IG a ISO UTC (con Z)."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        # fallback por si viene con milisegundos u offset extraño
        try:
            dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")
        except Exception:
            dt = datetime.fromisoformat(ts)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def ig_get(path_or_url: str, params=None, raw=False) -> dict:
    """GET con token; soporta URL 'next' con raw=True (inyectando token si falta)."""
    global API_CALLS
    if API_CALLS >= API_BUDGET:
        raise RuntimeError(f"Budget de llamadas agotado ({API_BUDGET}).")

    if raw:
        url = path_or_url
        if "access_token=" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}access_token={IG_TOKEN}"
        r = requests.get(url, timeout=30)
    else:
        params = dict(params or {})
        params["access_token"] = IG_TOKEN
        r = requests.get(f"{BASE}/{path_or_url}", params=params, timeout=30)

    API_CALLS += 1
    if r.status_code == 429:
        # simple backoff
        time.sleep(60)
        r = requests.get(r.url, timeout=30)
        API_CALLS += 1
    if r.status_code >= 400:
        raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}")
    return r.json()

def probe_token_basic():
    """Chequeo mínimo de cuenta IG."""
    r = requests.get(f"{BASE}/{IG_USER_ID}",
                     params={"fields":"id,username","access_token":IG_TOKEN},
                     timeout=20)
    if r.status_code != 200:
        print(f"[FATAL] /{IG_USER_ID} falló: {r.status_code} {r.text[:250]}")
        sys.exit(1)
    print("[OK] Cuenta IG accesible:", r.json().get("username"))

def list_media_last_month(limit_per_page=100, max_pages=10):
    """Lista medios del último mes con campos básicos (incluye comments_count)."""
    fields = "id,media_type,media_product_type,timestamp,comments_count,permalink"
    items, pages = [], 0
    data = ig_get(f"{IG_USER_ID}/media", {"fields": fields, "limit": limit_per_page})
    while True:
        for m in data.get("data", []):
            ts_iso = to_utc_iso(m["timestamp"])
            ts = datetime.fromisoformat(ts_iso.replace("Z","+00:00"))
            if ts >= CUTOFF_UTC:
                m["_ts_iso"] = ts_iso
                items.append(m)
        next_url = (data.get("paging") or {}).get("next")
        if not next_url or pages + 1 >= max_pages:
            break
        data = ig_get(next_url, raw=True)
        pages += 1
    items.sort(key=lambda x: x["_ts_iso"], reverse=True)
    return items

# =================== DB Helpers ===================
COMMENT_ROLLUP_FIELDS = ["comments_total", "comments_pos", "comments_neu", "comments_neg", "sentiment_avg_score"]

def upsert_rows(table: str, rows: List[dict], conflict: str) -> int:
    """UPSERT por lotes."""
    if not rows:
        return 0
    total = 0
    for i in range(0, len(rows), UPSERT_BATCH):
        chunk = rows[i:i+UPSERT_BATCH]
        sb.table(table).upsert(chunk, on_conflict=conflict).execute()
        total += len(chunk)
    return total

def update_sentiment_only(pairs: List[dict]) -> int:
    """UPDATE solo columnas de sentimiento, por comment_id (evita filas nuevas incompletas)."""
    if not pairs:
        return 0
    updated = 0
    for i in range(0, len(pairs), UPSERT_BATCH):
        chunk = pairs[i:i+UPSERT_BATCH]
        # actualiza uno por uno para no pisar otros campos
        for data in chunk:
            cid = data.pop("comment_id")
            resp = sb.table("ig_media_comments").update(data).eq("comment_id", cid).execute()
            updated += len(resp.data or [])
    return updated

def select_existing_sentiment(comment_ids: List[str]) -> Dict[str, Dict]:
    """Devuelve {comment_id: {...}} incluyendo label/score/conf para decidir si falta."""
    if not comment_ids:
        return {}
    res = sb.table("ig_media_comments") \
            .select("comment_id,sentiment_model,sentiment_label,sentiment_score,sentiment_conf") \
            .in_("comment_id", comment_ids) \
            .execute()
    return {row["comment_id"]: row for row in (res.data or [])}

def has_sentiment(rec: Dict) -> bool:
    """True si la fila ya tiene los 4 campos de sentimiento poblados."""
    return bool(
        (rec.get("sentiment_model")) and
        (rec.get("sentiment_label") in ("pos","neg","neu")) and
        (rec.get("sentiment_score") is not None) and
        (rec.get("sentiment_conf") is not None)
    )


def latest_insights_missing_rollup(media_id: str) -> bool:
    """True si la snapshot mas reciente en ig_media_insights tiene campos de comentarios vacios."""
    try:
        res = (
            sb.table("ig_media_insights")
            .select("taken_at," + ",".join(COMMENT_ROLLUP_FIELDS))
            .eq("media_id", media_id)
            .order("taken_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as e:
        print(f"[WARN] rollup check fallo media={media_id}: {e}")
        return False
    rows = res.data or []
    if not rows:
        return False
    latest = rows[0]
    return any(latest.get(field) is None for field in COMMENT_ROLLUP_FIELDS)

# =================== Comentarios & Sentimiento ===================
FIELDS_COM = "id,text,username,timestamp,like_count"

def paginate_comments(media_id: str) -> List[dict]:
    """Top-level comments de un media (paginado completo)."""
    params = {"fields": FIELDS_COM, "limit": PAGE_LIMIT}
    items, pages = [], 0
    try:
        data = ig_get(f"{media_id}/comments", params=params)
    except Exception as e:
        print(f"[WARN] /{media_id}/comments falló: {e}")
        return items

    while True:
        items.extend(data.get("data", []))
        next_url = (data.get("paging") or {}).get("next")
        if not next_url:
            break
        data = ig_get(next_url, raw=True)
        pages += 1
        if pages > 200:
            break
    return items

def paginate_replies(comment_id: str) -> List[dict]:
    """Replies (1 nivel) para un comentario."""
    params = {"fields": FIELDS_COM, "limit": PAGE_LIMIT}
    items, pages = [], 0
    try:
        data = ig_get(f"{comment_id}/replies", params=params)
    except Exception as e:
        print(f"[WARN] /{comment_id}/replies falló: {e}")
        return items

    while True:
        items.extend(data.get("data", []))
        next_url = (data.get("paging") or {}).get("next")
        if not next_url:
            break
        data = ig_get(next_url, raw=True)
        pages += 1
        if pages > 200:
            break
    return items

# ---- ONNX (batch): helper
def analyze_texts_es(texts: List[str]) -> List[Tuple[Optional[str], Optional[float], Optional[float], Optional[str]]]:
    """
    Batch ONNX -> [(label, score, conf, model)]
      - label: 'pos' | 'neu' | 'neg' | None
      - score: P(pos) - P(neg)
      - conf : max(POS, NEU, NEG)
    """
    preds = sent_es(texts)  # [(label, score, conf)]
    out = []
    for lab, score, conf in preds:
        if lab is None:
            out.append((None, None, None, None))
        else:
            out.append((lab, float(score), float(conf), SENT_MODEL_NAME))
    return out

def sync_media_comments_with_sentiment(media_id: str) -> Tuple[int, int, int]:
    """
    Inserta/actualiza comentarios + sentimiento SOLO si falta.
    Retorna: (nuevos, updates_sentimiento, nuevos_replies)
    Hace rollup a insights SOLO si hubo cambios (o ALWAYS_ROLLUP=1).
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # ---- 1) Top-level comments
    top = paginate_comments(media_id)
    top_ids = [c["id"] for c in top]
    existing_top = select_existing_sentiment(top_ids)

    new_rows, senti_updates = [], []

    # 1.A seleccionar los que faltan sentimiento
    need_ids_top, need_texts_top = [], []
    for c in top:
        cid  = c["id"]
        text = c.get("text") or ""
        exists = existing_top.get(cid)
        need_sent = (not exists) or (not has_sentiment(exists))
        if need_sent:
            need_ids_top.append(cid)
            need_texts_top.append(text)

    pred_map_top: Dict[str, Tuple[str,float,float,str]] = {}
    if need_texts_top:
        preds = analyze_texts_es(need_texts_top)
        pred_map_top = {cid: preds[i] for i, cid in enumerate(need_ids_top)}

    # 1.B construir filas nuevas y updates
    for c in top:
        cid  = c["id"]
        text = c.get("text") or ""
        exists = existing_top.get(cid)
        need_sent = (not exists) or (not has_sentiment(exists))

        label = score = conf = model = None
        if cid in pred_map_top:
            label, score, conf, model = pred_map_top[cid]

        if not exists:
            new_rows.append({
                "comment_id": cid,
                "media_id": media_id,
                "parent_id": None,
                "user_id": None,
                "username": c.get("username"),
                "text": text,
                "like_count": c.get("like_count"),
                "hidden": None,
                "lang_code": "es",
                "timestamp_utc": to_utc_iso(c["timestamp"]) if c.get("timestamp") else now_iso,
                "detected_at": now_iso,
                "sentiment_label": label,
                "sentiment_score": score,
                "sentiment_conf": conf,
                "sentiment_model": model,
            })
        elif need_sent and label is not None:
            senti_updates.append({
                "comment_id": cid,
                "sentiment_label": label,
                "sentiment_score": score,
                "sentiment_conf": conf,
                "sentiment_model": model,
            })

    # ---- 2) Replies (1 nivel) para todos los top-level (acumulado)
    replies_accum: List[dict] = []
    for c in top:
        rid_list = paginate_replies(c["id"])
        for r in rid_list:
            r["parent_id"] = c["id"]
            replies_accum.append(r)

    rep_ids = [r["id"] for r in replies_accum]
    existing_rep = select_existing_sentiment(rep_ids)

    # 2.A seleccionar los replies que faltan
    need_ids_rep, need_texts_rep = [], []
    for r in replies_accum:
        rid  = r["id"]
        text = r.get("text") or ""
        exists = existing_rep.get(rid)
        need_sent = (not exists) or (not has_sentiment(exists))
        if need_sent:
            need_ids_rep.append(rid)
            need_texts_rep.append(text)

    pred_map_rep: Dict[str, Tuple[str,float,float,str]] = {}
    if need_texts_rep:
        preds = analyze_texts_es(need_texts_rep)
        pred_map_rep = {rid: preds[i] for i, rid in enumerate(need_ids_rep)}

    replies_rows = []
    for r in replies_accum:
        rid  = r["id"]
        text = r.get("text") or ""
        exists = existing_rep.get(rid)
        need_sent = (not exists) or (not has_sentiment(exists))

        label = score = conf = model = None
        if rid in pred_map_rep:
            label, score, conf, model = pred_map_rep[rid]

        if not exists:
            replies_rows.append({
                "comment_id": rid,
                "media_id": media_id,
                "parent_id": r["parent_id"],
                "user_id": None,
                "username": r.get("username"),
                "text": text,
                "like_count": r.get("like_count"),
                "hidden": None,
                "lang_code": "es",
                "timestamp_utc": to_utc_iso(r["timestamp"]) if r.get("timestamp") else now_iso,
                "detected_at": now_iso,
                "sentiment_label": label,
                "sentiment_score": score,
                "sentiment_conf": conf,
                "sentiment_model": model,
            })
        elif need_sent and label is not None:
            senti_updates.append({
                "comment_id": rid,
                "sentiment_label": label,
                "sentiment_score": score,
                "sentiment_conf": conf,
                "sentiment_model": model,
            })

    # ---- 3) Persistencia
    n_new  = upsert_rows("ig_media_comments", new_rows, "comment_id")
    n_new += upsert_rows("ig_media_comments", replies_rows, "comment_id")
    n_senti= update_sentiment_only(senti_updates)

    # ---- 4) Rollup -> insights (force if latest snapshot missing values)
    should_rollup = bool(ALWAYS_ROLLUP) or (n_new > 0 or n_senti > 0)
    if not should_rollup:
        should_rollup = latest_insights_missing_rollup(media_id)
    if should_rollup:
        try:
            r = sb.rpc("ig_rollup_comments_to_insights_media", {"_media_id": media_id}).execute()
            print(f"[ROLLUP] media={media_id} filas_afectadas={r.data}")
        except Exception as e:
            print(f"[WARN] rollup falló media={media_id}: {e}")

    return n_new, n_senti, len(replies_rows)

# =================== RUN ===================
if __name__ == "__main__":
    print(f"[INFO] Comentarios + sentimiento + rollup | v={GRAPH_VER}")
    print("[DBG] IG_USER_ID:", IG_USER_ID)
    print("[DBG] TOKEN len:", len(IG_TOKEN), "…", (IG_TOKEN[-8:] if IG_TOKEN else ""))

    probe_token_basic()
    medias = list_media_last_month()

    if not medias:
        print("[INFO] No hay medios en el último mes.")
        sys.exit(0)

    # Prioriza medios con comments_count > 0
    medias = [m for m in medias if (m.get("comments_count") or 0) > 0] or medias
    medias = medias[:MAX_MEDIA]

    print("[INFO] Medios por procesar:", len(medias))
    for m in medias:
        print(f"  - {m['id']} | {m.get('media_product_type')}/{m.get('media_type')} | {m.get('_ts_iso')} | comments_count={m.get('comments_count')}")

    tot_new = tot_senti = tot_rep = 0
    for m in medias:
        mid = m["id"]
        cc  = m.get("comments_count")
        ts  = m.get("_ts_iso")
        print(f"\n[MEDIA] {mid} | {m.get('media_product_type')}/{m.get('media_type')} | {ts} | comments_count={cc}")
        try:
            n_new, n_senti, n_rep = sync_media_comments_with_sentiment(mid)
            tot_new   += n_new
            tot_senti += n_senti
            tot_rep   += n_rep
            print(f"[OK] new={n_new} senti_updates={n_senti} replies_new={n_rep}")
        except Exception as e:
            print(f"[WARN] media={mid} falló: {e}")
        time.sleep(0.05)

    print("\n" + "="*60)
    print(f"FIN: new={tot_new} senti_updates={tot_senti} replies_new={tot_rep} | llamadas HTTP={API_CALLS}")
    print("="*60)
