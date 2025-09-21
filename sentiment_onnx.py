# sentiment_onnx.py
# ---------------------------------------------
# ONNX Runtime wrapper para sentimiento (ES)
# Modelo base: pysentimiento/robertuito-sentiment-analysis (exportado a ONNX)
# Descarga modelo/tokenizer desde Hugging Face Hub (repo privado) usando HF_TOKEN.
# Devuelve: [(label, score, conf)] con label in {'pos','neg','neu'}
# ---------------------------------------------
import os, re, json, hashlib
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# ---- Descarga/caché desde HF (sin transformers) ----
from huggingface_hub import hf_hub_download, snapshot_download

# === Parámetros (sobre-escribibles por entorno) ===
# ONNX en tu repo privado (ej: krizzcs2/sentiment-robertuito-v1/weights/model.onnx)
HF_REPO_ID   = os.getenv("SENT_HF_REPO_ID", "krizzcs2/sentiment-robertuito-v1").strip()
HF_REVISION  = os.getenv("SENT_HF_REVISION", "").strip() or None  # ej. "v1"
ONNX_FILE    = os.getenv("SENT_ONNX_FILENAME", "weights/model.onnx").strip()

# Opcional: override local (desarrollo) para evitar descargas
LOCAL_ONNX_PATH = os.getenv("SENT_ONNX_PATH", "").strip() or None

# Tokenizer:
#  1) intenta en tu mismo repo HF: tokenizer.json (en raíz o en /weights/)
#  2) fallback: snapshot de "pysentimiento/robertuito-sentiment-analysis"
HF_TOK_REPO  = os.getenv("SENT_TOKENIZER_REPO", HF_REPO_ID).strip()  # por defecto, el mismo repo privado
TOK_FILE1    = os.getenv("SENT_TOKENIZER_FILE", "tokenizer.json").strip()
TOK_FILE2    = os.getenv("SENT_TOKENIZER_FILE_ALT", "weights/tokenizer.json").strip()
FALLBACK_TOK = os.getenv("SENT_FALLBACK_TOKENIZER", "pysentimiento/robertuito-sentiment-analysis").strip()

# Labels (id2label) – intenta config.json en tu repo, si no, usa fallback habitual
CFG_FILE1    = os.getenv("SENT_CONFIG_FILE", "config.json").strip()
CFG_FILE2    = os.getenv("SENT_CONFIG_FILE_ALT", "weights/config.json").strip()

# ORT provider y longitudes
PROVIDER     = os.getenv("SENT_ONNX_PROVIDER", "CPUExecutionProvider").strip()
MAX_LEN      = int(os.getenv("SENT_MAX_LEN", "160"))
DO_PREPROC   = os.getenv("SENT_PREPROCESS", "1") not in ("0", "false", "False")

# === Estado global (lazy) ===
_sess = None
_tok = None
_id2label = None
_label2id = None
_input_names = None

# === Preprocesador ===
try:
    from pysentimiento.preprocessing import preprocess_tweet as _ps_pre
    def _preprocess(text, lang="es"):
        try:    return _ps_pre(text, lang=lang)
        except TypeError:  return _ps_pre(text)
except Exception:
    _re_user = re.compile(r"@\w+")
    _re_url  = re.compile(r"https?://\S+|www\.\S+")
    _re_hash = re.compile(r"#(\w+)")
    _re_rep  = re.compile(r"(.)\1{2,}", re.UNICODE)
    _re_lol  = re.compile(r"(ja){2,}|(je){2,}|(ji){2,}|(jo){2,}|(ju){2,}", re.IGNORECASE)
    try:
        import emoji
        def _emoji_to_words(s: str) -> str:
            t = emoji.demojize(s).replace(":", " ").replace("_", " ").strip()
            return re.sub(r"\s+", " ", f"emoji {t} emoji") if t else s
    except Exception:
        def _emoji_to_words(s: str) -> str: return s
    def _preprocess(text, lang="es"):
        if not text: return text
        t = _re_user.sub("@usuario", str(text))
        t = _re_url.sub("url", t)
        t = _re_hash.sub(lambda m: m.group(1).lower(), t)
        t = _re_lol.sub("jaja", t)
        t = _re_rep.sub(r"\1\1", t)
        t = _emoji_to_words(t)
        return t.strip()

# --------- Helpers HF ----------
def _hf_download(repo_id: str, filename: str) -> str:
    """Descarga a la caché (~/.cache/huggingface/hub) y retorna ruta local."""
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=os.getenv("HF_TOKEN"),
        repo_type="model",
        revision=HF_REVISION,
        local_dir=None,
        local_dir_use_symlinks=False,
    )

def _ensure_onnx_path() -> str:
    """Prefiere override local; si no, descarga desde HF (y cachea)."""
    if LOCAL_ONNX_PATH and os.path.exists(LOCAL_ONNX_PATH):
        return LOCAL_ONNX_PATH
    try:
        return _hf_download(HF_REPO_ID, ONNX_FILE)
    except Exception as e:
        raise RuntimeError(f"No se pudo descargar el ONNX desde HF: {HF_REPO_ID}/{ONNX_FILE}. "
                           f"Asegura HF_TOKEN y que el archivo existe. Detalle: {e}")

def _ensure_tokenizer_and_labels():
    """Obtiene tokenizer.json y labels (id2label) desde tu repo HF o fallback."""
    # 1) tokenizer.json
    tok_path = None
    for f in (TOK_FILE1, TOK_FILE2):
        try:
            tok_path = _hf_download(HF_TOK_REPO, f)
            break
        except Exception:
            tok_path = None
    if tok_path is None:
        # fallback: baja tokenizer/config del repo público base
        try:
            repo_dir = snapshot_download(
                repo_id=FALLBACK_TOK,
                allow_patterns=["tokenizer.json", "config.json"],
            )
            cand = os.path.join(repo_dir, "tokenizer.json")
            if os.path.exists(cand):
                tok_path = cand
        except Exception:
            tok_path = None
    if tok_path is None:
        raise FileNotFoundError(
            "No se encontró tokenizer.json ni en tu repo HF ni en el fallback. "
            "Sube tokenizer.json a tu repo privado o habilita acceso al fallback."
        )

    # 2) labels (id2label) desde config.json si existe
    labels = None
    cfg_path = None
    for f in (CFG_FILE1, CFG_FILE2):
        try:
            cfg_path = _hf_download(HF_TOK_REPO, f)
            break
        except Exception:
            cfg_path = None
    if cfg_path is None:
        # intenta del fallback si lo descargamos arriba
        try:
            repo_dir = snapshot_download(
                repo_id=FALLBACK_TOK,
                allow_patterns=["config.json"],
            )
            cand = os.path.join(repo_dir, "config.json")
            cfg_path = cand if os.path.exists(cand) else None
        except Exception:
            cfg_path = None

    if cfg_path and os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            id2label = cfg.get("id2label") or {}
            # orden por índice numérico
            labels = [id2label[str(i)].upper() for i in range(len(id2label))]
        except Exception:
            labels = None

    # fallback razonable (orden típico en robertuito)
    if not labels:
        labels = ["NEG", "NEU", "POS"]

    return tok_path, labels

# --------- Carga perezosa ----------
def _load():
    global _sess, _tok, _id2label, _label2id, _input_names
    if _sess is not None:  # ya cargado
        return

    # 1) Modelo
    onnx_path = _ensure_onnx_path()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _sess = ort.InferenceSession(onnx_path, providers=[PROVIDER], sess_options=so)
    _input_names = {i.name for i in _sess.get_inputs()}

    # 2) Tokenizer + labels
    tok_path, labels = _ensure_tokenizer_and_labels()
    _tok = Tokenizer.from_file(tok_path)
    _tok.enable_truncation(max_length=MAX_LEN)
    pad_id = _tok.token_to_id("[PAD]") or _tok.token_to_id("<pad>") or 0
    _tok.enable_padding(length=MAX_LEN, pad_id=pad_id, pad_token="[PAD]")

    _id2label = {i: lab for i, lab in enumerate(labels)}
    _label2id = {lab: i for i, lab in enumerate(labels)}

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def _encode_batch(texts):
    encs = _tok.encode_batch(texts)
    input_ids = np.array([e.ids for e in encs], dtype="int64")
    attn = np.array([e.attention_mask for e in encs], dtype="int64")
    tti = np.zeros_like(input_ids, dtype="int64")  # para grafos estilo BERT
    return input_ids, attn, tti

# --------- API pública ----------
def sent_es(texts):
    """
    texts: list[str]
    return: list[tuple(label:str|None, score:float|None, conf:float|None)]
      - label ∈ {'pos','neg','neu'} o None si vacío
      - score ≈ P.POS - P.NEG
      - conf = max(P.POS, P.NEG, P.NEU)
    """
    _load()

    idx, feed = [], []
    for i, t in enumerate(texts):
        t = (t or "").strip()
        if not t:
            continue
        if DO_PREPROC:
            t = _preprocess(t, lang="es")
        idx.append(i); feed.append(t)

    out = [(None, None, None) for _ in texts]
    if not feed:
        return out

    input_ids, attention_mask, token_type_ids = _encode_batch(feed)
    feeds = {}
    for name in _input_names:
        if name == "input_ids":        feeds[name] = input_ids
        elif name == "attention_mask": feeds[name] = attention_mask
        elif name == "token_type_ids": feeds[name] = token_type_ids

    logits = _sess.run(None, feeds)[0]
    probs  = _softmax(logits)

    def _find(label):
        return _label2id.get(label.upper(),
               _label2id.get(label.capitalize(),
               _label2id.get(label.lower(), None)))
    i_pos = _find("POS"); i_neg = _find("NEG"); i_neu = _find("NEU")

    for j, row in enumerate(probs):
        pred_idx = int(row.argmax())
        lab_up   = _id2label.get(pred_idx, "NEU")
        ppos = float(row[i_pos]) if i_pos is not None else 0.0
        pneg = float(row[i_neg]) if i_neg is not None else 0.0
        pneu = float(row[i_neu]) if i_neu is not None else 0.0
        conf = max(ppos, pneg, pneu) if (i_pos is not None and i_neg is not None and i_neu is not None) else float(row[pred_idx])
        score = ppos - pneg if (i_pos is not None and i_neg is not None) else (float(row[pred_idx]) - (1.0 - float(row[pred_idx])))
        out[idx[j]] = (lab_up.lower(), round(score, 3), round(conf, 3))
    return out

def sent_one(text: str):
    return sent_es([text])[0]
