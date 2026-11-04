"""
PHASE 6 — Point d'entrée de PRODUCTION (Hugging Face Spaces).

Différences avec src/api/main.py :
  - fichier unique, aucune dépendance au package src/ (le Space ne contient
    que les fichiers strictement nécessaires) ;
  - pas de mode --reload, pas d'outils de développement ;
  - écoute sur le port 7860, standard des Spaces Hugging Face ;
  - la fonction de nettoyage est dupliquée ici VOLONTAIREMENT : elle doit
    rester identique à celle de l'entraînement, et l'image Docker ne copie
    pas tout le dossier src/.

Lancement local pour tester l'image :
    uvicorn app_api:app --host 0.0.0.0 --port 7860
"""

import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

MODEL_PATH = Path(__file__).parent / "models" / "sentiment_model.joblib"
MODEL_VERSION = "1.0.0"
MAX_BATCH_SIZE = 500
LABEL_NAMES = {-1: "négatif", 0: "neutre", 1: "positif"}
START_TIME = time.time()


# ===========================================================================
# Nettoyage — DOIT rester identique à src/data/clean_data.py
# ===========================================================================
RE_URL = re.compile(r"(https?://\S+|www\.\S+)")
RE_MENTION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#(\w+)")
RE_HTML = re.compile(r"<[^>]+>|&\w+;")
RE_EMOJI = re.compile(
    "[\U0001F300-\U0001F9FF\U0001FA70-\U0001FAFF\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF\U00002190-\U000021FF\U0000FE0F]+",
    flags=re.UNICODE,
)
RE_SPECIAL = re.compile(r"[^a-zà-ÿ0-9\s']")
RE_SPACES = re.compile(r"\s+")
EMOJI_MAP = {
    "😀": " happy ", "😃": " happy ", "😄": " happy ", "😁": " happy ",
    "😊": " happy ", "🙂": " happy ", "😍": " love ", "🥰": " love ",
    "❤": " love ", "❤️": " love ", "💖": " love ", "👍": " good ",
    "🔥": " fire ", "😂": " funny ", "🤣": " funny ", "👏": " applause ",
    "😢": " sad ", "😭": " sad ", "😡": " angry ", "🤬": " angry ",
    "👎": " bad ", "💩": " bad ", "😠": " angry ", "🙄": " annoyed ",
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for emoji, word in EMOJI_MAP.items():
        text = text.replace(emoji, word)
    text = text.lower()
    text = RE_URL.sub(" ", text)
    text = RE_MENTION.sub(" ", text)
    text = RE_HASHTAG.sub(r"\1", text)
    text = RE_HTML.sub(" ", text)
    text = RE_EMOJI.sub(" ", text)
    text = RE_SPECIAL.sub(" ", text)
    return RE_SPACES.sub(" ", text).strip()


# ===========================================================================
# Schémas
# ===========================================================================
class PredictBatchRequest(BaseModel):
    comments: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)

    @field_validator("comments")
    @classmethod
    def _non_vide(cls, v):
        if all(not (c or "").strip() for c in v):
            raise ValueError("Le batch ne contient que des chaînes vides.")
        return [(c or "")[:5000] for c in v]


class PredictionItem(BaseModel):
    index: int
    text: str
    sentiment: Literal[-1, 0, 1]
    label: str
    confidence: float


class Statistics(BaseModel):
    total: int
    positif: int
    neutre: int
    negatif: int
    pourcentage_positif: float
    pourcentage_neutre: float
    pourcentage_negatif: float
    score_global: float
    sentiment_dominant: str
    confiance_moyenne: float


class PredictBatchResponse(BaseModel):
    predictions: list[PredictionItem]
    statistics: Statistics
    processing_time_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    n_features: int | None = None
    uptime_seconds: float


# ===========================================================================
# Modèle (chargé une seule fois au démarrage)
# ===========================================================================
MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
        print(f"[startup] modèle chargé : {MODEL_PATH}")
    else:
        print(f"[startup] ERREUR — modèle introuvable : {MODEL_PATH}")
    yield
    print("[shutdown] arrêt de l'API")


app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="TF-IDF + Logistic Regression — labels : -1 négatif, 0 neutre, 1 positif.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error",
                 "detail": str(exc.errors()[0].get("msg", "données invalides"))},
    )


@app.get("/")
async def root():
    return {"service": "YouTube Sentiment Analysis API",
            "version": MODEL_VERSION,
            "endpoints": ["/health", "/predict_batch", "/docs"]}


@app.get("/health", response_model=HealthResponse)
async def health():
    n_features = None
    if MODEL is not None:
        try:
            n_features = len(MODEL.named_steps["tfidf"].vocabulary_)
        except Exception:                                    # noqa: BLE001
            pass
    return HealthResponse(
        status="healthy" if MODEL is not None else "degraded",
        model_loaded=MODEL is not None,
        model_version=MODEL_VERSION,
        n_features=n_features,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.post("/predict_batch", response_model=PredictBatchResponse)
async def predict_batch(payload: PredictBatchRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    t0 = time.perf_counter()
    try:
        cleaned = [clean_text(c) for c in payload.comments]
        empty_idx = {i for i, t in enumerate(cleaned) if len(t.strip()) < 2}
        cleaned = [t if t.strip() else "empty" for t in cleaned]

        preds = np.asarray(MODEL.predict(cleaned))
        if hasattr(MODEL, "predict_proba"):
            confidences = MODEL.predict_proba(cleaned).max(axis=1)
        else:
            margins = MODEL.decision_function(cleaned)
            exp = np.exp(margins - margins.max(axis=1, keepdims=True))
            confidences = (exp / exp.sum(axis=1, keepdims=True)).max(axis=1)
        confidences = np.asarray(confidences, dtype=float)

        for i in empty_idx:
            preds[i] = 0
            confidences[i] = 0.0

        predictions = [
            {"index": i, "text": (payload.comments[i] or "")[:200],
             "sentiment": int(p), "label": LABEL_NAMES[int(p)],
             "confidence": round(float(c), 4)}
            for i, (p, c) in enumerate(zip(preds, confidences))
        ]

        total = len(preds)
        n_pos, n_neu, n_neg = (int((preds == v).sum()) for v in (1, 0, -1))
        pct = lambda n: round(100 * n / total, 2) if total else 0.0   # noqa: E731
        stats = {
            "total": total, "positif": n_pos, "neutre": n_neu, "negatif": n_neg,
            "pourcentage_positif": pct(n_pos), "pourcentage_neutre": pct(n_neu),
            "pourcentage_negatif": pct(n_neg),
            "score_global": round(float(preds.mean()), 4) if total else 0.0,
            "sentiment_dominant": max(
                (("positif", n_pos), ("neutre", n_neu), ("négatif", n_neg)),
                key=lambda x: x[1])[0],
            "confiance_moyenne": round(float(np.mean(confidences)), 4) if total else 0.0,
        }
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence : {exc}")

    return PredictBatchResponse(
        predictions=predictions,
        statistics=stats,
        processing_time_ms=round((time.perf_counter() - t0) * 1000, 2),
        model_version=MODEL_VERSION,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
