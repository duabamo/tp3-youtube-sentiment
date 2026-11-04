"""
PHASE 4 — Application FastAPI.

Lancement local :
    uvicorn src.api.main:app --reload --port 8000

Documentation interactive : http://localhost:8000/docs
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.predictor import MODEL_VERSION, SentimentPredictor
from src.api.schemas import (HealthResponse, PredictBatchRequest,
                             PredictBatchResponse)
from src.utils.logger import get_logger

log = get_logger("api.main")
START_TIME = time.time()


# --------------------------------------------------------------------------
# Cycle de vie : le modèle est chargé UNE fois au démarrage du serveur.
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Démarrage de l'API — chargement du modèle ...")
    SentimentPredictor.get()
    yield
    log.info("Arrêt de l'API.")


app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description=(
        "API REST d'analyse de sentiment pour les commentaires YouTube.\n\n"
        "Modèle : TF-IDF + Logistic Regression (scikit-learn).\n"
        "Labels : -1 négatif, 0 neutre, 1 positif."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# --------------------------------------------------------------------------
# CORS : sans ça, le navigateur BLOQUE les requêtes venant de l'extension.
# Une page youtube.com (origine A) qui appelle notre API (origine B) déclenche
# la politique same-origin ; le serveur doit explicitement autoriser l'origine
# via les en-têtes Access-Control-Allow-*. allow_origins=["*"] convient ici
# (API publique, pas de cookie d'authentification).
# --------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Gestion d'erreurs centralisée
# --------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    log.warning("Requête invalide sur %s : %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "validation_error",
                 "detail": str(exc.errors()[0].get("msg", "données invalides"))},
    )


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    log.exception("Erreur inattendue sur %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_error", "detail": str(exc)},
    )


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "YouTube Sentiment Analysis API",
        "version": MODEL_VERSION,
        "endpoints": ["/health", "/predict_batch", "/docs"],
    }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health():
    """
    Sonde de santé : utilisée par Hugging Face / un orchestrateur pour savoir
    si le conteneur doit être redémarré, et par l'extension pour afficher
    l'état de l'API avant d'envoyer un batch.
    """
    predictor = SentimentPredictor.get()
    return HealthResponse(
        status="healthy" if predictor.is_ready else "degraded",
        model_loaded=predictor.is_ready,
        model_version=MODEL_VERSION,
        n_features=predictor.n_features,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.post("/predict_batch", response_model=PredictBatchResponse, tags=["prediction"])
async def predict_batch(payload: PredictBatchRequest):
    """
    Analyse un lot de commentaires et retourne, pour chacun, le sentiment
    prédit et la confiance, plus des statistiques agrégées sur le lot.

    Le traitement par batch évite un aller-retour HTTP par commentaire :
    100 commentaires = 1 requête au lieu de 100.
    """
    predictor = SentimentPredictor.get()
    if not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé — réessayez dans quelques secondes.",
        )

    try:
        predictions, stats, elapsed = predictor.predict_batch(payload.comments)
    except Exception as exc:                                   # noqa: BLE001
        log.exception("Échec de la prédiction")
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence : {exc}")

    log.info("predict_batch : %d commentaires en %.1f ms", len(payload.comments), elapsed)
    return PredictBatchResponse(
        predictions=predictions,
        statistics=stats,
        processing_time_ms=elapsed,
        model_version=MODEL_VERSION,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
