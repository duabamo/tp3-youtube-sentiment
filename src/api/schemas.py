"""
PHASE 4 — Schémas Pydantic (validation des données).

Rôle de Pydantic ici :
  - rejeter automatiquement une requête mal formée AVANT qu'elle n'atteigne
    le modèle (batch vide, 5000 commentaires d'un coup, mauvais type) ;
  - documenter l'API : FastAPI génère /docs à partir de ces classes ;
  - typer la réponse, donc garantir un contrat stable pour l'extension Chrome.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_BATCH_SIZE = 500
MAX_COMMENT_LENGTH = 5000


class PredictBatchRequest(BaseModel):
    comments: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="Liste des commentaires bruts à analyser (1 à 500).",
        json_schema_extra={"example": ["Great video!", "I hate this", "ok"]},
    )

    @field_validator("comments")
    @classmethod
    def _non_vide(cls, v: list[str]) -> list[str]:
        if all(not (c or "").strip() for c in v):
            raise ValueError("Le batch ne contient que des chaînes vides.")
        return [(c or "")[:MAX_COMMENT_LENGTH] for c in v]


class PredictionItem(BaseModel):
    index: int
    text: str = Field(description="Commentaire original tronqué à 200 caractères.")
    sentiment: Literal[-1, 0, 1]
    label: Literal["négatif", "neutre", "positif"]
    confidence: float = Field(ge=0.0, le=1.0)


class Statistics(BaseModel):
    total: int
    positif: int
    neutre: int
    negatif: int
    pourcentage_positif: float
    pourcentage_neutre: float
    pourcentage_negatif: float
    score_global: float = Field(
        description="Score de -1 (très négatif) à +1 (très positif)."
    )
    sentiment_dominant: str
    confiance_moyenne: float


class PredictBatchResponse(BaseModel):
    predictions: list[PredictionItem]
    statistics: Statistics
    processing_time_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    model_loaded: bool
    model_version: str
    n_features: int | None = None
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: str
