"""
PHASE 4 — Couche d'inférence (singleton).

Pourquoi un singleton ?
  Charger un .joblib prend ~0.3 s. Si on le chargeait à chaque requête,
  l'API passerait 99 % de son temps à lire le disque. Ici le modèle est
  chargé UNE fois au démarrage (événement lifespan de FastAPI) et gardé
  en mémoire : chaque prédiction coûte ensuite ~1 ms.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np

from src.data.clean_data import clean_text
from src.utils.logger import get_logger

log = get_logger("api.predictor")

LABEL_NAMES = {-1: "négatif", 0: "neutre", 1: "positif"}
MODEL_VERSION = "1.0.0"


class SentimentPredictor:
    """Encapsule le pipeline sklearn et la logique métier de l'API."""

    _instance: "SentimentPredictor | None" = None

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model = None
        self.loaded_at: float | None = None
        self.load()

    # -- singleton ---------------------------------------------------------
    @classmethod
    def get(cls, model_path: Path | None = None) -> "SentimentPredictor":
        if cls._instance is None:
            if model_path is None:
                from src.utils.config import MODEL_PATH
                model_path = MODEL_PATH
            cls._instance = cls(model_path)
        return cls._instance

    # -- chargement --------------------------------------------------------
    def load(self) -> None:
        if not self.model_path.exists():
            log.error("Modèle introuvable : %s", self.model_path)
            self.model = None
            return
        t0 = time.perf_counter()
        self.model = joblib.load(self.model_path)
        self.loaded_at = time.time()
        log.info("Modèle chargé en %.0f ms (%s)",
                 (time.perf_counter() - t0) * 1000, self.model_path.name)

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def n_features(self) -> int | None:
        if not self.is_ready:
            return None
        try:
            return len(self.model.named_steps["tfidf"].vocabulary_)
        except Exception:                                   # noqa: BLE001
            return None

    # -- inférence ---------------------------------------------------------
    def predict_batch(self, comments: list[str]) -> tuple[list[dict], dict, float]:
        """Retourne (prédictions, statistiques, temps_ms)."""
        if not self.is_ready:
            raise RuntimeError("Le modèle n'est pas chargé.")

        t0 = time.perf_counter()

        # Même nettoyage qu'à l'entraînement -> pas de training/serving skew.
        cleaned = [clean_text(c) for c in comments]

        # Cas limite : un commentaire vide après nettoyage (emoji seul supprimé,
        # ponctuation pure, spam de caractères) n'a aucun signal exploitable.
        # On le force à "neutre" plutôt que de laisser le modèle inventer une
        # prédiction à partir d'un vecteur TF-IDF nul.
        empty_idx = {i for i, t in enumerate(cleaned) if len(t.strip()) < 2}
        cleaned = [t if t.strip() else "empty" for t in cleaned]

        preds = self.model.predict(cleaned)

        # La confiance : probabilité max si le modèle l'expose,
        # sinon un softmax sur les marges de décision (cas LinearSVC).
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(cleaned)
            confidences = proba.max(axis=1)
        else:
            margins = self.model.decision_function(cleaned)
            exp = np.exp(margins - margins.max(axis=1, keepdims=True))
            confidences = (exp / exp.sum(axis=1, keepdims=True)).max(axis=1)

        preds = np.asarray(preds)
        confidences = np.asarray(confidences, dtype=float)
        for i in empty_idx:                       # neutralisation des cas vides
            preds[i] = 0
            confidences[i] = 0.0

        predictions = [
            {
                "index": i,
                "text": (comments[i] or "")[:200],
                "sentiment": int(p),
                "label": LABEL_NAMES[int(p)],
                "confidence": round(float(c), 4),
            }
            for i, (p, c) in enumerate(zip(preds, confidences))
        ]

        stats = self._statistics(preds, confidences)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return predictions, stats, elapsed_ms

    @staticmethod
    def _statistics(preds, confidences) -> dict:
        preds = np.asarray(preds)
        total = len(preds)
        n_pos = int((preds == 1).sum())
        n_neu = int((preds == 0).sum())
        n_neg = int((preds == -1).sum())
        pct = lambda n: round(100 * n / total, 2) if total else 0.0  # noqa: E731

        # Score global : moyenne des labels -> -1 (hostile) à +1 (enthousiaste)
        score = round(float(preds.mean()), 4) if total else 0.0
        dominant = max(
            (("positif", n_pos), ("neutre", n_neu), ("négatif", n_neg)),
            key=lambda x: x[1],
        )[0]

        return {
            "total": total,
            "positif": n_pos,
            "neutre": n_neu,
            "negatif": n_neg,
            "pourcentage_positif": pct(n_pos),
            "pourcentage_neutre": pct(n_neu),
            "pourcentage_negatif": pct(n_neg),
            "score_global": score,
            "sentiment_dominant": dominant,
            "confiance_moyenne": round(float(np.mean(confidences)), 4) if total else 0.0,
        }
