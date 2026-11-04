"""
PHASE 3 — Entraînement et optimisation du modèle.

Usage :
    python -m src.models.train_model              # pipeline complet
    python -m src.models.train_model --fast       # sans GridSearch (rapide)

Déroulé :
  1. Baseline        : TF-IDF + LogisticRegression (paramètres par défaut)
  2. Comparaison     : LogisticRegression / LinearSVC / MultinomialNB / RandomForest
  3. Optimisation    : GridSearchCV sur le meilleur candidat (5-fold, scoring=f1_macro)
  4. Sauvegarde      : models/sentiment_model.joblib + tfidf_vectorizer.joblib

Gestion du déséquilibre : `class_weight="balanced"`. Le poids de chaque classe
devient inversement proportionnel à sa fréquence, donc une erreur sur la classe
minoritaire (négatif, 23 %) coûte plus cher qu'une erreur sur la majoritaire.
C'est préférable à un sur-échantillonnage : pas de duplication de texte,
donc pas de sur-apprentissage sur des commentaires répétés.
"""

import json
import sys
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.utils.config import (MODEL_PATH, MODELS_DIR, RANDOM_STATE, REPORTS_DIR,
                              TEST_CSV, TRAIN_CSV, VECTORIZER_PATH, ensure_dirs)
from src.utils.logger import get_logger

log = get_logger("models.train")


# ---------------------------------------------------------------------------
def build_vectorizer() -> TfidfVectorizer:
    """
    TF-IDF plutôt que CountVectorizer :
      - IDF pénalise les mots présents partout ("the", "video") qui n'apportent
        aucune information discriminante ;
      - la normalisation L2 rend comparables un commentaire de 5 mots et un de
        200 mots — crucial ici où la longueur varie de 2 à 1300 mots.

    ngram_range=(1,2) : les bigrammes capturent les négations ("not good"),
    que les unigrammes seuls interprètent à l'envers.
    """
    return TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,              # un mot vu une seule fois = bruit
        max_df=0.9,            # un mot présent dans 90 % des docs = non discriminant
        sublinear_tf=True,     # 1+log(tf) : atténue les répétitions
        strip_accents="unicode",
    )


CANDIDATES = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "LinearSVC": LinearSVC(
        class_weight="balanced", random_state=RANDOM_STATE),
    "MultinomialNB": MultinomialNB(),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, n_jobs=-1, class_weight="balanced",
        random_state=RANDOM_STATE),
}

PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "clf__C": [0.5, 1.0, 5.0, 10.0],
}


# ---------------------------------------------------------------------------
def compare_models(X_train, y_train, X_test, y_test) -> dict:
    """Étape 2 : on entraîne chaque candidat et on compare le F1-macro."""
    results = {}
    for name, clf in CANDIDATES.items():
        pipe = Pipeline([("tfidf", build_vectorizer()), ("clf", clf)])
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        y_pred = pipe.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        acc = (y_pred == y_test).mean()
        results[name] = {"f1_macro": round(float(f1), 4),
                         "accuracy": round(float(acc), 4),
                         "fit_time_s": round(fit_time, 1)}
        log.info("%-20s | acc=%.4f | f1_macro=%.4f | %.1fs",
                 name, acc, f1, fit_time)
    return results


def optimize(X_train, y_train) -> GridSearchCV:
    """Étape 3 : GridSearchCV 5-fold sur la régression logistique."""
    pipe = Pipeline([
        ("tfidf", build_vectorizer()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])
    grid = GridSearchCV(
        pipe, PARAM_GRID, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1,
    )
    log.info("GridSearchCV : %d combinaisons x 5 folds ...",
             len(PARAM_GRID["tfidf__ngram_range"]) * len(PARAM_GRID["tfidf__min_df"])
             * len(PARAM_GRID["clf__C"]))
    t0 = time.perf_counter()
    grid.fit(X_train, y_train)
    log.info("Terminé en %.0f s. Meilleurs paramètres : %s",
             time.perf_counter() - t0, grid.best_params_)
    log.info("Meilleur F1-macro (CV) : %.4f", grid.best_score_)
    return grid


# ---------------------------------------------------------------------------
def run(fast: bool = False) -> dict:
    ensure_dirs()
    train_df = pd.read_csv(TRAIN_CSV).dropna(subset=["text"])
    test_df = pd.read_csv(TEST_CSV).dropna(subset=["text"])
    X_train, y_train = train_df["text"].astype(str), train_df["label"]
    X_test, y_test = test_df["text"].astype(str), test_df["label"]
    log.info("Train=%d  Test=%d", len(X_train), len(X_test))

    comparison = compare_models(X_train, y_train, X_test, y_test)

    if fast:
        best_pipe = Pipeline([
            ("tfidf", build_vectorizer()),
            ("clf", LogisticRegression(max_iter=1000, C=5.0,
                                       class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]).fit(X_train, y_train)
        best_params, cv_score = {"mode": "fast"}, None
    else:
        grid = optimize(X_train, y_train)
        best_pipe = grid.best_estimator_
        best_params = {k: str(v) for k, v in grid.best_params_.items()}
        cv_score = round(float(grid.best_score_), 4)

    # --- Évaluation finale sur le test set ---------------------------------
    y_pred = best_pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True,
                                   target_names=["négatif", "neutre", "positif"])
    log.info("\n%s", classification_report(
        y_test, y_pred, target_names=["négatif", "neutre", "positif"]))

    # --- Sauvegarde --------------------------------------------------------
    # On sauvegarde le pipeline COMPLET (vectoriseur + classifieur) : impossible
    # d'oublier d'appliquer le TF-IDF à l'inférence. On exporte aussi le
    # vectoriseur seul, comme demandé par l'énoncé.
    joblib.dump(best_pipe, MODEL_PATH, compress=3)
    joblib.dump(best_pipe.named_steps["tfidf"], VECTORIZER_PATH, compress=3)
    log.info("Modèle sauvegardé : %s (%.1f Mo)",
             MODEL_PATH.name, MODEL_PATH.stat().st_size / 1e6)

    summary = {
        "comparaison_modeles": comparison,
        "meilleurs_hyperparametres": best_params,
        "f1_macro_cv": cv_score,
        "test": {
            "accuracy": round(float(report["accuracy"]), 4),
            "f1_macro": round(float(report["macro avg"]["f1-score"]), 4),
            "par_classe": {
                c: {k: round(float(v), 4) for k, v in report[c].items()}
                for c in ("négatif", "neutre", "positif")
            },
        },
        "n_features": int(len(best_pipe.named_steps["tfidf"].vocabulary_)),
    }
    (REPORTS_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    run(fast="--fast" in sys.argv)
