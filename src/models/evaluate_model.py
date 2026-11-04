"""
PHASE 3 / PHASE 7 — Évaluation détaillée du modèle sauvegardé.

Usage :
    python -m src.models.evaluate_model

Produit :
  - reports/figures/matrice_confusion.png  (effectifs + version normalisée)
  - reports/metrics.json                   (toutes les métriques)
  - vérification des 3 critères de performance de l'énoncé
"""

import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score)

from src.utils.config import (FIGURES_DIR, METRICS_PATH, MODEL_PATH, TEST_CSV,
                              ensure_dirs)
from src.utils.logger import get_logger

log = get_logger("models.evaluate")

CLASSES = [-1, 0, 1]
NAMES = ["négatif", "neutre", "positif"]

# Critères imposés par l'énoncé
SEUIL_ACCURACY = 0.80
SEUIL_F1_CLASSE = 0.75
SEUIL_LATENCE_MS = 100          # pour un batch de 50 commentaires


def run() -> dict:
    ensure_dirs()
    model = joblib.load(MODEL_PATH)
    test = pd.read_csv(TEST_CSV).dropna(subset=["text"])
    X, y = test["text"].astype(str), test["label"]

    y_pred = model.predict(X)
    report = classification_report(y, y_pred, target_names=NAMES, output_dict=True)
    log.info("\n%s", classification_report(y, y_pred, target_names=NAMES, digits=3))

    # ---- Matrice de confusion (brute + normalisée) ------------------------
    cm = confusion_matrix(y, y_pred, labels=CLASSES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(cm, display_labels=NAMES).plot(
        ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
    axes[0].set_title("Matrice de confusion (effectifs)")
    ConfusionMatrixDisplay(
        confusion_matrix(y, y_pred, labels=CLASSES, normalize="true"),
        display_labels=NAMES,
    ).plot(ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f")
    axes[1].set_title("Matrice de confusion (normalisée par ligne)")
    for ax in axes:
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Vérité terrain")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "matrice_confusion.png", dpi=150)
    plt.close(fig)

    # ---- Temps d'inférence sur un batch de 50 ----------------------------
    batch = X.sample(50, random_state=0).tolist()
    model.predict(batch[:5])                       # warm-up
    durations = []
    for _ in range(20):
        t0 = time.perf_counter()
        model.predict(batch)
        durations.append((time.perf_counter() - t0) * 1000)
    latence = {
        "moyenne_ms": round(float(np.mean(durations)), 2),
        "mediane_ms": round(float(np.median(durations)), 2),
        "p95_ms": round(float(np.percentile(durations, 95)), 2),
        "par_commentaire_ms": round(float(np.mean(durations) / 50), 3),
    }
    log.info("Latence batch de 50 : %.1f ms (p95 %.1f ms)",
             latence["moyenne_ms"], latence["p95_ms"])

    # ---- Cas limites (demandés en Phase 7) -------------------------------
    edge_cases = {
        "texte très court": "ok",
        "texte vide après nettoyage": "!!!",
        "emojis seuls": "🔥🔥🔥",
        "texte très long": "this video is amazing " * 200,
        "langues mélangées": "c'est vraiment nice, i love it bezzaf",
        "majuscules + ponctuation": "THIS IS THE WORST THING EVER!!!",
    }
    from src.data.clean_data import clean_text
    edge_results = {}
    for name, raw in edge_cases.items():
        cleaned = clean_text(raw) or "empty"
        pred = int(model.predict([cleaned])[0])
        edge_results[name] = {"prediction": pred, "label": NAMES[CLASSES.index(pred)]}
        log.info("  cas limite %-28s -> %s", name, edge_results[name]["label"])

    # ---- Vérification des critères ---------------------------------------
    acc = float(report["accuracy"])
    f1_par_classe = {n: float(report[n]["f1-score"]) for n in NAMES}
    criteres = {
        "accuracy >= 0.80": {"valeur": round(acc, 4), "ok": acc >= SEUIL_ACCURACY},
        "F1 par classe > 0.75": {
            "valeur": {k: round(v, 4) for k, v in f1_par_classe.items()},
            "ok": all(v > SEUIL_F1_CLASSE for v in f1_par_classe.values()),
        },
        "latence batch 50 < 100 ms": {
            "valeur": latence["moyenne_ms"],
            "ok": latence["moyenne_ms"] < SEUIL_LATENCE_MS,
        },
    }
    for nom, c in criteres.items():
        log.info("[%s] %s = %s", "OK " if c["ok"] else "KO", nom, c["valeur"])

    metrics = {
        "accuracy": round(acc, 4),
        "f1_macro": round(float(f1_score(y, y_pred, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y, y_pred, average="weighted")), 4),
        "par_classe": {
            n: {k: round(float(v), 4) for k, v in report[n].items()} for n in NAMES
        },
        "matrice_confusion": cm.tolist(),
        "latence": latence,
        "cas_limites": edge_results,
        "criteres_enonce": criteres,
        "n_test": int(len(y)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    log.info("Métriques écrites dans %s", METRICS_PATH)
    return metrics


if __name__ == "__main__":
    run()
