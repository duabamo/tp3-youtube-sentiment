"""
PHASE 2 — Étape 1 : téléchargement du dataset.

Usage :
    python -m src.data.download_data

Ce script :
  1. télécharge automatiquement le CSV depuis GitHub,
  2. le sauvegarde dans data/raw/reddit.csv,
  3. affiche les statistiques du dataset (volume, distribution des labels).

Le fichier atterrit dans data/raw/ et n'est JAMAIS modifié ensuite :
c'est la source de vérité immuable du pipeline.
"""

import sys
import urllib.request

import pandas as pd

from src.utils.config import DATASET_URL, LABELS, RAW_CSV, ensure_dirs
from src.utils.logger import get_logger

log = get_logger("data.download")


def download(force: bool = False) -> None:
    ensure_dirs()

    if RAW_CSV.exists() and not force:
        log.info("Fichier déjà présent (%s) — téléchargement ignoré.", RAW_CSV.name)
    else:
        log.info("Téléchargement depuis %s ...", DATASET_URL)
        try:
            urllib.request.urlretrieve(DATASET_URL, RAW_CSV)
        except Exception as exc:                      # noqa: BLE001
            log.error("Échec du téléchargement : %s", exc)
            sys.exit(1)
        log.info("Enregistré dans %s (%.1f Mo)",
                 RAW_CSV, RAW_CSV.stat().st_size / 1e6)

    describe()


def describe() -> None:
    """Statistiques demandées par l'énoncé."""
    df = pd.read_csv(RAW_CSV)
    log.info("Colonnes          : %s", list(df.columns))
    log.info("Nombre de lignes  : %d", len(df))
    log.info("Valeurs manquantes: %d", int(df.isna().sum().sum()))

    label_col = "category" if "category" in df.columns else "label"
    counts = df[label_col].value_counts().sort_index()
    log.info("--- Distribution des labels ---")
    for label, n in counts.items():
        name = LABELS.get(int(label), str(label))
        log.info("  %2d (%-7s) : %6d  (%5.2f %%)",
                 int(label), name, n, 100 * n / len(df))

    # Vérification des contraintes de l'énoncé
    assert len(df) >= 1000, "Le dataset doit contenir au moins 1000 commentaires"
    assert counts.min() >= 300, "Il faut au moins 300 commentaires par classe"
    log.info("Contraintes de l'énoncé respectées (>=1000 lignes, >=300/classe).")


if __name__ == "__main__":
    download(force="--force" in sys.argv)
