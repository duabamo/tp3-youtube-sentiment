"""
PHASE 2 — Étape 4 : split train/test reproductible.

Usage :
    python -m src.data.split_data

Deux garanties :
  - `random_state=42`  -> le même split est obtenu à chaque exécution,
    sur n'importe quelle machine. Sans ça, impossible de comparer deux
    expériences : une amélioration de score pourrait venir du hasard du split.
  - `stratify=y`       -> les proportions de chaque classe sont identiques
    dans le train et dans le test. Indispensable avec un dataset déséquilibré,
    sinon le test set peut sous-représenter la classe minoritaire.

Ratio retenu : 80 / 20 (cf. réponses aux questions de réflexion).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import (CLEAN_CSV, LABELS, RANDOM_STATE, TEST_CSV,
                              TEST_SIZE, TRAIN_CSV, ensure_dirs)
from src.utils.logger import get_logger

log = get_logger("data.split")


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    df = pd.read_csv(CLEAN_CSV)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],          # <- proportions conservées
        shuffle=True,
    )

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    log.info("Train : %d lignes -> %s", len(train_df), TRAIN_CSV.name)
    log.info("Test  : %d lignes -> %s", len(test_df), TEST_CSV.name)
    for name, part in (("train", train_df), ("test", test_df)):
        dist = (part["label"].value_counts(normalize=True).sort_index() * 100)
        log.info("  %-5s : %s", name,
                 {LABELS[int(k)]: f"{v:.1f}%" for k, v in dist.items()})
    return train_df, test_df


if __name__ == "__main__":
    run()
