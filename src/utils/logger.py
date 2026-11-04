"""
Logger partagé par tous les scripts du pipeline.

Pourquoi logger plutôt que print() ?
  -> En production, on a besoin de savoir QUAND un événement a eu lieu,
     de quel module il vient et avec quelle gravité. Les logs sont
     écrits à la fois en console et dans logs/pipeline.log, ce qui permet
     de rejouer un incident après coup (monitoring/observabilité MLOps).
"""

import logging
import sys

from src.utils.config import LOGS_DIR

LOGS_DIR.mkdir(parents=True, exist_ok=True)

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:            # évite les doublons si réimporté
        return logger

    logger.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FORMAT))

    file_handler = logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_FORMAT))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
