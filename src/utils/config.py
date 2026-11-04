"""
Configuration centrale du projet.

Pourquoi un fichier de config ?
  -> Un seul endroit pour les chemins et hyperparamètres. Si on déplace
     le projet ou qu'on change le seuil d'un split, on modifie UNE ligne
     et tout le pipeline suit. C'est un principe MLOps de base :
     la configuration est du code, versionné avec le reste.
"""

from pathlib import Path

# --- Racine du projet (2 niveaux au-dessus de src/utils/) -------------------
ROOT_DIR = Path(__file__).resolve().parents[2]

# --- Dossiers de données ----------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# --- Modèles, logs, rapports ------------------------------------------------
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Fichiers ---------------------------------------------------------------
DATASET_URL = (
    "https://raw.githubusercontent.com/Himanshu-1703/"
    "reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
)
RAW_CSV = RAW_DIR / "reddit.csv"
CLEAN_CSV = PROCESSED_DIR / "reddit_clean.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# --- Paramètres du pipeline -------------------------------------------------
RANDOM_STATE = 42          # reproductibilité : même split à chaque exécution
TEST_SIZE = 0.20           # 80 % train / 20 % test
MIN_TEXT_LENGTH = 3        # on jette les commentaires trop courts
LABELS = {-1: "négatif", 0: "neutre", 1: "positif"}


def ensure_dirs() -> None:
    """Crée tous les dossiers nécessaires s'ils n'existent pas."""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
