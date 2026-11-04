"""
PHASE 2 — Étape 2 : nettoyage des données.

Usage :
    python -m src.data.clean_data

Transformations appliquées (dans cet ordre) :
  1. harmonisation des colonnes  -> 'text' / 'label'
  2. suppression des NaN et des doublons exacts
  3. passage en minuscules
  4. suppression des URLs        (http://…, www…)
  5. suppression des mentions    (@pseudo) et des hashtags (#tag -> tag)
  6. suppression des balises HTML (&amp;, <br>, …) laissées par YouTube
  7. conversion des emojis en texte neutre (on garde le signal, pas le glyphe)
  8. suppression des caractères spéciaux et de la ponctuation résiduelle
  9. normalisation des espaces
 10. filtrage des textes trop courts (< 3 caractères)

IMPORTANT : la même fonction `clean_text` est importée par l'API.
Le texte vu à l'inférence subit donc EXACTEMENT le même traitement
qu'à l'entraînement — c'est ce qui évite le "training/serving skew".
"""

import re
import sys

import pandas as pd

from src.utils.config import (CLEAN_CSV, MIN_TEXT_LENGTH, RAW_CSV, ensure_dirs)
from src.utils.logger import get_logger

log = get_logger("data.clean")

# --- Expressions régulières compilées une seule fois (performance) ---------
RE_URL = re.compile(r"(https?://\S+|www\.\S+)")
RE_MENTION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#(\w+)")
RE_HTML = re.compile(r"<[^>]+>|&\w+;")
RE_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001F9FF"   # pictogrammes, émoticônes
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000027BF"   # symboles divers
    "\U0001F1E6-\U0001F1FF"   # drapeaux
    "\U00002190-\U000021FF"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)
RE_SPECIAL = re.compile(r"[^a-zà-ÿ0-9\s']")
RE_SPACES = re.compile(r"\s+")

# Emojis les plus fréquents -> mot-clé porteur de sentiment.
# On ne les jette pas : dans un commentaire YouTube, l'emoji EST souvent
# le seul porteur du sentiment ("🔥🔥🔥").
EMOJI_MAP = {
    "😀": " happy ", "😃": " happy ", "😄": " happy ", "😁": " happy ",
    "😊": " happy ", "🙂": " happy ", "😍": " love ", "🥰": " love ",
    "❤": " love ", "❤️": " love ", "💖": " love ", "👍": " good ",
    "🔥": " fire ", "😂": " funny ", "🤣": " funny ", "👏": " applause ",
    "😢": " sad ", "😭": " sad ", "😡": " angry ", "🤬": " angry ",
    "👎": " bad ", "💩": " bad ", "😠": " angry ", "🙄": " annoyed ",
}


def clean_text(text: str) -> str:
    """Nettoie un commentaire brut. Utilisée à l'entraînement ET à l'inférence."""
    if not isinstance(text, str):
        return ""

    for emoji, word in EMOJI_MAP.items():      # 7. emojis -> mots
        text = text.replace(emoji, word)

    text = text.lower()                        # 3.
    text = RE_URL.sub(" ", text)               # 4.
    text = RE_MENTION.sub(" ", text)           # 5.
    text = RE_HASHTAG.sub(r"\1", text)
    text = RE_HTML.sub(" ", text)              # 6.
    text = RE_EMOJI.sub(" ", text)             # emojis restants
    text = RE_SPECIAL.sub(" ", text)           # 8.
    text = RE_SPACES.sub(" ", text).strip()    # 9.
    return text


def run() -> pd.DataFrame:
    ensure_dirs()
    df = pd.read_csv(RAW_CSV)
    n0 = len(df)
    log.info("Chargé %d lignes brutes.", n0)

    # 1. harmonisation des noms de colonnes
    df = df.rename(columns={"clean_comment": "text", "category": "label"})
    df = df[["text", "label"]]

    # 2. NaN + doublons
    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"])
    log.info("Après dropna + dédoublonnage : %d lignes (-%d).", len(df), n0 - len(df))

    # 3-9. nettoyage textuel
    df["text"] = df["text"].astype(str).map(clean_text)

    # 10. filtrage des textes trop courts
    before = len(df)
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]
    df = df[df["text"].str.split().str.len() >= 2]
    log.info("Après filtrage des textes courts : %d lignes (-%d).",
             len(df), before - len(df))

    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([-1, 0, 1])].reset_index(drop=True)

    df.to_csv(CLEAN_CSV, index=False)
    log.info("Dataset nettoyé écrit dans %s (%d lignes).", CLEAN_CSV, len(df))
    log.info("Distribution finale :\n%s", df["label"].value_counts().sort_index())
    return df


if __name__ == "__main__":
    sys.exit(0 if run() is not None else 1)
