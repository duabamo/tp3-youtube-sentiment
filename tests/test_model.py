"""
PHASE 7 — Tests du MODÈLE.

Lancement :
    pytest tests/test_model.py -v
"""

import time

import joblib
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.data.clean_data import clean_text
from src.utils.config import MODEL_PATH, TEST_CSV

CLASSES = [-1, 0, 1]


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), "Lancez d'abord : python -m src.models.train_model"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def test_set():
    df = pd.read_csv(TEST_CSV).dropna(subset=["text"])
    return df["text"].astype(str), df["label"]


# ----------------------------------------------------- nettoyage
class TestCleaning:
    def test_supprime_urls(self):
        assert "http" not in clean_text("check this https://youtu.be/abc123 now")

    def test_supprime_mentions(self):
        assert "@someone" not in clean_text("thanks @someone for the help")

    def test_convertit_emojis(self):
        assert "fire" in clean_text("this is 🔥")

    def test_minuscules(self):
        assert clean_text("THIS IS GREAT") == "this is great"

    def test_entree_non_textuelle(self):
        assert clean_text(None) == ""
        assert clean_text(12345) == ""

    def test_idempotent(self):
        """Nettoyer deux fois doit donner le même résultat."""
        txt = "WOW!! https://x.co @bob 🔥🔥 c'est génial"
        assert clean_text(clean_text(txt)) == clean_text(txt)


# ----------------------------------------------------- performance
class TestPerformance:
    def test_accuracy_superieure_a_80(self, model, test_set):
        X, y = test_set
        acc = (model.predict(X) == y).mean()
        assert acc >= 0.80, f"Accuracy {acc:.3f} < 0.80 (critère de l'énoncé)"

    def test_f1_par_classe_superieur_a_075(self, model, test_set):
        X, y = test_set
        f1 = f1_score(y, model.predict(X), average=None, labels=CLASSES)
        for classe, score in zip(CLASSES, f1):
            assert score > 0.75, f"F1 de la classe {classe} = {score:.3f} <= 0.75"

    def test_latence_batch_50_inferieure_a_100ms(self, model, test_set):
        X, _ = test_set
        batch = X.sample(50, random_state=0).tolist()
        model.predict(batch[:5])                     # warm-up
        durations = []
        for _ in range(10):
            t0 = time.perf_counter()
            model.predict(batch)
            durations.append((time.perf_counter() - t0) * 1000)
        moyenne = sum(durations) / len(durations)
        assert moyenne < 100, f"Latence {moyenne:.1f} ms >= 100 ms"


# ----------------------------------------------------- cas limites
class TestCasLimites:
    @pytest.mark.parametrize(
        "texte",
        [
            "ok",                                   # très court
            "a" * 10_000,                           # très long
            "🔥🔥🔥😂😂",                            # emojis seuls
            "!!!???...",                            # ponctuation seule
            "c'est vraiment nice, i love it bezzaf",  # langues mélangées
            "THIS IS THE WORST!!!",                 # majuscules
            "   ",                                  # espaces
            "123456789",                            # chiffres
        ],
    )
    def test_ne_leve_pas_dexception(self, model, texte):
        cleaned = clean_text(texte) or "empty"
        pred = model.predict([cleaned])
        assert int(pred[0]) in CLASSES

    def test_batch_homogene(self, model):
        """Un même texte répété doit donner la même prédiction partout."""
        preds = model.predict(["this video is amazing"] * 20)
        assert len(set(preds)) == 1

    def test_sentiments_evidents(self, model):
        """Garde-fou métier sur des cas non ambigus."""
        assert int(model.predict([clean_text("i love this so much, perfect")])[0]) == 1
        assert int(model.predict([clean_text("i hate this, it is terrible and awful")])[0]) == -1


# ----------------------------------------------------- reproductibilité
class TestReproductibilite:
    def test_predictions_deterministes(self, model):
        textes = ["great content", "boring video", "neutral statement here"]
        assert list(model.predict(textes)) == list(model.predict(textes))

    def test_vectoriseur_present_dans_le_pipeline(self, model):
        assert "tfidf" in model.named_steps
        assert len(model.named_steps["tfidf"].vocabulary_) > 1000
