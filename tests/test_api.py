"""
PHASE 7 — Tests de l'API (développement et production).

Lancement :
    pytest tests/test_api.py -v
"""

import time

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ----------------------------------------------------- /health
class TestHealth:
    def test_status_200(self, client):
        assert client.get("/health").status_code == 200

    def test_modele_charge(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"
        assert data["n_features"] > 1000

    def test_contient_la_version(self, client):
        assert "model_version" in client.get("/health").json()


# ----------------------------------------------------- /predict_batch
class TestPredictBatch:
    def test_batch_simple(self, client):
        r = client.post("/predict_batch",
                        json={"comments": ["great video", "awful content"]})
        assert r.status_code == 200
        data = r.json()
        assert len(data["predictions"]) == 2
        assert data["statistics"]["total"] == 2

    @pytest.mark.parametrize("taille", [1, 5, 50, 100, 250])
    def test_differentes_tailles_de_batch(self, client, taille):
        r = client.post("/predict_batch",
                        json={"comments": ["this is a test comment"] * taille})
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == taille

    def test_structure_de_la_reponse(self, client):
        data = client.post("/predict_batch",
                           json={"comments": ["nice"]}).json()
        p = data["predictions"][0]
        assert set(p) == {"index", "text", "sentiment", "label", "confidence"}
        assert p["sentiment"] in (-1, 0, 1)
        assert 0.0 <= p["confidence"] <= 1.0
        assert p["label"] in ("négatif", "neutre", "positif")

    def test_statistiques_coherentes(self, client):
        data = client.post("/predict_batch", json={
            "comments": ["i love it", "i hate it", "it exists", "amazing", "terrible"]
        }).json()
        s = data["statistics"]
        assert s["positif"] + s["neutre"] + s["negatif"] == s["total"] == 5
        assert abs(s["pourcentage_positif"] + s["pourcentage_neutre"]
                   + s["pourcentage_negatif"] - 100) < 0.1
        assert -1.0 <= s["score_global"] <= 1.0
        assert s["sentiment_dominant"] in ("positif", "neutre", "négatif")


# ----------------------------------------------------- erreurs
class TestGestionErreurs:
    def test_batch_vide_rejete(self, client):
        assert client.post("/predict_batch", json={"comments": []}).status_code == 422

    def test_champ_manquant_rejete(self, client):
        assert client.post("/predict_batch", json={}).status_code == 422

    def test_mauvais_type_rejete(self, client):
        assert client.post("/predict_batch",
                           json={"comments": "pas une liste"}).status_code == 422

    def test_batch_trop_grand_rejete(self, client):
        r = client.post("/predict_batch", json={"comments": ["x"] * 501})
        assert r.status_code == 422

    def test_que_des_chaines_vides_rejete(self, client):
        assert client.post("/predict_batch",
                           json={"comments": ["", "  ", ""]}).status_code == 422

    def test_chaine_vide_isolee_acceptee_et_neutre(self, client):
        """Une chaîne vide dans un batch valide ne doit pas faire échouer la requête."""
        data = client.post("/predict_batch",
                           json={"comments": ["great video", ""]}).json()
        assert data["predictions"][1]["sentiment"] == 0

    def test_texte_tres_long_tronque(self, client):
        r = client.post("/predict_batch", json={"comments": ["mot " * 5000]})
        assert r.status_code == 200

    def test_reponse_erreur_structuree(self, client):
        body = client.post("/predict_batch", json={"comments": []}).json()
        assert "error" in body and "detail" in body


# ----------------------------------------------------- performance
class TestTempsDeReponse:
    def test_batch_50_sous_500ms(self, client):
        payload = {"comments": ["this is a youtube comment to analyze"] * 50}
        client.post("/predict_batch", json=payload)            # warm-up
        t0 = time.perf_counter()
        r = client.post("/predict_batch", json=payload)
        elapsed = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200
        assert elapsed < 500, f"{elapsed:.0f} ms (HTTP compris)"
        assert r.json()["processing_time_ms"] < 100

    def test_health_rapide(self, client):
        t0 = time.perf_counter()
        client.get("/health")
        assert (time.perf_counter() - t0) * 1000 < 100


# ----------------------------------------------------- CORS
class TestCORS:
    def test_entetes_cors_presents(self, client):
        r = client.options("/predict_batch", headers={
            "Origin": "https://www.youtube.com",
            "Access-Control-Request-Method": "POST",
        })
        assert r.headers.get("access-control-allow-origin") == "*"
