---
title: YouTube Sentiment API
emoji: 💬
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# YouTube Sentiment Analysis API

API REST d'analyse de sentiment pour les commentaires YouTube.
Projet réalisé dans le cadre du **TP3 — Cloud Computing**, filière INDIA,
ENSAM Rabat (Université Mohammed V).

## Modèle

| | |
|---|---|
| Algorithme | TF-IDF (unigrammes, 22 575 features) + Logistic Regression |
| Entraînement | 28 711 commentaires Reddit labellisés |
| Accuracy (test) | 89,3 % |
| F1-macro (test) | 0,885 |
| Latence | ~1,3 ms pour un batch de 50 commentaires |

Labels : `-1` négatif · `0` neutre · `1` positif

## Endpoints

### `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "n_features": 22575,
  "uptime_seconds": 142.8
}
```

### `POST /predict_batch`

Requête :

```json
{ "comments": ["Great video!", "I hate this", "it is 10 minutes long"] }
```

Réponse :

```json
{
  "predictions": [
    { "index": 0, "text": "Great video!", "sentiment": 1,
      "label": "positif", "confidence": 0.9712 }
  ],
  "statistics": {
    "total": 3, "positif": 1, "neutre": 1, "negatif": 1,
    "pourcentage_positif": 33.33, "score_global": 0.0,
    "sentiment_dominant": "positif", "confiance_moyenne": 0.91
  },
  "processing_time_ms": 4.21,
  "model_version": "1.0.0"
}
```

Limites : 1 à 500 commentaires par requête, 5 000 caractères par commentaire.

### Exemple cURL

```bash
curl -X POST https://VOTRE-USER-yt-sentiment-api.hf.space/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"comments":["amazing tutorial","worst video ever"]}'
```

Documentation interactive : `/docs`
