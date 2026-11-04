# =============================================================================
# PHASE 6 — Image Docker de production (Hugging Face Spaces)
#
# Bonnes pratiques appliquées :
#   1. image de base légère      : python:3.10-slim (~45 Mo contre ~350 Mo
#                                  pour python:3.10 complet)
#   2. optimisation des layers   : requirements.txt copié AVANT le code, donc
#                                  le layer `pip install` reste en cache tant
#                                  que les dépendances ne changent pas
#   3. copie sélective           : seuls app_api.py et models/ entrent dans
#                                  l'image — pas de data/, pas de notebooks,
#                                  pas de tests
#   4. --no-cache-dir            : pip ne conserve pas les wheels téléchargés
#   5. utilisateur non-root      : exigé par Hugging Face Spaces, et bonne
#                                  pratique de sécurité
#   6. HEALTHCHECK               : l'orchestrateur sait si le conteneur est vivant
# =============================================================================

FROM python:3.10-slim

# Variables d'environnement Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- Layer 1 : dépendances (rarement modifiées -> reste en cache) ----------
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# --- Layer 2 : artefacts du modèle ----------------------------------------
COPY models/sentiment_model.joblib models/sentiment_model.joblib

# --- Layer 3 : code applicatif (souvent modifié -> dernier) ---------------
COPY app_api.py .

# --- Utilisateur non-root (requis par Hugging Face Spaces) ----------------
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Port standard des Spaces Hugging Face
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health').status==200 else 1)"

CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "7860"]
