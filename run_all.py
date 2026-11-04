"""
Exécute tout le pipeline MLOps dans l'ordre, en une commande.

    python run_all.py              # pipeline complet (~5 min)
    python run_all.py --fast       # sans GridSearchCV (~1 min)
    python run_all.py --skip-train # données + EDA seulement

Chaque étape est indépendante : si l'une échoue, le script s'arrête et
indique laquelle, pour que tu puisses la relancer seule.
"""

import subprocess
import sys
import time

FAST = "--fast" in sys.argv
SKIP_TRAIN = "--skip-train" in sys.argv

ETAPES = [
    ("Phase 2.1 — Téléchargement du dataset", ["src.data.download_data"]),
    ("Phase 2.2 — Nettoyage des données", ["src.data.clean_data"]),
    ("Phase 2.3 — Analyse exploratoire", ["src.data.eda"]),
    ("Phase 2.4 — Split train/test stratifié", ["src.data.split_data"]),
]

if not SKIP_TRAIN:
    ETAPES += [
        ("Phase 3.1 — Entraînement du modèle",
         ["src.models.train_model"] + (["--fast"] if FAST else [])),
        ("Phase 3.2 — Évaluation et métriques", ["src.models.evaluate_model"]),
    ]


def main() -> int:
    print("=" * 70)
    print("  TP3 — Pipeline MLOps : analyse de sentiment YouTube")
    print("=" * 70)

    debut = time.perf_counter()

    for i, (titre, args) in enumerate(ETAPES, 1):
        print(f"\n{'─' * 70}\n[{i}/{len(ETAPES)}] {titre}\n{'─' * 70}")
        t0 = time.perf_counter()

        resultat = subprocess.run([sys.executable, "-m", *args])

        if resultat.returncode != 0:
            print(f"\nÉCHEC à l'étape : {titre}")
            print(f"Relance-la seule avec :  python -m {args[0]}")
            return resultat.returncode

        print(f"  -> terminé en {time.perf_counter() - t0:.1f} s")

    total = time.perf_counter() - debut
    print(f"\n{'=' * 70}")
    print(f"  Pipeline terminé en {total:.0f} s")
    print("=" * 70)
    print("\nProchaines étapes :")
    print("  1. Lancer les tests   :  pytest")
    print("  2. Démarrer l'API     :  uvicorn src.api.main:app --reload --port 8000")
    print("  3. Ouvrir la doc API  :  http://localhost:8000/docs")
    print("  4. Charger l'extension dans chrome://extensions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
