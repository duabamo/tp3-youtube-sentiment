# Guide de démarrage — faire tourner le projet sur ta machine

Windows (PowerShell). Les équivalents macOS/Linux sont indiqués quand ils diffèrent.
Compte environ **15 minutes** la première fois, dont 5 de calcul.

---

## Étape 0 — Vérifier Python

```powershell
python --version
```

Il faut **3.10 ou supérieur**. Si la commande n'est pas reconnue ou renvoie une
version plus ancienne, installe Python depuis <https://www.python.org/downloads/>
en cochant **« Add python.exe to PATH »** pendant l'installation.

Si `python` ne marche pas mais `py` oui, remplace `python` par `py` dans toutes
les commandes qui suivent.

---

## Étape 1 — Ouvrir le projet

```powershell
cd "C:\Users\bamoh\Downloads\industrie 4.0\tp3-youtube-sentiment"
```

Les guillemets sont obligatoires : le chemin contient un espace et un point.

---

## Étape 2 — Créer l'environnement virtuel

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Tu dois voir `(venv)` apparaître au début de ta ligne de commande.

> **Si PowerShell refuse** avec « l'exécution de scripts est désactivée », lance
> une fois :
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> puis réessaie l'activation. Alternative sans rien changer : utiliser
> l'invite de commandes classique (`cmd`) et taper `venv\Scripts\activate.bat`.

*macOS/Linux : `python3 -m venv venv && source venv/bin/activate`*

---

## Étape 3 — Installer les dépendances

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Compte 2 à 4 minutes. Si tu vois des avertissements jaunes, ignore-les ; seules
les lignes rouges `ERROR` comptent.

---

## Étape 4 — Lancer le pipeline complet

```powershell
python run_all.py
```

Ce script enchaîne les six étapes : téléchargement du dataset, nettoyage, EDA,
split train/test, entraînement avec GridSearchCV, évaluation. **Environ 5 minutes.**

Pressée ? `python run_all.py --fast` saute l'optimisation des hyperparamètres
(1 minute). Attention : le modèle obtenu est moins bon (~87 % au lieu de 89 %) et
**il écrase le modèle optimisé**. À n'utiliser que pour vérifier que la chaîne
tourne, puis relancer `python run_all.py` en entier.

Tu dois voir à la fin :

```
[OK ] accuracy >= 0.80 = 0.8933
[OK ] F1 par classe > 0.75 = {'négatif': 0.8234, 'neutre': 0.9295, 'positif': 0.9006}
[OK ] latence batch 50 < 100 ms = 1.27
```

### Si tu préfères lancer les étapes une par une

```powershell
python -m src.data.download_data
python -m src.data.clean_data
python -m src.data.eda
python -m src.data.split_data
python -m src.models.train_model
python -m src.models.evaluate_model
```

---

## Étape 5 — Vérifier avec les tests

```powershell
pytest
```

Attendu : **46 passed**. C'est la preuve que le modèle respecte les critères de
l'énoncé et que l'API répond correctement — à montrer pendant la soutenance.

---

## Étape 6 — Démarrer l'API

```powershell
uvicorn src.api.main:app --reload --port 8000
```

Laisse cette fenêtre ouverte, l'API tourne dedans. Puis dans ton navigateur :

- **Documentation interactive** : <http://localhost:8000/docs> — tu peux tester
  `/predict_batch` directement depuis la page, bouton *Try it out*.
- **État du service** : <http://localhost:8000/health>

Pour tester en ligne de commande, **ouvre un second terminal** (le premier est
occupé par le serveur), réactive le venv, puis :

```powershell
curl.exe -X POST http://localhost:8000/predict_batch `
  -H "Content-Type: application/json" `
  -d '{\"comments\":[\"amazing tutorial\",\"worst video ever\"]}'
```

> Sur Windows, `curl` seul est un alias PowerShell qui ne gère pas ces options :
> écris bien **`curl.exe`**. Le plus simple reste `/docs`.

Pour arrêter le serveur : `Ctrl + C` dans sa fenêtre.

---

## Étape 7 — Installer l'extension Chrome

1. Ouvre `chrome://extensions`
2. Active le **Mode développeur** (interrupteur en haut à droite)
3. Clique sur **Charger l'extension non empaquetée**
4. Sélectionne le dossier
   `C:\Users\bamoh\Downloads\industrie 4.0\tp3-youtube-sentiment\chrome-extension`
5. Ouvre une vidéo YouTube **avec des commentaires activés**
6. Clique sur l'icône rouge de l'extension, puis sur **Analyser les commentaires**

L'API doit tourner (étape 6) : la pastille en haut du popup doit être **verte**
(« API en ligne »). Si elle est rouge, le serveur n'est pas démarré ou tourne sur
un autre port.

> **Si rien ne se passe** : recharge la page YouTube (F5) puis réessaie. Les
> commentaires YouTube se chargent au défilement — l'extension scrolle
> automatiquement, laisse-lui 5 à 15 secondes selon le nombre demandé.

---

## Ordre de lancement, en résumé

```
venv activé  →  run_all.py  →  pytest  →  uvicorn (terminal 1)  →  Chrome
```

L'API doit tourner **avant** d'utiliser l'extension.

---

## Problèmes fréquents

| Symptôme | Cause | Solution |
|---|---|---|
| `python n'est pas reconnu` | Python absent du PATH | Réinstaller en cochant « Add to PATH », ou utiliser `py` |
| `Activate.ps1 ne peut pas être chargé` | Politique d'exécution PowerShell | Voir l'encadré de l'étape 2 |
| `ModuleNotFoundError: No module named 'src'` | Lancé depuis le mauvais dossier | Se placer à la racine du projet, celle qui contient `run_all.py` |
| `ModuleNotFoundError: fastapi` | venv non activé | Vérifier que `(venv)` est affiché, sinon réactiver |
| `InconsistentVersionWarning` au chargement du modèle | Version de scikit-learn différente de celle d'entraînement | Relancer `python -m src.models.train_model` pour régénérer le `.joblib` avec ta version |
| `[Errno 48] address already in use` | Port 8000 déjà pris | Utiliser `--port 8001` et modifier `API_BASE` dans `chrome-extension/js/popup.js` |
| Pastille rouge dans le popup | API non démarrée ou mauvais port | Vérifier <http://localhost:8000/health> dans le navigateur |
| « Aucun commentaire trouvé » | Commentaires désactivés, ou page pas encore chargée | Essayer une autre vidéo, recharger la page |
| « Could not establish connection » | Page ouverte avant l'installation de l'extension | Recharger la page YouTube (F5) |

---

## Une subtilité à connaître : les versions de scikit-learn

Un modèle `.joblib` est un objet Python sérialisé. Il porte la trace de la
version de scikit-learn qui l'a créé. Le recharger avec une version différente
donne au mieux un avertissement, au pire des prédictions fausses sans aucune
erreur visible.

Le modèle livré a été entraîné avec **scikit-learn 1.9.1**. `requirements.txt`
utilise des plages de versions, donc pip installera peut-être autre chose chez
toi selon ta version de Python.

C'est sans conséquence si tu exécutes l'étape 4 : **le pipeline régénère le
`.joblib` avec tes propres versions**, et le problème disparaît.

En revanche, si tu déploies sur Hugging Face après avoir réentraîné en local,
pense à mettre à jour `requirements-prod.txt` avec tes versions exactes :

```powershell
pip freeze | findstr /I "scikit-learn numpy scipy joblib"
```

C'est exactement le genre de détail qui fait qu'une API marche en local et casse
en production — un bon point à mentionner en soutenance.

---

## Pour la démo / la soutenance

Ordre qui montre le plus de choses en peu de temps :

1. `python run_all.py` — le pipeline complet, avec les trois critères validés à la fin
2. `pytest` — les 46 tests au vert
3. `uvicorn ...` puis `/docs` — l'API documentée automatiquement, un appel en direct
4. L'extension sur une vraie vidéo YouTube — filtres, mode sombre, export JSON
5. `docs/RAPPORT_TECHNIQUE.pdf` — matrice de confusion et justification des choix

Le déploiement Hugging Face est décrit séparément dans `deploy/DEPLOIEMENT.md`.
