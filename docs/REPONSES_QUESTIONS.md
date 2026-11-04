# Réponses aux questions de réflexion

TP3 — Cloud Computing · ENSAM Rabat · BAMOHAMED Douaa

---

## Phase 1 — Configuration de l'environnement MLOps

### Pourquoi séparer les données raw et processed ?

**`data/raw/` est la source de vérité immuable.** C'est exactement ce qui a été
téléchargé, jamais modifié. `data/processed/` contient le résultat du pipeline :
nettoyage, filtrage, split.

Trois raisons concrètes :

1. **Reproductibilité.** Si on découvre dans six mois que la regex qui supprime
   les emojis effaçait aussi les apostrophes, on corrige `clean_data.py` et on
   relance le pipeline sur `raw/`. Si on avait écrasé le fichier d'origine,
   l'erreur serait irréversible — il faudrait retélécharger, en espérant que la
   source existe encore et n'ait pas changé.
2. **Traçabilité.** `processed = f(raw, code)`. En versionnant le code et en
   gardant `raw` intact, on peut toujours expliquer d'où vient une ligne du
   dataset d'entraînement. Sans cette séparation, la provenance est perdue.
3. **Expérimentation.** On peut tester trois stratégies de nettoyage différentes
   en parallèle sans jamais toucher à la donnée d'origine.

Dans ce projet : 37 249 lignes brutes → 35 889 lignes nettoyées (−450 doublons
et NaN, −910 textes trop courts). Ces 1 360 lignes supprimées sont une décision
de modélisation, pas une perte de données : elles sont toujours dans `raw/`.

### Quels fichiers ne doivent jamais être commités sur Git ?

| Catégorie | Exemples | Pourquoi |
|---|---|---|
| **Secrets** | `.env`, tokens API, clés SSH, `credentials.json` | Un secret poussé sur GitHub est compromis définitivement — même après `git rm`, il reste dans l'historique. Les bots scannent GitHub en continu. |
| **Données volumineuses** | `data/raw/reddit.csv` (6,6 Mo), datasets | Git stocke chaque version en entier. Un CSV modifié 20 fois = 20 copies dans `.git`. Le dépôt devient ingérable. |
| **Environnements virtuels** | `venv/`, `node_modules/` | Des centaines de Mo, dépendants de l'OS. `requirements.txt` suffit à les reconstruire. |
| **Artefacts compilés** | `__pycache__/`, `*.pyc`, `dist/` | Régénérés automatiquement, sources de conflits de merge inutiles. |
| **Logs** | `logs/*.log` | Grossissent sans fin, contiennent parfois des données utilisateur. |
| **Fichiers d'IDE / OS** | `.vscode/`, `.idea/`, `.DS_Store` | Spécifiques à une machine, polluent les diffs de l'équipe. |

**Cas particulier des modèles.** Les `.joblib` sont des binaires, donc contraires
à l'esprit de Git. Ici `sentiment_model.joblib` ne fait que 0,8 Mo et il est
nécessaire au déploiement : je le versionne. Au-delà de ~50 Mo, il faudrait
passer par **Git LFS**, **DVC** ou un **registre de modèles** (MLflow, Hugging
Face Hub).

### Quelle est l'importance d'un environnement virtuel ?

Il isole les dépendances d'un projet de celles du système et des autres projets.

- **Évite les conflits de versions.** Ce projet exige `scikit-learn==1.5.2`. Un
  autre projet sur la même machine peut exiger la 1.3. Sans venv, l'un des deux
  casse.
- **Garantit la reproductibilité.** `pip freeze` dans un venv liste exactement
  les dépendances du projet. Sur le système global, il listerait tout ce qui
  traîne sur la machine, et `pip install -r` deviendrait ininstallable ailleurs.
- **Prépare la containerisation.** Le `Dockerfile` reproduit la même logique à
  l'échelle de l'OS : partir d'une base propre et installer uniquement ce qui
  est listé. Un venv propre en développement, c'est un `requirements.txt` fiable,
  donc une image Docker qui démarre du premier coup.
- **Préserve le système.** Pas de `pip install` qui écrase une bibliothèque dont
  dépend un outil système.

Point critique en ML : une prédiction peut changer entre deux versions mineures
de scikit-learn. Un modèle sérialisé avec la 1.5.2 et rechargé avec la 1.3
déclenche au mieux un avertissement, au pire des prédictions silencieusement
fausses. D'où les versions **figées** dans `requirements-prod.txt`.

---

## Phase 2 — Pipeline de données et préparation

### Pourquoi nettoyer les URLs et les mentions ?

**Elles ne portent aucun signal de sentiment, mais elles saturent le vocabulaire.**

Chaque URL est une chaîne quasi unique (`youtu.be/dQw4w9WgXcQ`). Sans nettoyage,
le TF-IDF crée un token par URL : des milliers de features vues une seule fois,
qui diluent le vocabulaire utile et alourdissent le modèle.

Pire, elles créent des **corrélations fallacieuses**. Si le spam promotionnel du
dataset est majoritairement labellisé négatif, le modèle apprend « présence de
`http` → négatif ». Il prédira alors « négatif » sur un commentaire enthousiaste
qui partage simplement un lien. C'est du sur-apprentissage sur un artefact.

Même raisonnement pour les mentions `@pseudo` : un pseudo identifie une personne,
pas une émotion. Les hashtags, eux, sont traités différemment — je retire le `#`
et je **garde le mot** (`#amazing` → `amazing`), parce que le mot lui-même porte
du sentiment.

Effet mesuré : le vocabulaire final descend à 22 575 features (avec `min_df=2`,
qui élimine en plus tout ce qui n'apparaît qu'une fois).

### Comment gérer les emojis dans le texte ?

Trois stratégies possibles :

| Stratégie | Avantage | Inconvénient |
|---|---|---|
| Suppression pure | Simple | Perte de signal : « 🔥🔥🔥 » devient vide |
| Conservation brute | Aucune perte | Les variantes Unicode (❤ / ❤️ / 💖) sont des tokens distincts → vocabulaire éclaté |
| **Mapping vers des mots** | Signal conservé et regroupé | Table à maintenir |

**J'ai retenu le mapping** (`EMOJI_MAP` dans `clean_data.py`) : les ~24 emojis les
plus fréquents sont traduits en mots-clés avant le passage en minuscules.
❤, ❤️ et 💖 deviennent tous `love`, donc un seul token au lieu de trois. Les
emojis restants, non mappés, sont supprimés par `RE_EMOJI`.

C'est le bon compromis pour YouTube, où un commentaire se réduit souvent à des
emojis. Un texte qui devient vide après nettoyage est alors forcé à **neutre**
par l'API, plutôt que de laisser le modèle prédire à partir d'un vecteur TF-IDF
nul — c'était un vrai bug avant correction : la chaîne vide ressortait
« négatif » avec 91 % de confiance.

Limite assumée : le dataset d'entraînement (commentaires Reddit) contient très
peu d'emojis, donc le modèle n'a pas appris de poids forts pour `fire` ou `love`.
Le mapping prépare surtout le terrain pour un futur réentraînement sur de vrais
commentaires YouTube.

### Quel ratio train/test est approprié ?

**80 / 20, stratifié, avec `random_state=42`.**

Le ratio arbitre entre deux besoins contradictoires :

- plus de données d'entraînement → meilleur modèle ;
- plus de données de test → estimation de performance plus fiable.

Avec 35 889 commentaires, 20 % donnent **7 178 exemples de test**, dont 1 641
pour la classe minoritaire. C'est largement assez pour que l'intervalle de
confiance sur l'accuracy soit étroit (±0,7 point environ). Monter à 30 % de test
n'améliorerait pas sensiblement cette précision tout en retirant 3 600 exemples
à l'entraînement.

Règle générale : sur un petit dataset (< 1 000 lignes), on prend 70/30 ou on
passe en validation croisée ; sur un très gros dataset (> 1 M), 95/5 suffit.

Deux garanties sont indispensables ici :

- **`stratify=y`** — le dataset est déséquilibré (ratio max/min = 1,91). Sans
  stratification, le test set pourrait sous-représenter la classe négative et
  donner un score trompeur. Vérification : la répartition est identique à 0,1 %
  près entre train et test (22,9 % / 33,4 % / 43,7 %).
- **`random_state=42`** — sans graine fixe, chaque exécution produirait un split
  différent, et on ne saurait jamais si un gain de 1 point vient d'une vraie
  amélioration ou du hasard du tirage.

À noter : le `train_test_split` sert à l'évaluation finale, tandis que le
GridSearchCV utilise une **validation croisée 5-fold sur le train uniquement**.
Le test set n'est touché qu'une seule fois, à la toute fin — sinon on
sur-apprendrait sur lui par le choix des hyperparamètres.

---

## Phase 3 — Développement et entraînement du modèle

### Pourquoi utiliser TF-IDF plutôt que Count Vectorizer ?

`CountVectorizer` compte les occurrences. `TfidfVectorizer` pondère ce compte par
l'inverse de la fréquence documentaire :

```
tfidf(t, d) = tf(t, d) × log(N / df(t))
```

Trois différences décisives :

1. **Pénalisation des mots omniprésents.** « the », « video », « this »
   apparaissent partout : leur IDF tend vers 0, donc leur poids s'effondre.
   Avec CountVectorizer, ils dominent le vecteur par leur simple fréquence, alors
   qu'ils ne distinguent aucune classe.
2. **Normalisation L2.** Chaque document est ramené à une norme de 1. Sans cela,
   un commentaire de 200 mots aurait des valeurs 40× supérieures à un commentaire
   de 5 mots, et la régression logistique interpréterait cette longueur comme un
   signal. Dans notre dataset, la longueur va de 2 à 1 307 mots (médiane 14) —
   la normalisation est donc critique.
3. **Valorisation des mots rares et discriminants.** « masterpiece », « garbage »
   sont peu fréquents globalement mais très informatifs. L'IDF leur donne un poids
   élevé, exactement là où l'information est.

J'ajoute `sublinear_tf=True` (`1 + log(tf)` au lieu de `tf`) : un mot répété
10 fois n'est pas 10 fois plus informatif qu'une seule occurrence.

Mesure sur ce projet : le GridSearchCV a testé unigrammes et bigrammes. Avec un
vocabulaire TF-IDF correctement filtré (`min_df=2`), les **unigrammes seuls**
l'emportent (F1-macro CV 0,873), les bigrammes n'apportant que du bruit sur ce
corpus.

### Comment interpréter la matrice de confusion ?

| Vérité ↓ / Prédit → | négatif | neutre | positif | Total |
|---|---|---|---|---|
| **négatif** | **1 331** | 128 | 182 | 1 641 |
| **neutre** | 29 | **2 322** | 47 | 2 398 |
| **positif** | 232 | 148 | **2 759** | 3 139 |

Lecture : la **diagonale** contient les bonnes prédictions (6 412 / 7 178 = 89,3 %).
Chaque **ligne** est une vraie classe, chaque **colonne** une prédiction.

- **Ligne « négatif »** : rappel = 1 331/1 641 = **81,1 %**. C'est le point faible.
  182 commentaires négatifs sont classés positifs — probablement de l'ironie et du
  sarcasme, que le sac de mots ne peut pas capter (« great, another useless
  update » contient « great »).
- **Ligne « neutre »** : rappel = 2 322/2 398 = **96,8 %**, le meilleur score.
  Les commentaires factuels ont un vocabulaire caractéristique et peu ambigu.
- **Ligne « positif »** : rappel = 2 759/3 139 = **87,9 %**. 232 positifs partent
  en négatif.
- **Colonne « négatif »** : précision = 1 331/(1 331+29+232) = **83,6 %**.

**Confusion dominante : négatif ↔ positif** (182 + 232 = 414 erreurs), contre
seulement 29 + 47 = 76 erreurs impliquant « neutre vers autre chose ». C'est
contre-intuitif : on attendrait que les erreurs portent sur la frontière
neutre/polarisé. L'explication est que les commentaires neutres sont descriptifs
(vocabulaire propre), alors que positif et négatif partagent un vocabulaire
émotionnel intense dont seul le contexte donne la polarité.

**Pourquoi la matrice plutôt que l'accuracy seule ?** Avec 43,5 % de positifs,
un modèle qui prédirait toujours « positif » afficherait 43,5 % d'accuracy sans
rien apprendre. La matrice rend visible ce genre d'effondrement, et surtout elle
indique *quelle* erreur on commet — information indispensable pour décider où
investir (ici : gérer la négation et l'ironie).

### Quels hyperparamètres ont le plus d'impact sur les performances ?

Résultats du GridSearchCV (16 combinaisons × 5 folds) :

| Hyperparamètre | Valeurs testées | Retenue | Impact |
|---|---|---|---|
| `clf__C` | 0,5 · 1 · 5 · **10** | **10,0** | **Fort** — +5 points de F1 entre C=1 et C=10 |
| `tfidf__ngram_range` | (1,1) · (1,2) | **(1,1)** | Moyen |
| `tfidf__min_df` | 1 · **2** | **2** | Faible mais positif |

**1. `C` (inverse de la régularisation) — de loin le plus influent.**
Un `C` faible force les coefficients vers zéro (fort biais, sous-apprentissage) ;
un `C` élevé laisse le modèle exploiter finement le vocabulaire. Avec 22 575
features pour 28 711 exemples, le modèle est en régime haute dimension : la
régularisation par défaut (`C=1`) est trop agressive. Le passage de la
LogisticRegression par défaut (F1 0,835) à la version optimisée (F1 0,885)
vient essentiellement de là.

**2. `ngram_range` — compromis expressivité / bruit.**
Les bigrammes capturent les négations (« not good »), ce qui est précieux pour
le sentiment. Mais ils multiplient le vocabulaire par ~5 et, sur un corpus de
36 k textes, la plupart n'apparaissent que quelques fois. Ici le bruit l'emporte :
`(1,1)` gagne.

**3. `min_df=2` — débruitage.**
Élimine le vocabulaire vu une seule fois (fautes de frappe, pseudos, mots d'une
autre langue) qui ne généralise jamais.

**4. `class_weight="balanced"` — hors grille, mais déterminant.**
Non testé par le GridSearch car imposé dès le départ. Il pondère chaque classe
par l'inverse de sa fréquence, donc une erreur sur « négatif » (22,9 % du
dataset) coûte environ deux fois plus cher qu'une erreur sur « positif ». Sans
lui, le F1 de la classe négative serait nettement sous le seuil de 0,75.
Je l'ai préféré au sur-échantillonnage (SMOTE ou duplication), qui aurait
dupliqué des textes et fait sur-apprendre le modèle sur des commentaires répétés.

**Choix de l'algorithme.** LinearSVC obtenait un meilleur F1 brut (0,868 contre
0,835) que la LogisticRegression par défaut. J'ai quand même retenu la
régression logistique pour deux raisons : une fois optimisée elle passe devant
(0,885), et surtout elle expose `predict_proba`, donc un **score de confiance
calibré** que l'extension affiche pour chaque commentaire. Avec LinearSVC il
aurait fallu passer par un softmax sur les marges de décision — moins
interprétable. Le code gère tout de même ce cas de repli.

---

## Phase 4 — Développement de l'API REST

### Pourquoi FastAPI plutôt que Flask ?

| Critère | FastAPI | Flask |
|---|---|---|
| Validation des entrées | Native via Pydantic | Manuelle ou extension |
| Documentation | `/docs` auto-générée (OpenAPI) | À écrire soi-même |
| Asynchrone | Natif (ASGI) | Nécessite une extension |
| Performance | ~3× Flask (Starlette + Uvicorn) | Correcte |
| Typage | Imposé par les annotations Python | Optionnel |

Trois arguments décisifs **dans ce projet précis** :

1. **La validation est gratuite et déclarative.** Déclarer
   `comments: list[str] = Field(min_length=1, max_length=500)` suffit à rejeter
   automatiquement un batch vide, un batch de 10 000 éléments ou un type
   incorrect, avec un message d'erreur exploitable. En Flask, il faudrait écrire
   et maintenir ces vérifications à la main dans chaque route — et en oublier une
   finit en exception 500.
2. **`/docs` est un livrable.** L'énoncé demande une documentation API avec
   endpoints et exemples de requêtes. FastAPI la génère depuis les schémas
   Pydantic : elle ne peut pas se désynchroniser du code.
3. **Le contrat avec l'extension est typé.** `response_model=PredictBatchResponse`
   garantit la forme exacte du JSON reçu par `popup.js`. Si je modifie l'API sans
   mettre à jour le schéma, l'erreur apparaît côté serveur, pas sous forme de
   `undefined` silencieux dans le JavaScript.

Flask resterait un choix raisonnable pour un prototype jetable. Pour une API de
production consommée par un client distinct, FastAPI évite une quantité de code
défensif.

### À quoi sert la configuration CORS ?

CORS (*Cross-Origin Resource Sharing*) lève, de façon contrôlée, la **politique
de même origine** appliquée par les navigateurs.

Le problème concret : le popup de l'extension s'exécute dans une origine
(`chrome-extension://<id>` ou `https://www.youtube.com`) et appelle l'API sur une
autre (`https://...hf.space`). Par défaut, le navigateur **émet** la requête mais
**refuse de donner la réponse** au JavaScript, et affiche en console :

```
Access to fetch at 'https://...hf.space/predict_batch' from origin
'https://www.youtube.com' has been blocked by CORS policy
```

Important : ce n'est pas le serveur qui bloque, c'est le navigateur. La même
requête en `curl` fonctionne parfaitement — d'où des heures de débogage si on
ignore le mécanisme.

La solution est le middleware qui ajoute les en-têtes
`Access-Control-Allow-Origin`, `-Methods` et `-Headers`. Le navigateur envoie
d'abord une requête **preflight** `OPTIONS` ; si les en-têtes autorisent
l'origine et la méthode, il laisse passer le `POST`.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
)
```

`allow_origins=["*"]` est acceptable ici : l'API est publique, en lecture seule,
sans authentification ni cookie. Dans un contexte avec des données utilisateur,
il faudrait restreindre aux origines légitimes et activer `allow_credentials`
(les deux sont d'ailleurs incompatibles : le wildcard est refusé par les
navigateurs quand les credentials sont activés).

Côté extension, il faut en plus déclarer l'API dans `host_permissions` du
`manifest.json` — sans quoi Chrome bloque l'appel avant même le CORS.

### Comment gérer efficacement le chargement du modèle ?

**Charger une fois au démarrage, garder en mémoire, servir depuis la RAM.**

Le mauvais réflexe est de charger le `.joblib` dans le handler de la route : avec
~300 ms de lecture disque contre ~1 ms d'inférence, l'API passerait 99,7 % de son
temps à relire le même fichier.

Trois mécanismes utilisés :

1. **Le `lifespan` de FastAPI** (successeur de `@app.on_event("startup")`) charge
   le modèle une fois, avant que le serveur accepte la première requête.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    SentimentPredictor.get()      # chargement unique
    yield
```

2. **Un singleton** (`SentimentPredictor._instance`) garantit qu'un seul
   exemplaire existe, même si plusieurs modules l'appellent. La mémoire est
   partagée entre toutes les requêtes.

3. **Le pipeline complet est sérialisé**, vectoriseur inclus. Impossible
   d'oublier d'appliquer le TF-IDF à l'inférence, ou d'utiliser un vectoriseur
   désynchronisé du classifieur — une source classique de bugs silencieux en
   production.

Un quatrième point, moins visible mais tout aussi important : **`clean_text()`
est partagée entre l'entraînement et l'inférence**. Si le texte servi était
nettoyé différemment de celui vu à l'entraînement, le modèle verrait une
distribution décalée — c'est le *training/serving skew*, qui dégrade les
performances sans jamais lever d'erreur. Dans `app_api.py`, la fonction est
dupliquée volontairement (le conteneur ne copie pas `src/`), et un test vérifie
qu'elle reste identique.

Enfin, `/health` expose `model_loaded` et `n_features` : la sonde permet à
Hugging Face de redémarrer un conteneur dont le modèle n'a pas chargé, et à
l'extension d'avertir l'utilisateur avant d'envoyer un batch.

---

## Phase 5 — Développement de l'extension Chrome

### Comment gérer les commentaires avec des réponses (threads) ?

YouTube structure les commentaires en deux niveaux dans le DOM :

```html
<ytd-comment-thread-renderer>      <!-- le fil complet -->
  <ytd-comment-view-model>         <!-- le commentaire racine -->
    <div id="content-text">…</div>
  </ytd-comment-view-model>
  <div id="replies">               <!-- les réponses, repliées par défaut -->
    <ytd-comment-view-model>…</ytd-comment-view-model>
  </div>
</ytd-comment-thread-renderer>
```

Ma stratégie : **extraire tout `#content-text`, mais marquer chaque élément**
selon qu'il se trouve ou non sous `#replies`.

```js
const isReply = !!node.closest("#replies");
```

Ce drapeau laisse le choix de la politique d'agrégation :

- **Tout compter à égalité** (ce que fait l'extension) : simple, et les réponses
  expriment un sentiment réel sur la vidéo.
- **Ne garder que les racines** : évite qu'un fil de 50 réponses hostiles sur un
  détail hors sujet fasse basculer le score global d'une vidéo par ailleurs bien
  reçue.
- **Pondérer** : racine = 1, réponse = 0,5.

J'ai retenu la première par défaut, parce que les réponses ne sont de toute façon
chargées que si l'utilisateur déplie le fil — elles restent donc minoritaires,
et le biais est limité. Le drapeau `isReply` est conservé dans l'export JSON pour
permettre une analyse séparée.

Une subtilité : les réponses sont repliées derrière un bouton « Afficher les N
réponses ». Les extraire toutes exigerait de cliquer chaque bouton, ce qui
multiplie les requêtes réseau de YouTube et ralentit considérablement
l'extraction. J'ai choisi de ne pas les déplier.

### Quelle stratégie pour extraire les commentaires dynamiques ?

YouTube charge les commentaires en **lazy loading** : au chargement de la page,
`#comments` est vide ; les commentaires n'apparaissent qu'au scroll, par paquets
d'environ 20. Un `querySelectorAll` immédiat renvoie zéro résultat.

Ma stratégie, dans `content.js` :

```js
async function loadComments(target = 100) {
  document.querySelector("ytd-comments#comments")
          .scrollIntoView({ behavior: "smooth", block: "start" });
  await sleep(1200);

  let previous = 0, stalled = 0;
  while (stalled < 3) {
    const current = document.querySelectorAll(SELECTOR_COMMENT).length;
    if (current >= target) break;          // objectif atteint
    if (current === previous) stalled++;   // rien de nouveau
    else stalled = 0;
    previous = current;
    window.scrollBy({ top: innerHeight * 1.5, behavior: "smooth" });
    await sleep(900);
  }
  return readComments().slice(0, target);
}
```

Quatre décisions :

1. **Scroll par paliers de 1,5 écran** plutôt qu'un saut en bas de page : YouTube
   ne déclenche le chargement que quand le sentinel entre dans le viewport.
2. **Attente de 900 ms entre paliers** : laisse le temps à la requête réseau
   d'aboutir et au DOM de se mettre à jour.
3. **Compteur `stalled`** : après 3 paliers sans nouveau commentaire, on conclut
   qu'on a atteint la fin de la liste et on s'arrête. Sans ce garde-fou, la
   boucle tournerait indéfiniment sur une vidéo à 12 commentaires.
4. **Double condition de sortie** : objectif atteint *ou* fin de liste — la
   boucle se termine dans tous les cas.

Trois autres alternatives, écartées :

- **`MutationObserver`** : plus élégant sur le papier, mais il faut quand même
  provoquer le scroll, et la gestion de l'arrêt devient plus complexe.
- **L'API YouTube Data v3** : propre et fiable, mais elle exige une clé API, un
  quota journalier et l'identifiant de la vidéo — donc une configuration
  utilisateur. L'énoncé demande explicitement l'extraction depuis la page.
- **Lecture du JSON interne `ytInitialData`** : rapide, mais ce format non
  documenté change sans préavis.

Robustesse : si le content script n'a pas été injecté (page ouverte avant
l'installation de l'extension), `popup.js` envoie un `ping` et, en cas d'échec,
l'injecte à la demande via `chrome.scripting.executeScript`. Cela évite
l'erreur « Could not establish connection » qui obligerait l'utilisateur à
recharger sa page.

### Comment optimiser les performances avec de nombreux commentaires ?

Cinq optimisations, à trois niveaux :

**Réseau — le batching.** C'est le gain principal. 200 commentaires en 200
requêtes HTTP coûteraient 200 × (latence réseau + overhead) ≈ 20 s. En lots de
100, c'est 2 requêtes ≈ 400 ms. L'API est conçue pour cela : `/predict_batch`
accepte jusqu'à 500 commentaires, et `sklearn.predict()` vectorise le lot entier
d'un coup, ce qui est bien plus rapide que 100 appels unitaires.

**Réseau — le timeout.** Un `AbortController` à 20 s évite qu'un popup reste
bloqué indéfiniment sur une API endormie (les Spaces Hugging Face gratuits se
mettent en veille après inactivité et mettent ~30 s à redémarrer).

**Rendu — `DocumentFragment`.** La liste est construite hors du DOM puis insérée
en une seule opération :

```js
const frag = document.createDocumentFragment();
items.forEach(p => frag.appendChild(buildComment(p)));
list.appendChild(frag);          // 1 reflow au lieu de N
```

Avec 200 `appendChild` directs, le navigateur recalcule la mise en page 200 fois.

**Rendu — pas de librairie de graphiques.** Chart.js pèse ~250 Ko et ajouterait
un temps de parsing à chaque ouverture du popup. Le graphique en anneau est
dessiné en ~40 lignes de Canvas 2D natif. Le popup reste instantané et
l'extension n'embarque aucune dépendance tierce — ce qui simplifie aussi la
politique de sécurité de contenu de Manifest V3.

**Rendu — `textContent` plutôt que `innerHTML`.** Un bénéfice double : pas de
parsing HTML (plus rapide) et aucune possibilité d'injection de script depuis un
commentaire malveillant.

**Mesure** (10 commentaires, API locale) : API 12 ms, total 7 ms côté client.
Le goulot d'étranglement réel n'est ni l'inférence ni le rendu, mais
**l'extraction par scroll** — environ 1 s par palier de 20 commentaires imposée
par YouTube. C'est pourquoi le sélecteur du nombre de commentaires est limité à
200 : au-delà, l'attente devient désagréable sans gain analytique.

---

## Phase 6 — Déploiement cloud avec Docker

### Pourquoi séparer les requirements dev et prod ?

`requirements.txt` (développement) contient 13 paquets ; `requirements-prod.txt`
en contient 6. Les absents : `pandas`, `matplotlib`, `seaborn`, `pytest`,
`httpx`, `requests`.

Quatre raisons :

1. **Taille de l'image.** matplotlib + seaborn + pandas pèsent environ 150 Mo
   avec leurs dépendances. Or l'API n'en a aucun besoin : elle manipule des
   listes Python, pas des DataFrames, et ne trace aucun graphique. Une image plus
   légère se télécharge plus vite, démarre plus vite et coûte moins cher à
   stocker.
2. **Surface d'attaque.** Chaque dépendance est du code tiers susceptible de
   contenir une CVE. Un paquet qui n'est pas dans l'image ne peut pas être
   vulnérable.
3. **Temps de build et de démarrage.** Moins de paquets à installer, moins
   d'imports au lancement du conteneur.
4. **Clarté du contrat.** La liste de production documente exactement ce dont
   l'API dépend réellement. Si quelqu'un ajoute un `import pandas` dans
   `app_api.py`, le build échoue immédiatement — un garde-fou utile.

Les versions sont **figées** (`==` et non `>=`) : une image reconstruite dans six
mois produira exactement les mêmes prédictions. Avec `scikit-learn>=1.5`, une
version future pourrait changer subtilement le comportement du modèle sérialisé.

### Comment réduire la taille de l'image Docker ?

Cinq techniques, appliquées dans le `Dockerfile` :

**1. Image de base minimale.**

| Base | Taille |
|---|---|
| `python:3.13` | ~350 Mo |
| **`python:3.13-slim`** | **~45 Mo** |
| `python:3.13-alpine` | ~25 Mo |

J'ai retenu `slim`. L'énoncé suggérait la 3.10 ; j'utilise la **3.13** parce que
scikit-learn 1.9.1 — la version qui a produit le `.joblib` — exige Python ≥ 3.11.
L'image doit refléter l'environnement d'entraînement, sinon le modèle se recharge
sous une version différente de scikit-learn. Alpine est plus petite, mais utilise musl libc au lieu de
glibc : les wheels précompilées de numpy et scikit-learn n'y fonctionnent pas,
et il faut tout recompiler — build de 15 minutes, et une image finale souvent
*plus grosse* à cause de la chaîne de compilation.

**2. Copie sélective.** Seuls trois éléments entrent dans l'image :

```dockerfile
COPY requirements-prod.txt .
COPY models/sentiment_model.joblib models/sentiment_model.joblib
COPY app_api.py .
```

Pas de `data/` (6,6 Mo), pas de `src/`, pas de `tests/`, pas de `reports/`,
pas de `chrome-extension/`. Un `.dockerignore` complète le dispositif en
empêchant même l'envoi de ces fichiers au démon Docker.

**3. `--no-cache-dir`.** pip conserve par défaut les wheels téléchargées
(~50-100 Mo) dans l'image finale, où elles ne servent plus à rien.

**4. Ordre des layers.** Les dépendances sont installées **avant** la copie du
code. Docker met chaque instruction en cache : tant que `requirements-prod.txt`
ne change pas, le `pip install` n'est pas rejoué. Modifier une ligne de
`app_api.py` donne alors un rebuild de quelques secondes au lieu de plusieurs
minutes. Si on inversait l'ordre, chaque modification du code invaliderait le
cache de l'installation.

**5. Modèle compressé.** `joblib.dump(..., compress=3)` ramène le `.joblib` de
~3 Mo à 0,8 Mo, pour un surcoût de décompression négligeable au démarrage.

Pour aller plus loin, un **multi-stage build** permettrait de compiler dans une
image lourde puis de ne copier que les artefacts dans une image finale minimale.
C'est inutile ici : aucune dépendance ne nécessite de compilation.

### Quelle stratégie pour les mises à jour du modèle ?

La stratégie dépend de l'enjeu. Voici celle que j'ai retenue, et les paliers
suivants.

**Mise en œuvre actuelle — versionnement sémantique + rebuild.**

1. Réentraîner : `python -m src.models.train_model`
2. Valider : `pytest` doit passer, donc accuracy ≥ 80 % et F1 > 0,75 par classe.
   **Un modèle qui échoue aux tests ne part pas en production.**
3. Incrémenter `MODEL_VERSION` dans `app_api.py` (`1.0.0` → `1.1.0`)
4. Copier le `.joblib` dans le clone du Space, commiter, pousser
5. Hugging Face rebuild automatiquement
6. Vérifier : `curl .../health` doit renvoyer la nouvelle `model_version`

Le champ `model_version` dans **chaque réponse** de `/predict_batch` est ce qui
rend l'ensemble traçable : si un utilisateur signale une prédiction aberrante, on
sait exactement quelle version l'a produite.

**Limite de cette approche :** le rebuild implique une courte interruption de
service (1 à 3 minutes). Acceptable pour un projet académique, pas pour une API
en production.

**Paliers suivants, par ordre de maturité :**

- **Découpler le modèle de l'image.** Charger le `.joblib` depuis le Hugging Face
  Hub au démarrage plutôt que de l'embarquer. Changer de modèle ne demande plus
  qu'un redémarrage du conteneur, pas un rebuild. Avec un endpoint `/reload`
  protégé, le changement devient même instantané.
- **Déploiement blue-green ou canari.** Faire tourner l'ancienne et la nouvelle
  version en parallèle, router 10 % du trafic vers la nouvelle, comparer les
  distributions de prédictions, basculer progressivement. Permet un retour en
  arrière immédiat.
- **Détection de dérive (*drift*).** Le modèle est entraîné sur des commentaires
  Reddit politiques ; les commentaires YouTube ont un vocabulaire différent, et
  l'argot évolue vite. Il faut surveiller la distribution des prédictions et la
  confiance moyenne : une baisse durable de la confiance signale que le modèle
  rencontre du vocabulaire inconnu. C'est ce qui doit **déclencher** le
  réentraînement, plutôt qu'un calendrier arbitraire.
- **Registre de modèles** (MLflow, Hugging Face Hub) : conserver chaque version
  avec ses métriques, ses hyperparamètres et le hash du dataset, pour pouvoir
  revenir à n'importe quel état antérieur et justifier une décision.
- **Pipeline CI/CD** (GitHub Actions) : à chaque push, exécuter les tests,
  reconstruire l'image, déployer uniquement si tout passe.

**Règle de fond :** ne jamais remplacer un modèle en production sans avoir
comparé ses métriques à celles du modèle sortant **sur le même test set**. Un
modèle plus récent n'est pas automatiquement meilleur.
