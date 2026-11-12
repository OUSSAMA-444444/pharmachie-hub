# PharmaCheck Studio

Projet universitaire de détection d'anomalies de qualité **simulées** en fabrication
pharmaceutique : Random Forest, XGBoost et SVM RBF, avec interface Streamlit violette.

Cette version a été **reconstruite** à partir des CSV originaux et des exigences de la
conversation. Les résultats ont été recalculés; l'archive n'est pas une copie binaire
de l'ancienne archive perdue. Les modèles ne sont pas validés pour un usage industriel réel.

## Démarrage rapide — Windows

1. Extraire `my-project.zip` puis ouvrir le dossier `PharmaCheck`.
2. Installer Python **3.12** et, si souhaité, VS Code avec son extension Python.
3. Double-cliquer sur **start_app.bat**.
4. Le navigateur affiche l'application à `http://localhost:8501`.

Le lanceur crée `.venv`, installe les versions de bibliothèques fournies et lance Streamlit.
Une connexion Internet est nécessaire pour installer les dépendances la première fois.

Installation manuelle dans PowerShell :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## macOS / Linux

```bash
bash start_app.sh
```

Ou manuellement :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Dans VS Code, sélectionner l'interpréteur `.venv`. Les Pipelines déjà ajustées sont incluses;
aucun réentraînement n'est requis pour utiliser l'interface.

## Interface

- **Vue d'ensemble** : scores CV de la version active et benchmark initial clairement séparés.
- **Analyser un CSV** : nouveaux lots, alertes, scores, désaccords, filtres, exports et distributions.
- **Lot individuel** : saisie des huit mesures et prédictions des trois modèles.
- **Améliorer les modèles** : nouvelles données étiquetées, CV stratifiée et versions isolées.
- **Historique & exports** : modèles actifs, expériences et téléchargement du projet.

Le CSV doit être encodé en UTF-8. Virgule, point-virgule et tabulation sont détectés.
Le séparateur décimal se choisit dans l'interface. Les huit colonnes numériques sont :

```text
force_compression_kN
humidite_granules_pct
poids_comprime_mg
durete_comprime_N
friabilite_pct
dissolution_30min_pct
teneur_principe_actif_pct
variabilite_teneur_rsd_pct
```

`identifiant_lot`, `date_production` et `etiquette_anomalie` sont facultatifs pour prédire.
Sans identifiant, un identifiant technique est généré. Les mesures manquantes sont imputées
avec les médianes **déjà ajustées dans chaque Pipeline**. Aucun ajustement sur l'import.
Une ligne sans aucune mesure est refusée. Les valeurs atypiques plausibles restent présentes.

Pour évaluer, fournir des étiquettes confirmées 0 (normal) ou 1 (anomalie). Avec des étiquettes
partielles, seul le sous-ensemble étiqueté est évalué et son effectif est affiché. ROC-AUC
et AP sont indisponibles si une seule classe est étiquetée. Sans étiquette, aucun F1 n'est inventé.
Le score SVM est une distance signée à sa frontière, pas une probabilité. Les scores des arbres
sont des probabilités non calibrées; les échelles ne se comparent pas directement entre modèles.

Les filtres changent la sélection affichée. L'export complet conserve tous les lots et les trois
modèles; un export distinct contient uniquement la sélection filtrée. La comparaison des distributions
est descriptive; elle ne prouve ni dérive statistique ni dégradation de la performance.

## Nouvelles expériences

Fournir des étiquettes confirmées pour chaque nouveau lot. Choisir l'ajout aux 2 400 observations
initiales ou l'apprentissage sur le seul nouveau fichier. Les doublons d'identifiants et de mesures
sont bloqués, sans suppression silencieuse. Chaque classe doit avoir au moins autant de lots que de plis.

Le mode rapide compare deux pondérations par modèle avec les réglages de baseline. Le mode complet
utilise les 60 configurations du protocole initial. Baselines et candidats sont réajustés sur les
mêmes plis. La recommandation maximise le F1 moyen de la classe 1, puis le rappel en cas d'égalité.

Les expériences sont stockées dans `experiments/exp_…/`. Elles conservent les Pipelines, données,
scores par pli, paramètres, empreintes et versions. Les modèles initiaux sont conservés.
Les lots ou mesures exactes du test initial sont interdits en apprentissage : seules les mesures
et identifiants du test sont lus pour ce contrôle, jamais ses étiquettes. Les prédictions ne sont
pas transformées automatiquement en étiquettes. Les seuils restent les seuils par défaut des estimateurs.

Une hausse du F1 CV peut être optimiste et **ne garantit pas** un gain sur de nouveaux lots.
Confirmer l'amélioration sur un nouvel échantillon indépendant. Une validation temporelle ou par
groupe doit être conçue pour les lots apparentés; elle n'est pas automatisée dans cette interface.

## Reproduire l'expérience initiale

```bash
python train_models.py
python evaluate_models.py
python -m unittest discover -s tests -v
```

Le premier script n'ouvre que le fichier d'apprentissage. Cinq plis stratifiés mélangés,
random_state=42, sont communs aux modèles et baselines. L'imputation est dans chaque Pipeline;
StandardScaler est dans celle du SVM. Les grilles comprennent 12 Random Forest, 24 XGBoost et
24 SVM, soit 300 ajustements de CV plus les baselines et réajustements finaux.

L'évaluation finale utilise le test fourni sans le changer ni réoptimiser les modèles. La
permutation SVM sur test vient après cette évaluation et porte explicitement le statut exploratoire.
Les scores CV de réglage ne sont pas issus d'une CV imbriquée et peuvent être optimistes.
Les scripts réécrivent les résultats de leur destination; pour conserver les anciens :

```bash
python train_models.py --output-dir execution_2 --jobs 1
python evaluate_models.py --output-dir execution_2
```

`--train`, `--test` et `--cv-folds` sont aussi disponibles. Changer les données ou le nombre
de plis définit une autre expérience. Python 3.12 et les versions de requirements.txt sont vérifiés.

## Structure

```text
PharmaCheck/
├── data/                              CSV originaux inchangés
├── models/                            Trois Pipelines .joblib
├── results/
│   ├── figures/                       Onze figures Matplotlib / Seaborn
│   ├── comparaison_modeles.csv
│   ├── predictions_test.csv
│   ├── rapport_resultats.md
│   ├── comparaison_validation_croisee.csv
│   ├── recherche_cv_*.csv
│   ├── baseline_plis_cv.csv
│   ├── plis_cv.csv
│   ├── selection_modeles.json
│   ├── evaluation_test.json
│   ├── audit_donnees.json
│   ├── rapport_classification_*.txt
│   └── importance_*.csv
├── experiments/                       Nouvelles versions, séparées
├── tests/test_project.py
├── .streamlit/config.toml
├── app.py
├── ui_backend.py
├── pharmacheck_utils.py
├── train_models.py
├── evaluate_models.py
├── plots.py
├── start_app.bat
├── start_app.sh
├── requirements.txt
└── README.md
```

Le rapport français contient tous les résultats effectivement calculés, la sélection et ses
limites. Les matrices utilisent [[VN, FP], [FN, VP]]. `pr_auc_ap` est Average Precision.
Les `.joblib` incluent l'imputation et, pour SVM, la mise à l'échelle. Charger uniquement des
modèles de confiance : Joblib peut exécuter du code au chargement.

## Télécharger l'archive et ChatGPT Plus

Un chemin `/workspace/...` désigne le système de fichiers du cloud, pas celui de votre ordinateur.
Si un fichier `my-project.zip` apparaît comme pièce jointe ou dans **Outputs**, utiliser son bouton
de téléchargement. Le fichier doit réellement apparaître; un chemin écrit en texte n'est pas une pièce jointe.
ChatGPT Plus ne transforme pas un chemin du cloud en lien public de téléchargement.

Dans une session proposant l'accès aux fichiers du workspace, télécharger `outputs/my-project.zip`.
Dans un navigateur ayant accès à l'application, ouvrir **Historique & exports → Télécharger my-project.zip**.
Si la session n'offre ni pièce jointe, ni sortie téléchargeable, ni accès aux fichiers, une copie
publiée dans le dépôt GitHub choisi par l'utilisateur fournit un autre moyen de récupération.
Un ZIP sur GitHub se télécharge depuis sa page avec **Download raw file**; pour tout le dépôt,
utiliser **Code → Download ZIP**. Un dépôt privé nécessite de se connecter au compte qui y a accès.

## Limites

Données initiales synthétiques, petit nombre d'anomalies dans le test, grilles limitées,
absence de validation temporelle et de calibration clinique/industrielle. Un faux négatif manque
une anomalie simulée; un faux positif cause une investigation inutile. Le F1 ne représente pas
le coût de ces deux erreurs. Aucun usage réel n'est validé par ce projet académique.
