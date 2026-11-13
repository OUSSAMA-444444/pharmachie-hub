# PharmaCheck · Qualité Intelligence

Interface Streamlit de détection d’anomalies de qualité : **Vue d’ensemble**, **Lot individuel** et **Historique**.

[![Application checks](https://github.com/OUSSAMA-444444/pharmachie-hub/actions/workflows/streamlit.yml/badge.svg)](https://github.com/OUSSAMA-444444/pharmachie-hub/actions/workflows/streamlit.yml)

**Application :** [Ouvrir PharmaCheck](https://pharmachie-app-33wuv5wcwhaxuj9ylmp2lz.streamlit.app/)

- Interface violette avec logo, icônes et bannière communs aux trois pages.
- Analyse des huit mesures d’un lot avec Random Forest, XGBoost et SVM.
- Historique, export CSV complet, suppression d’une ligne ou de toutes les analyses, et annulation de la dernière suppression.

## Déploiement connecté à GitHub

Dans Streamlit Community Cloud, connecter le compte GitHub, puis choisir :

- **Repository :** `OUSSAMA-444444/pharmachie-hub`
- **Branch :** `main`
- **Main file path :** `PharmaCheck/app.py`
- **Python :** `3.12` dans les paramètres avancés lors du déploiement.

Community Cloud met automatiquement l’application à jour après les changements de la branche `main`. GitHub Actions exécute les tests et vérifie ensuite que l’application publique répond correctement. Le workflow peut aussi être lancé manuellement depuis l’onglet **Actions**.

Le thème est défini dans `.streamlit/config.toml` à la racine du dépôt. Les dépendances sont épinglées dans `PharmaCheck/requirements.txt`.

L’historique de cette application partagée est commun à ses utilisateurs. Le stockage local de Community Cloud peut être réinitialisé lors d’un redémarrage ou d’un redéploiement ; télécharger le CSV pour conserver les résultats.

## Exécution locale

```bash
cd PharmaCheck
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Documentation du protocole et des limites : [PharmaCheck/README.md](PharmaCheck/README.md).
