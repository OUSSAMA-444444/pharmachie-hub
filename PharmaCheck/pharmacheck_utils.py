"""Schéma commun, validation des données et traçabilité de PharmaCheck."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEED = 42
ID, DATE, TARGET = "identifiant_lot", "date_production", "etiquette_anomalie"
FEATURES = ["force_compression_kN", "humidite_granules_pct", "poids_comprime_mg",
            "durete_comprime_N", "friabilite_pct", "dissolution_30min_pct",
            "teneur_principe_actif_pct", "variabilite_teneur_rsd_pct"]
FEATURE_LABELS = dict(zip(FEATURES, ["Force de compression (kN)", "Humidité des granulés (%)",
    "Poids du comprimé (mg)", "Dureté du comprimé (N)", "Friabilité (%)",
    "Dissolution à 30 min (%)", "Teneur en principe actif (%)", "Variabilité de teneur — RSD (%)"]))
MODEL_NAMES = {"random_forest": "Random Forest", "xgboost": "XGBoost", "svm": "SVM (RBF)"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def versions():
    import platform
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name)
        for name in ["numpy", "pandas", "scikit-learn", "xgboost", "matplotlib", "seaborn", "joblib", "scipy"]}}


def read_csv_bytes(content, decimal="."):
    """Lecture UTF-8, contrôle des en-têtes et détection du séparateur."""
    if not content or len(content) > 20 * 1024 * 1024:
        raise ValueError("CSV vide ou supérieur à 20 Mo.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Le CSV doit être encodé en UTF-8.") from exc
    try:
        separator = csv.Sniffer().sniff(text[:16384], delimiters=",;\t").delimiter
    except csv.Error:
        separator = ","
    try:
        reader = csv.reader(io.StringIO(text), delimiter=separator, strict=True)
        header = [v.strip() for v in next(reader)]
        if not header or any(not v for v in header) or len(header) != len(set(header)):
            raise ValueError("Noms de colonnes vides ou dupliqués.")
        count = 0
        for row in reader:
            if not row:
                continue
            count += 1
            if len(row) != len(header):
                raise ValueError(f"Ligne {count} : nombre de champs incorrect.")
            if count > 20000:
                raise ValueError("Limite de 20 000 lots dépassée.")
        frame = pd.read_csv(io.StringIO(text), sep=separator, decimal=decimal, dtype={ID: str})
        frame.columns = header
        return frame
    except (csv.Error, StopIteration, pd.errors.ParserError) as exc:
        raise ValueError("CSV mal formé ou vide.") from exc


def validate_batches(frame, require_labels=False, strict=False):
    """Aucune suppression de ligne et aucun ajustement de prétraitement ici."""
    df = frame.copy().reset_index(drop=True)
    if df.empty or len(df) > 20000:
        raise ValueError("Le fichier doit contenir de 1 à 20 000 lots.")
    if df.columns.duplicated().any():
        raise ValueError("Noms de colonnes dupliqués.")
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError("Colonnes requises absentes : " + ", ".join(sorted(missing)))
    expected = [ID, DATE, *FEATURES, TARGET]
    if strict and set(df.columns) != set(expected):
        raise ValueError(f"Schéma initial attendu : {expected}")
    original_types = df.dtypes.astype(str).to_dict()
    duplicates = int(df.duplicated().sum())
    for feature in FEATURES:
        try:
            value = pd.to_numeric(df[feature], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Valeur non numérique : {feature}.") from exc
        if np.isinf(value).any() or value.lt(0).any():
            raise ValueError(f"Valeur négative ou infinie : {feature}.")
        if feature in [FEATURES[0], FEATURES[2], FEATURES[3]] and value.eq(0).any():
            raise ValueError(f"Valeur strictement positive attendue : {feature}.")
        if feature in [FEATURES[1], FEATURES[4]] and value.gt(100).any():
            raise ValueError(f"Pourcentage massique supérieur à 100 : {feature}.")
        df[feature] = value
    if df[FEATURES].isna().all(axis=1).any():
        raise ValueError("Un lot ne contient aucune mesure exploitable.")
    generated = ID not in df
    if generated:
        digest = hashlib.sha256(df[FEATURES].to_csv(index=False).encode()).hexdigest()[:8]
        df.insert(0, ID, [f"NOUVEAU-{digest}-{i+1:05d}" for i in range(len(df))])
    if df[ID].isna().any() or df[ID].astype(str).str.strip().eq("").any():
        raise ValueError("Identifiant de lot vide.")
    df[ID] = df[ID].astype(str).str.strip()
    if df[ID].duplicated().any() or duplicates:
        raise ValueError("Identifiants ou lignes dupliqués : corriger à la source.")
    if TARGET in df:
        try:
            labels = pd.to_numeric(df[TARGET], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError("Les étiquettes doivent être 0, 1 ou manquantes.") from exc
        if not labels.dropna().isin([0, 1]).all():
            raise ValueError("Les étiquettes doivent être 0 ou 1.")
        df[TARGET] = labels.astype("Int64")
    if require_labels:
        if TARGET not in df or df[TARGET].isna().any():
            raise ValueError("Chaque lot d'apprentissage doit avoir une étiquette confirmée.")
        if df[FEATURES].isna().all().any():
            raise ValueError("Une variable d'apprentissage est entièrement manquante.")
        df[TARGET] = df[TARGET].astype(int)
    if strict:
        dates = pd.to_datetime(df[DATE], format="%Y-%m-%d", errors="coerce")
        if dates.isna().any():
            raise ValueError("Dates absentes ou non conformes à AAAA-MM-JJ.")
        if set(df[TARGET].unique()) != {0, 1}:
            raise ValueError("Les deux classes sont requises dans chaque fichier initial.")
    audit = {"forme": list(df.shape), "colonnes_lues": list(frame.columns), "types_lus": original_types,
             "valeurs_manquantes": df.isna().sum().astype(int).to_dict(), "lignes_dupliquees": duplicates,
             "lignes_supprimees": 0, "identifiants_generes": generated,
             "colonnes_ignorees": [c for c in df if c not in expected],
             "etiquettes_connues": int(df[TARGET].notna().sum()) if TARGET in df else 0}
    if TARGET in df:
        audit["classes"] = {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()}
    return df, audit


def load_and_check(path, name):
    df, audit = validate_batches(read_csv_bytes(Path(path).read_bytes()), require_labels=True, strict=True)
    audit.update({"fichier": Path(path).name, "sha256": file_hash(path)})
    print(f"{name} : {df.shape}; classes : {audit['classes']}; manquants : {sum(audit['valeurs_manquantes'].values())}", flush=True)
    return df, audit


def overlap_counts(candidate, reference):
    def fingerprints(df):
        return pd.util.hash_pandas_object(df[FEATURES].astype(float), index=False)
    return {"identifiants": int(candidate[ID].astype(str).isin(reference[ID].astype(str)).sum()),
            "mesures": int(fingerprints(candidate).isin(fingerprints(reference)).sum())}


def anomaly_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, list(model.classes_).index(1)], "probabilite_non_calibree"
    return model.decision_function(X), "score_decision_non_probabiliste"


def markdown_table(df):
    def cell(value):
        return f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"] +
                     ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)])
