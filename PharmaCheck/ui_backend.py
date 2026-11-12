"""Fonctions de l'interface et réentraînement isolé sur nouvelles données."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import io
import shutil
import zipfile
import joblib
import numpy as np
import pandas as pd
from pharmacheck_utils import (ROOT, FEATURES, ID, DATE, TARGET, MODEL_NAMES, FEATURE_LABELS,
    validate_batches, read_csv_bytes, overlap_counts, file_hash, read_json, write_json, anomaly_scores)
from train_models import fit_comparison
from evaluate_models import calculate_metrics


def training_path(path):
    return path / ("data/PharmaCheck_train_80.csv" if path == ROOT else "data/apprentissage.csv")


def load_bundle(path):
    path = Path(path)
    meta = read_json(path / "results/selection_modeles.json")
    if meta["features"] != FEATURES or file_hash(training_path(path)) != meta["audit_train"]["sha256"]:
        raise ValueError("Données ou schéma modifiés depuis l'apprentissage.")
    models = {}
    for slug in MODEL_NAMES:
        location = path / f"models/{slug}.joblib"
        if file_hash(location) != meta["sha256_modeles"][slug]:
            raise ValueError(f"Empreinte du modèle incompatible : {slug}.")
        models[slug] = joblib.load(location)
    return models, meta


def predict_batches(models, frame):
    result = frame[[c for c in [ID, DATE, TARGET] if c in frame]].copy()
    for slug, model in models.items():
        result[f"prediction_{slug}"] = model.predict(frame[FEATURES]).astype(int)
        result[f"score_{slug}"] = anomaly_scores(model, frame[FEATURES])[0]
    result["desaccord_modeles"] = result[[f"prediction_{s}" for s in models]].nunique(axis=1).gt(1)
    return result


def evaluate_predictions(predictions):
    if TARGET not in predictions or not predictions[TARGET].notna().any():
        return pd.DataFrame(), {}
    known = predictions.loc[predictions[TARGET].notna()]
    rows, matrices = [], {}
    for slug, name in MODEL_NAMES.items():
        metrics, matrix = calculate_metrics(known[TARGET].astype(int), known[f"prediction_{slug}"], known[f"score_{slug}"])
        rows.append({"modele": slug, "nom": name, "lots_etiquetes": len(known), **metrics})
        matrices[slug] = matrix
    return pd.DataFrame(rows), matrices


def distribution_comparison(frame, reference):
    rows = []
    for feature in FEATURES:
        train, new = reference[feature].dropna(), frame[feature].dropna()
        iqr = train.quantile(.75) - train.quantile(.25)
        rows.append({"Variable": FEATURE_LABELS[feature], "Médiane apprentissage": train.median(),
                     "Médiane fichier": new.median(), "Décalage médian / IQR": float((new.median()-train.median())/iqr) if iqr > 0 and len(new) else np.nan,
                     "Hors étendue apprentissage": float(((new < train.min()) | (new > train.max())).mean()) if len(new) else np.nan,
                     "Mesures manquantes": int(frame[feature].isna().sum())})
    return pd.DataFrame(rows)


def prepare_training(frame, include_original, folds):
    new, _ = validate_batches(frame, require_labels=True)
    # Lecture des seules mesures/identifiants du test pour empêcher son introduction en apprentissage.
    protected = pd.read_csv(ROOT / "data/PharmaCheck_test_20.csv", usecols=[ID, *FEATURES], dtype={ID: str})
    if any(overlap_counts(new, protected).values()):
        raise ValueError("Le fichier contient des lots ou mesures du test initial : apprentissage bloqué.")
    if include_original:
        original = pd.read_csv(training_path(ROOT), dtype={ID: str})
        if any(overlap_counts(new, original).values()):
            raise ValueError("Des lots sont déjà présents dans l'apprentissage initial. Importez uniquement de nouveaux lots.")
        new = pd.concat([original, new], ignore_index=True)
    if new[FEATURES].duplicated().any():
        raise ValueError("Les huit mesures sont identiques pour plusieurs lots; résoudre ces doublons avant la CV.")
    if new[TARGET].nunique() != 2 or new[TARGET].value_counts().min() < folds:
        raise ValueError(f"Chaque classe doit compter au moins {folds} lots.")
    new[TARGET] = new[TARGET].astype(int)
    return new


def train_experiment(frame, include_original=True, folds=5, quick=True, progress=None, destination=None):
    combined = prepare_training(frame, include_original, folds)
    destination = Path(destination or ROOT / "experiments")
    destination.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("exp_%Y%m%dT%H%M%S_") + uuid4().hex[:6]
    staging, final = destination / (".tmp_" + run_id), destination / run_id
    (staging / "data").mkdir(parents=True)
    try:
        combined.to_csv(staging / "data/apprentissage.csv", index=False)
        audit = {"forme": list(combined.shape), "sha256": file_hash(staging / "data/apprentissage.csv"),
                 "classes": {str(k): int(v) for k, v in combined[TARGET].value_counts().items()}}
        meta = fit_comparison(combined, staging, audit, folds=folds, jobs=1, quick=quick, progress=progress)
        meta.update({"id": run_id, "nouveaux_lots": len(frame), "inclut_apprentissage_initial": include_original,
                     "test_initial_utilise_pour_entrainement": False,
                     "evaluation_independante": "À réaliser sur un nouvel échantillon étiqueté indépendant."})
        write_json(staging / "results/selection_modeles.json", meta)
        (staging / "README.md").write_text("# Nouvelle expérience PharmaCheck\n\n"
            "Les baselines et candidats sont ajustés sur les mêmes plis. Le meilleur F1 CV sert au choix. "
            "Ces scores de réglage peuvent être optimistes. Aucun lot du test initial n'est utilisé pour apprendre. "
            "Une amélioration CV doit être confirmée sur de nouvelles données indépendantes.\n", encoding="utf-8")
        staging.rename(final)
        return final
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def zip_folder(folder=ROOT, exclude_experiments=True):
    """ZIP du projet, sans dépendances installées, caches ni fichiers temporaires."""
    folder = Path(folder)
    excluded = {".git", "node_modules", ".venv", "venv", "__pycache__", ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    if exclude_experiments:
        excluded.add("experiments")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(folder, folder.name + "/")
        for path in sorted(folder.rglob("*")):
            relative = path.relative_to(folder)
            if any(part in excluded or part.startswith(".tmp_") for part in relative.parts):
                continue
            if path.suffix in [".pyc", ".pyo", ".tmp", ".temp", ".swp"] or path.name in [".DS_Store", "Thumbs.db"]:
                continue
            archive.write(path, str(Path(folder.name) / relative))
    return out.getvalue()
