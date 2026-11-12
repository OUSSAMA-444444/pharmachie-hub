"""Baselines, réglage et recommandation uniquement sur l'apprentissage."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from pharmacheck_utils import ROOT, FEATURES, TARGET, ID, SEED, MODEL_NAMES, load_and_check, file_hash, write_json, versions

SCORING = {"f1": make_scorer(f1_score, zero_division=0),
           "recall": make_scorer(recall_score, zero_division=0),
           "precision": make_scorer(precision_score, zero_division=0),
           "accuracy": "accuracy", "roc_auc": "roc_auc", "average_precision": "average_precision"}


def best_index(results):
    return int(sorted(range(len(results["mean_test_f1"])),
                      key=lambda i: (-results["mean_test_f1"][i], -results["mean_test_recall"][i], i))[0])


def build_models():
    def pipe(classifier, scale=False):
        steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        return Pipeline([*steps, ("classifier", classifier)])
    return {
        "random_forest": (pipe(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1)),
            {"classifier__max_depth": [None, 6, 12], "classifier__min_samples_leaf": [1, 3],
             "classifier__class_weight": [None, "balanced"]}),
        "xgboost": (pipe(XGBClassifier(n_estimators=300, max_depth=3, learning_rate=.1,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=SEED, n_jobs=1)),
            {"classifier__n_estimators": [200, 400], "classifier__max_depth": [2, 3, 5],
             "classifier__learning_rate": [.05, .1], "classifier__scale_pos_weight": [1, 4]}),
        "svm": (pipe(SVC(kernel="rbf", C=1, gamma="scale", probability=False, random_state=SEED), scale=True),
            {"classifier__C": [.1, 1, 10, 100], "classifier__gamma": ["scale", .01, .1],
             "classifier__class_weight": [None, "balanced"]})}


def fit_comparison(df, output, audit, folds=5, jobs=2, quick=False, progress=None):
    """Prétraitements dans les Pipelines, réajustés dans chaque pli."""
    output = Path(output)
    if df[TARGET].nunique() != 2 or df[TARGET].value_counts().min() < folds or folds < 2:
        raise ValueError(f"Chaque classe doit contenir au moins {folds} lots.")
    for directory in ["models", "results"]:
        (output / directory).mkdir(parents=True, exist_ok=True)
    X, y = df[FEATURES], df[TARGET].astype(int)
    splits = list(StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y))
    assignment = np.zeros(len(df), dtype=int)
    for fold, (_, indices) in enumerate(splits, 1):
        assignment[indices] = fold
    pd.DataFrame({ID: df[ID], "pli_validation": assignment}).to_csv(output / "results/plis_cv.csv", index=False)
    rows, baseline_rows, selected, hashes = [], [], {}, {}
    for number, (slug, (estimator, grid)) in enumerate(build_models().items()):
        if quick:
            grid = {"classifier__scale_pos_weight": [1, 4]} if slug == "xgboost" else {"classifier__class_weight": [None, "balanced"]}
        if progress:
            progress(number / 3, f"{MODEL_NAMES[slug]} : validation et réglage…")
        print(f"\n=== {MODEL_NAMES[slug]} ===", flush=True)
        baseline = cross_validate(estimator, X, y, cv=splits, scoring=SCORING, n_jobs=jobs, error_score="raise")
        for fold in range(folds):
            baseline_rows.append({"modele": slug, "pli": fold + 1,
                                  **{metric: float(baseline[f"test_{metric}"][fold]) for metric in SCORING}})
        search = GridSearchCV(estimator, grid, scoring=SCORING, refit=best_index, cv=splits,
                              n_jobs=jobs, pre_dispatch=jobs, error_score="raise")
        search.fit(X, y)
        index = search.best_index_
        details = pd.DataFrame(search.cv_results_)
        details["selectionne"] = details.index == index
        details.to_csv(output / f"results/recherche_cv_{slug}.csv", index=False)
        joblib.dump(search.best_estimator_, output / f"models/{slug}.joblib", compress=3)
        hashes[slug] = file_hash(output / f"models/{slug}.joblib")
        selected[slug] = {"hyperparametres": search.best_params_, "nombre_candidats": len(details),
            "baseline": {metric: float(baseline[f"test_{metric}"].mean()) for metric in SCORING},
            "cv": {metric: {"moyenne": float(search.cv_results_[f"mean_test_{metric}"][index]),
                            "ecart_type": float(search.cv_results_[f"std_test_{metric}"][index])} for metric in SCORING},
            "seuil": 0.0 if slug == "svm" else .5}
        rows.append({"modele": slug, "nom": MODEL_NAMES[slug],
                     **{f"baseline_{m}": selected[slug]["baseline"][m] for m in SCORING},
                     **{f"cv_{m}_{stat}": selected[slug]["cv"][m][stat] for m in SCORING for stat in ["moyenne", "ecart_type"]}})
        print(f"F1 CV={selected[slug]['cv']['f1']['moyenne']:.4f}; rappel={selected[slug]['cv']['recall']['moyenne']:.4f}", flush=True)
    recommended = sorted(selected, key=lambda s: (-selected[s]["cv"]["f1"]["moyenne"],
                                                  -selected[s]["cv"]["recall"]["moyenne"], s))[0]
    pd.DataFrame(rows).to_csv(output / "results/comparaison_validation_croisee.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(output / "results/baseline_plis_cv.csv", index=False)
    metadata = {"date_execution_utc": datetime.now(timezone.utc).isoformat(), "features": FEATURES, "target": TARGET,
        "audit_train": audit, "versions": versions(), "random_state": SEED, "modeles": selected,
        "sha256_modeles": hashes, "modele_recommande": recommended, "selection_effectuee_sans_test": True,
        "validation": {"plis": folds, "random_state": SEED, "shuffle": True, "cv_imbriquee": False,
                       "mode": "rapide" if quick else "complet", "criteres": ["f1_classe_1", "rappel_classe_1"]},
        "sha256_sources_apprentissage": {name: file_hash(ROOT / name) for name in ["train_models.py", "pharmacheck_utils.py"]}}
    write_json(output / "results/selection_modeles.json", metadata)
    if progress:
        progress(1.0, "Expérience sauvegardée.")
    print(f"Recommandation avant test : {MODEL_NAMES[recommended]}", flush=True)
    return metadata


def train(path, output=ROOT, folds=5, jobs=2):
    # Le chemin du test n'est jamais reçu ni ouvert par cette fonction.
    df, audit = load_and_check(path, "Apprentissage")
    return fit_comparison(df, output, audit, folds, jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/PharmaCheck_train_80.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs doit être positif.")
    train(args.train, args.output_dir, args.cv_folds, args.jobs)
