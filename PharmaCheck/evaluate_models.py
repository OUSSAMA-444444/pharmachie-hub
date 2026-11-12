"""Évaluation finale, prédictions, figures et rapport français."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                            average_precision_score, confusion_matrix, classification_report, make_scorer)
from pharmacheck_utils import (ROOT, FEATURES, ID, DATE, TARGET, MODEL_NAMES, SEED,
    load_and_check, file_hash, read_json, write_json, anomaly_scores, overlap_counts, markdown_table)
import plots


def calculate_metrics(y, predicted, score):
    cm = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())
    both = len(np.unique(y)) == 2
    return {"accuracy": float(accuracy_score(y, predicted)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, score)) if both else None,
            "pr_auc_ap": float(average_precision_score(y, score)) if both else None,
            "vrais_negatifs": tn, "faux_positifs": fp, "faux_negatifs": fn, "vrais_positifs": tp}, cm


def write_report(folder, metadata, audit, comparison, reports):
    recommended = metadata["modele_recommande"]
    selected = comparison.set_index("modele").loc[recommended]
    cv = pd.read_csv(folder / "comparaison_validation_croisee.csv")
    parts = ["# PharmaCheck — rapport de reconstruction et d'évaluation", "",
        "> Projet universitaire sur données synthétiques. Une prédiction n'est pas une non-conformité pharmaceutique confirmée.", "",
        "## Conclusion", "",
        f"Le modèle recommandé est **{MODEL_NAMES[recommended]}**, choisi uniquement par le F1 moyen de validation croisée "
        f"({metadata['modeles'][recommended]['cv']['f1']['moyenne']:.4f}), puis le rappel en cas d'égalité exacte. "
        "Cette sélection est sauvegardée avant l'évaluation du test.", "",
        f"Sur le test, ce modèle obtient F1={selected.f1:.4f}, rappel={selected.recall:.4f} et précision={selected.precision:.4f}. "
        f"Il manque {int(selected.faux_negatifs)} anomalies et produit {int(selected.faux_positifs)} fausses alertes.", "",
        f"Meilleur F1 descriptif sur le test : {comparison.loc[comparison.f1.idxmax(), 'nom']}. "
        f"Meilleur rappel descriptif : {comparison.loc[comparison.recall.idxmax(), 'nom']}. Ces constats ne modifient pas le choix.", "",
        "## Données et audit", "",
        "Les CSV originaux sont rechargés séparément. Les 2 400 observations d'apprentissage (1 920 normales, 480 anomalies) "
        "et les 600 observations du test (480 normales, 120 anomalies) restent inchangées. Aucune suppression de ligne, "
        "aucun rééchantillonnage et aucune redistribution entre fichiers.", "",
        f"Valeurs manquantes : {sum(audit['train']['valeurs_manquantes'].values())} dans l'apprentissage et "
        f"{sum(audit['test']['valeurs_manquantes'].values())} dans le test. Doublons de lignes : "
        f"{audit['train']['lignes_dupliquees']} et {audit['test']['lignes_dupliquees']}. Chevauchement : {audit['separation']}.", "",
        "Les en-têtes, types numériques, dates, cibles binaires, valeurs infinies et incohérences physiques élémentaires "
        "sont contrôlés. Les valeurs atypiques plausibles sont conservées. Une teneur supérieure à 100 % n'est pas "
        "automatiquement invalide car elle peut se rapporter à une valeur nominale. Aucun seuil de conformité réglementaire n'est inventé.", "",
        "Variables utilisées par les trois modèles :", "", *[f"- `{f}`" for f in FEATURES], "",
        "Les identifiants et dates sont exclus. La classe positive est etiquette_anomalie=1.", "",
        "![Classes](figures/distribution_classes.png)", "",
        "## Prétraitement et sélection", "",
        "Chaque modèle est une Pipeline avec imputation médiane ajustée dans chaque pli d'apprentissage. "
        "SVM ajoute StandardScaler dans sa Pipeline. Ni médianes, ni échelles, ni variables ne sont sélectionnées sur le test.", "",
        f"StratifiedKFold, {metadata['validation']['plis']} plis, mélange et random_state=42. Les mêmes plis sont utilisés "
        "pour les baselines et toutes les grilles. Baselines : Random Forest 300 arbres; XGBoost 300 arbres, profondeur 3, "
        "learning_rate=0,1; SVM RBF C=1 et gamma='scale'. La recherche complète compare 12 + 24 + 24 = 60 configurations.", "",
        markdown_table(cv[["nom", "baseline_f1", "baseline_recall", "cv_f1_moyenne", "cv_f1_ecart_type", "cv_recall_moyenne"]]), "",
        "Le meilleur F1 moyen est prioritaire; le rappel moyen départage les égalités. Chaque candidat choisi est réajusté "
        "sur tout l'apprentissage. Les seuils restent 0,5 pour les arbres et frontière de décision 0 pour SVM. Aucun early stopping "
        "ni réglage de seuil n'utilise le test. Les écarts-types entre plis ne sont pas des intervalles de confiance. "
        "Cette CV de réglage n'est pas imbriquée; ses meilleurs scores peuvent être optimistes.", "",
        "![CV](figures/comparaison_validation_croisee.png)", "", "### Réglages retenus", "",
        "```json", __import__('json').dumps({s: v['hyperparametres'] for s, v in metadata['modeles'].items()}, ensure_ascii=False, indent=2), "```", "",
        "## Résultats finaux sur le test", "",
        markdown_table(comparison.drop(columns=["recommande_selon_cv"])), "",
        "Précision, rappel et F1 concernent la classe 1, avec zero_division=0. Un modèle toujours normal aurait "
        "80 % d'exactitude mais un rappel et F1 nuls. ROC-AUC et AP sont calculés sur des scores continus : probabilités "
        "non calibrées des arbres et score de décision SVM, qui n'est pas une probabilité. AP correspond à Average Precision "
        "de Scikit-learn, sans intégration trapézoïdale. Sa référence aléatoire est la prévalence du test, 0,20.", "",
        "![Comparaison](figures/comparaison_metriques.png)", "", "![ROC](figures/courbes_roc.png)", "",
        "![Précision-rappel](figures/courbes_precision_rappel.png)", ""]
    for slug, report in reports.items():
        parts += [f"### {MODEL_NAMES[slug]}", "", "```text", report, "```", "",
                  f"![Matrice](figures/matrice_confusion_{slug}.png)", ""]
    parts += ["## Importance des variables", "",
        "Random Forest utilise la diminution d'impureté; XGBoost le gain normalisé. Les importances proviennent des "
        "modèles ajustés sur l'apprentissage, ne sont pas directement comparables entre méthodes et ne sont pas causales.", "",
        "![RF](figures/importance_random_forest.png)", "", "![XGBoost](figures/importance_xgboost.png)", "",
        "La permutation SVM est calculée **sur le test après l'évaluation finale**, avec 30 répétitions, F1 classe 1 et "
        "random_state=42. C'est une analyse **exploratoire**, qui ne sert à sélectionner ni variables, ni paramètres, ni modèle. "
        "Les barres représentent moyenne et écart-type des baisses de F1, sans intervalle de confiance. Les valeurs négatives "
        "sont conservées. Des variables corrélées peuvent rendre ces permutations peu représentatives.", "",
        "![Permutation](figures/importance_svm_exploratoire_test.png)", "",
        "## Compromis opérationnel et limites", "",
        "Un faux négatif manque une anomalie simulée. Un faux positif déclenche une investigation inutile. Augmenter "
        "le rappel peut accroître les alertes. Le F1 ne représente pas les coûts réels; aucun coût économique ou seuil "
        "acceptable n'est supposé. Tout seuil opérationnel devrait être développé sur des données de validation puis "
        "évalué sur un nouvel échantillon indépendant.", "",
        "- Les données sont synthétiques : aucune validation industrielle ou pharmaceutique réelle n'est démontrée.",
        "- Le test contient seulement 120 anomalies; les faibles écarts ne prouvent pas une supériorité statistique.",
        "- Les dates se chevauchent entre fichiers. Ce split ne valide pas la généralisation temporelle ou à d'autres sites.",
        "- Les grilles sont limitées; les scores CV sélectionnés peuvent être optimistes.",
        "- Les labels simulés et huit variables ne décrivent pas nécessairement toute la qualité des lots.",
        "- Une utilisation réelle exigerait données représentatives, validation externe et temporelle, surveillance de dérive "
        "et validation pharmaceutique formelle. Les modèles ne remplacent pas le contrôle qualité.", "",
        "## Interface et nouvelles expériences", "",
        "L'interface Streamlit violette accepte de nouveaux CSV et des lots saisis individuellement. Des étiquettes confirmées "
        "sont nécessaires pour évaluer ou réentraîner. Les expériences sont isolées dans experiments/, avec CV interne et "
        "versions distinctes. Les lots du test initial sont bloqués à l'apprentissage. Une hausse du F1 CV ne garantit "
        "pas une amélioration sur de nouvelles données indépendantes.", "",
        "## Reconstruction et traçabilité", "",
        "Cette archive a été reconstruite à partir des CSV originaux et du cahier des charges de la conversation; "
        "elle n'est pas présentée comme une copie binaire de l'archive perdue. Tous les résultats ci-dessus ont été recalculés.", "",
        f"Exécution UTC : {metadata['date_execution_utc']}. Versions : {metadata['versions']}.", "",
        f"SHA-256 apprentissage : `{audit['train']['sha256']}`.\n\nSHA-256 test : `{audit['test']['sha256']}`.", "",
        "Les Pipelines sont sauvegardées avec leurs prétraitements. Les métadonnées conservent les versions, empreintes, "
        "paramètres et critères. predictions_test.csv conserve l'ordre des lots, leurs identifiants, étiquettes réelles, "
        "prédictions et scores des trois modèles. Les fichiers de recherche conservent les configurations et scores par pli.", ""]
    (folder / "rapport_resultats.md").write_text("\n".join(parts), encoding="utf-8")


def evaluate(train_path=ROOT / "data/PharmaCheck_train_80.csv", test_path=ROOT / "data/PharmaCheck_test_20.csv", output=ROOT):
    output = Path(output)
    results = output / "results"
    metadata = read_json(results / "selection_modeles.json")
    if file_hash(train_path) != metadata["audit_train"]["sha256"] or metadata["features"] != FEATURES:
        raise ValueError("Données ou schéma incompatibles avec les modèles.")
    train, audit_train = load_and_check(train_path, "Apprentissage (provenance)")
    test, audit_test = load_and_check(test_path, "Test final")
    overlap = overlap_counts(test, train)
    if any(overlap.values()):
        raise ValueError(f"Chevauchement train/test : {overlap}.")
    audit = {"train": audit_train, "test": audit_test, "separation": overlap}
    write_json(results / "audit_donnees.json", audit)
    predictions = test[[ID, DATE, TARGET]].copy()
    rows, scores, matrices, reports, fitted, details = [], {}, {}, {}, {}, {}
    for slug, name in MODEL_NAMES.items():
        path = output / f"models/{slug}.joblib"
        if file_hash(path) != metadata["sha256_modeles"][slug]:
            raise ValueError(f"Modèle modifié : {slug}.")
        model = joblib.load(path)
        pred = model.predict(test[FEATURES])
        score, kind = anomaly_scores(model, test[FEATURES])
        metrics, matrix = calculate_metrics(test[TARGET], pred, score)
        rows.append({"modele": slug, "nom": name, "recommande_selon_cv": slug == metadata["modele_recommande"], **metrics})
        predictions[f"prediction_{slug}"], predictions[f"score_{slug}"] = pred, score
        scores[slug], matrices[slug], fitted[slug] = score, matrix, model
        report = classification_report(test[TARGET], pred, labels=[0, 1], target_names=["Normal (0)", "Anomalie (1)"], zero_division=0, digits=4)
        reports[slug] = report
        (results / f"rapport_classification_{slug}.txt").write_text(report, encoding="utf-8")
        details[slug] = {"metriques": metrics, "matrice_confusion": matrix.tolist(), "type_score": kind,
                         "seuil": metadata["modeles"][slug]["seuil"]}
        print(f"\n{name}\n{report}\n{metrics}", flush=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(results / "comparaison_modeles.csv", index=False)
    predictions.to_csv(results / "predictions_test.csv", index=False)
    write_json(results / "evaluation_test.json", {"modele_recommande_avant_test": metadata["modele_recommande"],
        "modeles": details, "sha256_test": audit_test["sha256"], "permutation_svm": "exploratoire_apres_evaluation"})
    importance = {}
    for slug in ["random_forest", "xgboost"]:
        importance[slug] = pd.DataFrame({"variable": FEATURES, "importance_moyenne": fitted[slug].named_steps["classifier"].feature_importances_})
        importance[slug].to_csv(results / f"importance_{slug}.csv", index=False)
    print("Permutation SVM exploratoire sur le test…", flush=True)
    perm = permutation_importance(fitted["svm"], test[FEATURES], test[TARGET],
        scoring=make_scorer(f1_score, zero_division=0), n_repeats=30, random_state=SEED, n_jobs=2)
    importance["svm"] = pd.DataFrame({"variable": FEATURES, "importance_moyenne": perm.importances_mean, "importance_ecart_type": perm.importances_std})
    importance["svm"].to_csv(results / "importance_svm_exploratoire_test.csv", index=False)
    pd.DataFrame(perm.importances.T, columns=FEATURES).to_csv(results / "permutations_svm_repetitions.csv", index=False)
    cv = pd.read_csv(results / "comparaison_validation_croisee.csv")
    plots.generate(train, test, comparison, scores, matrices, importance, cv, results / "figures")
    write_report(results, metadata, audit, comparison, reports)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/PharmaCheck_train_80.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data/PharmaCheck_test_20.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    evaluate(args.train, args.test, args.output_dir)
