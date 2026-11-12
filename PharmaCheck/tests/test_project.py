"""Contrôles du protocole, des artefacts et des parcours de l'interface."""
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, recall_score, roc_auc_score, average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pharmacheck_utils import ROOT, FEATURES, ID, TARGET, validate_batches, read_csv_bytes, read_json, file_hash
from train_models import build_models, best_index
from ui_backend import load_bundle, predict_batches, evaluate_predictions, prepare_training, train_experiment, zip_folder


def fixture():
    """Mesures perturbées uniquement pour tester le logiciel, sans résultat scientifique."""
    df = pd.read_csv(ROOT / "data/PharmaCheck_train_80.csv").iloc[:100].copy()
    rng = np.random.default_rng(123)
    for feature in FEATURES:
        df[feature] += rng.normal(0, df[feature].std() * .001, len(df))
    df[ID] = [f"TEST-LOGICIEL-{i}" for i in range(len(df))]
    return df


class ProjectTests(unittest.TestCase):
    def test_saved_artifacts_reproduce_metrics(self):
        test = pd.read_csv(ROOT / "data/PharmaCheck_test_20.csv")
        models, meta = load_bundle(ROOT)
        result = predict_batches(models, test)
        saved = pd.read_csv(ROOT / "results/predictions_test.csv")
        comparison = pd.read_csv(ROOT / "results/comparaison_modeles.csv").set_index("modele")
        self.assertEqual(test.shape, (600, 11))
        self.assertEqual(int(test[TARGET].sum()), 120)
        pd.testing.assert_series_equal(saved[ID], test[ID])
        for slug, model in models.items():
            self.assertEqual(list(model.feature_names_in_), FEATURES)
            np.testing.assert_array_equal(result[f"prediction_{slug}"], saved[f"prediction_{slug}"])
            np.testing.assert_allclose(result[f"score_{slug}"], saved[f"score_{slug}"], rtol=1e-6, atol=1e-10)
            self.assertAlmostEqual(f1_score(test[TARGET], result[f"prediction_{slug}"]), comparison.loc[slug, "f1"])
            self.assertAlmostEqual(roc_auc_score(test[TARGET], result[f"score_{slug}"]), comparison.loc[slug, "roc_auc"])
            self.assertAlmostEqual(average_precision_score(test[TARGET], result[f"score_{slug}"]), comparison.loc[slug, "pr_auc_ap"])
            self.assertEqual("scaler" in model.named_steps, slug == "svm")

    def test_selection_and_folds_are_training_only(self):
        meta = read_json(ROOT / "results/selection_modeles.json")
        cv = pd.read_csv(ROOT / "results/comparaison_validation_croisee.csv")
        winner = cv.sort_values(["cv_f1_moyenne", "cv_recall_moyenne", "modele"], ascending=[False, False, True]).iloc[0].modele
        self.assertEqual(winner, meta["modele_recommande"])
        self.assertTrue(meta["selection_effectuee_sans_test"])
        assignment = pd.read_csv(ROOT / "results/plis_cv.csv")
        train = pd.read_csv(ROOT / "data/PharmaCheck_train_80.csv")
        pd.testing.assert_series_equal(assignment[ID], train[ID])
        for fold in range(1, 6):
            self.assertEqual(train.loc[assignment.pli_validation.eq(fold), TARGET].value_counts().to_dict(), {0: 384, 1: 96})
        self.assertEqual(best_index({"mean_test_f1": [.8, .81], "mean_test_recall": [.99, .7]}), 1)
        self.assertEqual(best_index({"mean_test_f1": [.8, .8], "mean_test_recall": [.7, .9]}), 1)

    def test_imputation_scaling_do_not_learn_from_new_data(self):
        model = clone(build_models()["svm"][0])
        df = fixture()
        df.loc[:4, FEATURES[0]] = np.nan
        model.fit(df[FEATURES], df[TARGET])
        np.testing.assert_allclose(model.named_steps["imputer"].statistics_, df[FEATURES].median().to_numpy())
        np.testing.assert_allclose(model.named_steps["scaler"].mean_, df[FEATURES].fillna(df[FEATURES].median()).mean().to_numpy())
        before = model.named_steps["imputer"].statistics_.copy()
        other = df.iloc[:2][FEATURES].copy()
        other.iloc[0, 0] = 1e9
        model.predict(other)
        np.testing.assert_array_equal(before, model.named_steps["imputer"].statistics_)

    def test_csv_validation(self):
        df = fixture()
        df[ID] = [f"{i:04d}" for i in range(len(df))]
        parsed = read_csv_bytes(df.to_csv(index=False, sep=";", decimal=",").encode(), decimal=",")
        self.assertEqual(parsed.loc[0, ID], "0000")
        np.testing.assert_allclose(parsed[FEATURES], df[FEATURES])
        for raw in [b"a,a\n1,2\n", b"a,b\n1,2,3\n", b""]:
            with self.assertRaises(ValueError):
                read_csv_bytes(raw)
        for bad in [-1, np.inf, "invalide"]:
            modified = df.copy()
            modified[FEATURES[0]] = modified[FEATURES[0]].astype(object)
            modified.loc[0, FEATURES[0]] = bad
            with self.assertRaises(ValueError):
                validate_batches(modified)

    def test_unlabelled_and_partial_labels(self):
        models, _ = load_bundle(ROOT)
        frame = fixture().drop(columns=[ID, TARGET])
        frame.loc[0, FEATURES[0]] = np.nan
        checked, audit = validate_batches(frame)
        self.assertTrue(audit["identifiants_generes"])
        result = predict_batches(models, checked)
        self.assertTrue(evaluate_predictions(result)[0].empty)
        checked[TARGET] = [0, 0, 0] + [np.nan] * (len(checked)-3)
        metrics, matrices = evaluate_predictions(predict_batches(models, checked))
        self.assertTrue(metrics.lots_etiquetes.eq(3).all())
        self.assertTrue(metrics.roc_auc.isna().all())
        self.assertTrue(metrics.pr_auc_ap.isna().all())
        self.assertTrue(all(matrix.sum() == 3 for matrix in matrices.values()))

    def test_training_blocks_original_test(self):
        df = pd.read_csv(ROOT / "data/PharmaCheck_test_20.csv")
        df[ID] = [f"AUTRE-{i}" for i in range(len(df))]
        for include in [True, False]:
            with self.assertRaisesRegex(ValueError, "test initial"):
                prepare_training(df, include, folds=3)
        unlabeled = fixture().drop(columns=TARGET)
        with self.assertRaises(ValueError):
            prepare_training(unlabeled, True, folds=3)

    def test_retraining_is_isolated_and_reloadable(self):
        paths = [ROOT / f"models/{s}.joblib" for s in ["random_forest", "xgboost", "svm"]]
        paths += [ROOT / "data/PharmaCheck_test_20.csv", ROOT / "results/selection_modeles.json"]
        before = {p: file_hash(p) for p in paths}
        frame = fixture()
        with tempfile.TemporaryDirectory() as directory:
            run = train_experiment(frame, include_original=False, folds=3, destination=Path(directory))
            models, meta = load_bundle(run)
            self.assertFalse(meta["test_initial_utilise_pour_entrainement"])
            self.assertEqual(len(predict_batches(models, frame)), 100)
            with zipfile.ZipFile(io.BytesIO(zip_folder(run, exclude_experiments=False))) as z:
                self.assertIsNone(z.testzip())
        self.assertEqual(before, {p: file_hash(p) for p in paths})

    def test_streamlit_navigation_demo_and_single_batch(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        app.radio(key="navigation").set_value("Analyser un CSV").run()
        next(b for b in app.button if b.label == "Essayer les 600 lots de démonstration").click().run()
        self.assertFalse(app.exception)
        app.radio(key="navigation").set_value("Lot individuel").run()
        next(b for b in app.button if b.label == "Analyser ce lot").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 3)
        app.radio(key="navigation").set_value("Améliorer les modèles").run()
        self.assertFalse(app.exception)
        self.assertTrue(next(b for b in app.button if b.label == "Lancer la nouvelle expérience").disabled)
        app.radio(key="navigation").set_value("Historique & exports").run()
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
