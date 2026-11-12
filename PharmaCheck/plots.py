"""Figures reproductibles de l'évaluation finale, en français."""
import os
import tempfile
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pharmacheck-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
from pharmacheck_utils import FEATURES, FEATURE_LABELS, TARGET, MODEL_NAMES

sns.set_theme(style="whitegrid", font="DejaVu Sans")
COLORS = {"random_forest": "#9970C7", "xgboost": "#6F3CA0", "svm": "#46206B"}


def save(fig, folder, name):
    Path(folder).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(Path(folder) / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate(train, test, comparison, scores, matrices, importance, cv, folder):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, df, name in zip(axes, [train, test], ["Apprentissage", "Test"]):
        counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)
        bars = ax.bar(["Normal (0)", "Anomalie (1)"], counts, color=["#D4C0E7", "#7139A4"])
        ax.bar_label(bars, labels=[f"{n} ({n / len(df):.0%})" for n in counts], padding=4)
        ax.set(title=f"{name} — {len(df)} lots", xlabel="Classe réelle", ylabel="Nombre de lots", ylim=(0, 2200))
        ax.legend(bars, ["Normal", "Anomalie simulée"], title="Étiquette")
    fig.suptitle("Répartition des classes — split fourni inchangé")
    save(fig, folder, "distribution_classes")
    for slug, cm in matrices.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax,
                    xticklabels=["Normal", "Anomalie"], yticklabels=["Normal", "Anomalie"],
                    cbar_kws={"label": "Nombre de lots"})
        ax.set(title=f"Matrice de confusion — {MODEL_NAMES[slug]}\nTest final", xlabel="Classe prédite", ylabel="Classe réelle")
        save(fig, folder, f"matrice_confusion_{slug}")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(4)
    for i, row in enumerate(comparison.itertuples()):
        values = [row.accuracy, row.precision, row.recall, row.f1]
        bars = ax.bar(x + (i-1)*.24, values, .24, color=COLORS[row.modele], label=MODEL_NAMES[row.modele])
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=["Exactitude", "Précision (1)", "Rappel (1)", "F1 (1)"],
           ylim=(0, 1.12), xlabel="Indicateur", ylabel="Score", title="Performances sur le test inchangé")
    ax.legend(title="Modèle", ncol=3, loc="lower center")
    save(fig, folder, "comparaison_metriques")
    indexed = comparison.set_index("modele")
    fig, ax = plt.subplots(figsize=(7, 6))
    for slug, score in scores.items():
        fpr, tpr, _ = roc_curve(test[TARGET], score)
        ax.plot(fpr, tpr, color=COLORS[slug], label=f"{MODEL_NAMES[slug]} — AUC {indexed.loc[slug, 'roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Classement aléatoire")
    ax.set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs (rappel)", title="Courbes ROC — test", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(title="Modèle", loc="lower right")
    save(fig, folder, "courbes_roc")
    fig, ax = plt.subplots(figsize=(7, 6))
    for slug, score in scores.items():
        precision, recall, _ = precision_recall_curve(test[TARGET], score)
        ax.step(recall, precision, where="post", color=COLORS[slug], label=f"{MODEL_NAMES[slug]} — AP {indexed.loc[slug, 'pr_auc_ap']:.3f}")
    ax.axhline(test[TARGET].mean(), linestyle="--", color="gray", label="Référence aléatoire : prévalence")
    ax.set(xlabel="Rappel des anomalies", ylabel="Précision des alertes", title="Courbes précision-rappel — test", xlim=(0, 1), ylim=(0, 1.03))
    ax.legend(title="Modèle", loc="lower left")
    save(fig, folder, "courbes_precision_rappel")
    for slug, data in importance.items():
        ordered = data.sort_values("importance_moyenne")
        fig, ax = plt.subplots(figsize=(10, 5.6))
        exploratory = slug == "svm"
        ax.barh([FEATURE_LABELS[f] for f in ordered.variable], ordered.importance_moyenne,
                xerr=ordered.importance_ecart_type if exploratory else None, color=COLORS[slug],
                capsize=3 if exploratory else 0, label="Permutation sur test" if exploratory else "Modèle ajusté sur l'apprentissage")
        measure = "diminution d'impureté" if slug == "random_forest" else "gain normalisé"
        ax.set(title=f"{MODEL_NAMES[slug]} — " + ("permutation EXPLORATOIRE sur test\n30 répétitions, après évaluation" if exploratory else measure),
               xlabel="Baisse du F1 (moyenne ± écart-type)" if exploratory else "Importance normalisée", ylabel="Variable")
        ax.legend(title="Origine", loc="lower right")
        save(fig, folder, f"importance_{slug}" + ("_exploratoire_test" if exploratory else ""))
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cv))
    for offset, metric, label, color in [(-.17, "f1", "F1 classe 1", "#6F3CA0"), (.17, "recall", "Rappel classe 1", "#C9AFDF")]:
        ax.bar(x+offset, cv[f"cv_{metric}_moyenne"], .34, yerr=cv[f"cv_{metric}_ecart_type"], capsize=4, label=label, color=color)
    ax.set(xticks=x, xticklabels=cv.nom, ylim=(0, 1.03), ylabel="Score moyen ± écart-type", xlabel="Modèle réglé",
           title="Validation croisée — apprentissage uniquement")
    ax.legend(title="Indicateur")
    save(fig, folder, "comparaison_validation_croisee")
