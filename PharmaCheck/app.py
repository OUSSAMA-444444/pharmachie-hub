"""PharmaCheck Studio : interface professionnelle violette, exécution locale."""
from pathlib import Path
import html
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pharmacheck_utils import (ROOT, FEATURES, FEATURE_LABELS, ID, DATE, TARGET, MODEL_NAMES,
    read_csv_bytes, validate_batches, read_json, file_hash, overlap_counts)
from ui_backend import (load_bundle, training_path, predict_batches, evaluate_predictions,
    distribution_comparison, prepare_training, train_experiment, zip_folder)

st.set_page_config(page_title="PharmaCheck · Quality Intelligence", page_icon="💊", layout="wide")
PURPLE = "#7C3AED"
STYLE = """<style>
.stApp{background:#f8f7fc;font-family:'Segoe UI',sans-serif;}
.block-container{max-width:1500px;padding-top:2rem;padding-bottom:3rem;}
[data-testid=stHeader]{background:rgba(248,247,252,.96);}
[data-testid=stSidebar]{background:linear-gradient(160deg,#28163e,#1e122e);}
[data-testid=stSidebar] [data-testid=stSidebarContent]{padding:2rem 1.1rem;}
[data-testid=stSidebar] .stMarkdown p,[data-testid=stSidebar] .stCaptionContainer p,
[data-testid=stSidebar] [data-testid=stWidgetLabel] p,
[data-testid=stSidebar] [role=radiogroup] p{color:#e6d9f4!important;}
[data-testid=stSidebar] [role=radiogroup] label{padding:.7rem .5rem;border-radius:10px;}
[data-testid=stSidebar] [role=radiogroup] label:has(input:checked){background:#64418e;}
[data-testid=stSidebar] hr{border-color:#ffffff20;}
.brand{display:flex;align-items:center;gap:10px;margin-bottom:2.5rem;}
.brand-icon{background:linear-gradient(135deg,#c49eff,#8c51d4);width:43px;height:43px;border-radius:12px;
display:grid;place-items:center;font-size:29px;color:white;}
.brand-name{font-size:21px;letter-spacing:-.8px;font-weight:800;color:white;}
.brand-small{font-size:9px;letter-spacing:2px;color:#c8b4df;text-transform:uppercase;}
.crumb{color:#8c7d9d;font-size:11px;letter-spacing:.1em;margin:0 0 24px;}
.hero{position:relative;overflow:hidden;background:linear-gradient(115deg,#422064,#69349f 62%,#8953c4);
padding:30px 37px;border-radius:21px;color:white;margin-bottom:24px;}
.hero:after{content:'+';position:absolute;right:26px;top:-52px;font-size:240px;line-height:1.2;
color:#ffffff0b;transform:rotate(-16deg);pointer-events:none;}
.hero .eyebrow{font-size:10px;letter-spacing:2px;color:#e0c9f8;margin-bottom:15px;font-weight:700;}
.hero h1{font-size:32px;letter-spacing:-1.2px;color:white;line-height:1.25;margin:0 0 14px;font-weight:800;}
.hero p{font-size:14px;line-height:1.7;color:#e9d9f9;margin:0;max-width:760px;}
.hero-foot{font-size:11px;margin-top:25px;color:#e0c8f4;}
.stat{background:#fff;border:1px solid #e6dcf1;border-radius:16px;padding:21px;min-height:138px;margin-bottom:16px;}
.stat-label{color:#8c7c9c;font-size:11px;margin-bottom:10px;}
.stat-value{color:#372046;font-weight:800;font-size:30px;letter-spacing:-1.1px;line-height:1.2;}
.stat-detail{font-size:10px;color:#998aa7;margin-top:10px;}
.panel-title{font-size:16px;font-weight:750;letter-spacing:-.4px;color:#392447;margin-bottom:5px;}
.panel-sub{font-size:11px;line-height:1.6;color:#8d7b9e;margin-bottom:15px;}
.step{display:flex;gap:12px;margin-bottom:22px;}
.step-no{flex-shrink:0;background:#ede4f8;width:28px;height:28px;display:grid;place-items:center;
border-radius:9px;color:#7947b5;font-size:11px;font-weight:700;}
.step b{font-size:12px;color:#50365e;}.step p{font-size:11px;color:#94829f;margin:3px 0 0;line-height:1.5;}
.note{background:#f0e9f8;border:1px solid #e3d6f2;border-radius:13px;padding:18px;color:#6c4d88;font-size:12px;line-height:1.7;}
.footer{border-top:1px solid #e8dfef;font-size:10px;color:#9a8aa6;padding-top:17px;margin-top:24px;line-height:1.7;}
.stButton>button,.stDownloadButton>button{border-radius:10px;min-height:42px;font-size:13px;font-weight:600;}
@media(max-width:800px){.block-container{padding:1rem;}.hero{padding:25px;}.hero h1{font-size:25px;}}
</style>"""


def title(eyebrow, text, subtitle):
    st.markdown(f'<div class="hero"><div class="eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1>{html.escape(text)}</h1><p>{html.escape(subtitle)}</p>'
        '<div class="hero-foot">8 variables de production · 3 modèles de classification · versions traçables</div></div>', unsafe_allow_html=True)


def heading(text, detail):
    st.markdown(f'<div class="panel-title">{html.escape(text)}</div><div class="panel-sub">{html.escape(detail)}</div>', unsafe_allow_html=True)


def stat(label, value, detail):
    st.markdown(f'<div class="stat"><div class="stat-label">{html.escape(str(label))}</div>'
        f'<div class="stat-value">{html.escape(str(value))}</div><div class="stat-detail">{html.escape(str(detail))}</div></div>', unsafe_allow_html=True)


def style_chart(fig, height=320):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#806c91", size=11), margin=dict(l=5, r=15, t=20, b=5),
        legend=dict(orientation="h", y=1.15, x=0), colorway=[PURPLE, "#B89ADB", "#D6C5E8"])
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#ede5f4", zeroline=False)
    return fig


def download_csv(frame, label, filename, key):
    st.download_button(label, frame.to_csv(index=False).encode("utf-8-sig"), filename,
        mime="text/csv", key=key, width="stretch", on_click="ignore")


@st.cache_resource(show_spinner=False)
def cached_bundle(path, signature):
    return load_bundle(Path(path))


def overview(meta, reference):
    recommended = meta["modele_recommande"]
    title("QUALITY INTELLIGENCE / ESPACE DE TRAVAIL", "La qualité, sous un nouvel angle.",
          "Analysez vos lots, comparez les modèles et faites évoluer votre détection d’anomalies avec de nouvelles données.")
    for col, label, value, detail in zip(st.columns(4),
        ["Lots d’apprentissage actifs", "Modèles disponibles", "Meilleur F1 en validation croisée", "Modèle recommandé"],
        [f"{len(reference):,}".replace(",", " "), "03", f"{meta['modeles'][recommended]['cv']['f1']['moyenne']:.3f}", MODEL_NAMES[recommended]],
        ["Version sélectionnée", "Random Forest · XGBoost · SVM", "Classe anomalie · critère de sélection", "Sélection sur l’apprentissage uniquement"]):
        with col:
            stat(label, value, detail)
    left, right = st.columns([1.8, 1], gap="large")
    with left, st.container(border=True):
        heading("Performance des modèles", "Validation croisée de la version active · moyenne et écart-type entre plis")
        fig = go.Figure()
        for metric, label, color in [("f1", "F1 moyen", PURPLE), ("recall", "Rappel moyen", "#CBB5E3")]:
            fig.add_bar(name=label, x=list(MODEL_NAMES.values()), y=[meta["modeles"][s]["cv"][metric]["moyenne"] for s in MODEL_NAMES],
                error_y=dict(array=[meta["modeles"][s]["cv"][metric]["ecart_type"] for s in MODEL_NAMES], thickness=1, width=3),
                marker_color=color, width=.25, hovertemplate="%{x}<br>Score : %{y:.4f}<extra>%{fullData.name}</extra>")
        fig.update_layout(barmode="group", bargap=.4)
        fig.update_yaxes(range=[0, 1], tickformat=".0%", title="Score")
        st.plotly_chart(style_chart(fig), width="stretch", config={"displayModeBar": False})
        st.caption("Ces scores servent au réglage. Les écarts-types ne sont pas des intervalles de confiance; une évaluation indépendante reste nécessaire.")
    with right, st.container(border=True):
        heading("Votre prochain contrôle", "Un parcours simple, de la mesure à l’analyse.")
        for n, text, desc in [("01", "Importez vos lots", "Un CSV avec les huit mesures de production."),
                              ("02", "Examinez les alertes", "Scores, prédictions et désaccords entre modèles."),
                              ("03", "Apprenez de nouvelles données", "Des étiquettes confirmées pour réentraîner.")]:
            st.markdown(f'<div class="step"><div class="step-no">{n}</div><div><b>{text}</b><p>{desc}</p></div></div>', unsafe_allow_html=True)
        if st.button("Analyser mes lots →", type="primary", width="stretch"):
            st.session_state.next_page = "Analyser un CSV"
            st.rerun()
        st.markdown('<div class="note">Chaque expérience conserve ses modèles, ses données d’apprentissage et ses résultats de validation.</div>', unsafe_allow_html=True)
    left, right = st.columns([1.8, 1], gap="large")
    with left, st.container(border=True):
        heading("Benchmark initial", "Évaluation historique sur 600 lots synthétiques · les nouvelles versions ne sont pas décrites par ce tableau")
        comparison = pd.read_csv(ROOT / "results/comparaison_modeles.csv")
        display = comparison[["nom", "f1", "recall", "faux_negatifs", "faux_positifs"]].rename(columns={"nom": "Modèle", "f1": "F1 test", "recall": "Rappel test", "faux_negatifs": "Anomalies manquées", "faux_positifs": "Fausses alertes"})
        st.dataframe(display.style.format({"F1 test": "{:.4f}", "Rappel test": "{:.1%}"}), hide_index=True, width="stretch")
        st.caption("Le modèle recommandé est choisi sur la CV d’apprentissage, sans sélection sur ce test.")
    with right, st.container(border=True):
        heading("Composition de l’apprentissage", "Répartition réelle dans la version active")
        counts = reference[TARGET].value_counts().reindex([0, 1], fill_value=0)
        fig = go.Figure(go.Pie(labels=["Normaux", "Anomalies"], values=counts, hole=.76,
            marker_colors=["#E8DEF3", PURPLE], textinfo="none", sort=False))
        fig.update_layout(annotations=[dict(text=f"{counts.iloc[1]/len(reference):.0%}<br><span style='font-size:11px'>anomalies</span>", x=.5, y=.5, showarrow=False, font_size=26)])
        st.plotly_chart(style_chart(fig, 210), width="stretch", config={"displayModeBar": False})


def evaluation_view(result, frame, reference):
    metrics, matrices = evaluate_predictions(result)
    if metrics.empty:
        st.info("Ajoutez etiquette_anomalie avec des étiquettes confirmées pour calculer le F1, le rappel et les autres performances.")
        return
    st.caption(f"Évaluation sur {int(frame[TARGET].notna().sum())} lots étiquetés sur {len(frame)}. Les lots sans étiquette restent dans les prédictions.")
    if any(overlap_counts(frame, reference).values()):
        st.warning("Chevauchement avec l’apprentissage actif : cette évaluation est descriptive, sans indépendance.")
    elif any(overlap_counts(frame, pd.read_csv(ROOT / "data/PharmaCheck_test_20.csv")).values()):
        st.info("Le fichier recoupe le test initial déjà observé. Il ne constitue pas un nouvel échantillon d’évaluation.")
    else:
        st.caption("Aucun chevauchement exact détecté avec l’apprentissage actif et le test initial; cela ne garantit pas l’indépendance de la source.")
    if frame[TARGET].dropna().nunique() < 2:
        st.warning("Une seule classe est étiquetée : ROC-AUC et AP sont indisponibles.")
    visible = metrics.drop(columns=["modele", "vrais_negatifs", "vrais_positifs"]).rename(columns={"nom": "Modèle", "lots_etiquetes": "Lots étiquetés", "accuracy": "Exactitude", "precision": "Précision", "recall": "Rappel", "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc_ap": "AP", "faux_negatifs": "FN", "faux_positifs": "FP"})
    st.dataframe(visible.style.format({c: "{:.4f}" for c in ["Exactitude", "Précision", "Rappel", "F1", "ROC-AUC", "AP"]}, na_rep="—"), hide_index=True, width="stretch")
    for col, (slug, matrix) in zip(st.columns(3), matrices.items()):
        with col:
            fig = go.Figure(go.Heatmap(z=matrix, x=["Normal", "Anomalie"], y=["Normal", "Anomalie"],
                text=matrix, texttemplate="%{text}", showscale=False, colorscale=[[0, "#F1EAF8"], [1, "#7338AD"]]))
            fig.update_layout(title=MODEL_NAMES[slug], xaxis_title="Prédiction", yaxis_title="Classe réelle")
            fig.update_yaxes(autorange="reversed")
            styled = style_chart(fig, 260)
            styled.update_layout(margin=dict(t=40, b=5, l=10, r=10))
            st.plotly_chart(styled, width="stretch", config={"displayModeBar": False})
    download_csv(metrics, "Télécharger les métriques", "metriques_nouveaux_lots.csv", "metrics_dl")


def analyze(models, meta, reference):
    title("ANALYSE / NOUVEAUX LOTS", "Transformez vos mesures en décisions.",
          "Importez vos lots pour examiner les alertes, comparer les modèles et inspecter les distributions.")
    left, right = st.columns([2, 1], gap="large")
    with left, st.container(border=True):
        heading("Importer un fichier CSV", "UTF-8 · virgule, point-virgule ou tabulation · 20 Mo et 20 000 lots maximum")
        upload = st.file_uploader("CSV de lots", type="csv", key=f"batch_upload_{st.session_state.get('epoch', 0)}")
        decimal = st.selectbox("Séparateur décimal", ["Point (.)", "Virgule (,)"])
        if upload:
            st.session_state.csv_bytes, st.session_state.csv_name, st.session_state.demo = upload.getvalue(), upload.name, False
        if st.button("Essayer les 600 lots de démonstration", width="stretch"):
            st.session_state.csv_bytes = (ROOT / "data/PharmaCheck_test_20.csv").read_bytes()
            st.session_state.csv_name, st.session_state.demo = "Démonstration — test initial", True
            st.session_state.epoch = st.session_state.get("epoch", 0) + 1
            st.rerun()
    with right, st.container(border=True):
        heading("Le bon format", "Les huit mesures sont requises. Identifiant, date et étiquette réelle sont facultatifs pour prédire.")
        download_csv(pd.DataFrame(columns=[ID, DATE, *FEATURES, TARGET]), "Télécharger le modèle CSV", "modele_lots.csv", "template_csv")
        with st.expander("Voir les noms de colonnes"):
            st.code("\n".join(FEATURES), language=None)
    if "csv_bytes" not in st.session_state:
        st.info("Importez un fichier ou essayez la démonstration pour afficher l’analyse.")
        return
    try:
        symbol = "." if st.session_state.get("demo") or decimal == "Point (.)" else ","
        frame, audit = validate_batches(read_csv_bytes(st.session_state.csv_bytes, symbol))
    except ValueError as exc:
        st.error(str(exc))
        return
    st.caption(f"Fichier actif : {st.session_state.csv_name}")
    if st.session_state.get("demo"):
        st.info("Démonstration sur le test initial synthétique déjà évalué.")
    slug = st.selectbox("Modèle pour examiner les alertes", list(MODEL_NAMES),
        index=list(MODEL_NAMES).index(meta["modele_recommande"]), format_func=MODEL_NAMES.get)
    result = predict_batches(models, frame)
    alerts = int(result[f"prediction_{slug}"].sum())
    for col, label, value, detail in zip(st.columns(4), ["Lots analysés", "Alertes prédites", "Sans alerte", "Désaccords entre modèles"],
        [len(frame), alerts, len(frame)-alerts, int(result.desaccord_modeles.sum())],
        ["Fichier complet", f"{alerts/len(frame):.1%} des lots · à examiner", "Selon le modèle sélectionné", "Au moins un modèle diffère"]):
        with col:
            stat(label, value, detail)
    first, second, third = st.tabs(["Résultats & alertes", "Qualité & distributions", "Évaluation avec étiquettes"])
    with first:
        left, right = st.columns([1.5, 1])
        with left, st.container(border=True):
            heading("Lots à examiner", "Une prédiction n’est pas une non-conformité confirmée.")
            only = st.toggle("Afficher uniquement les alertes")
            query = st.text_input("Rechercher un identifiant", placeholder="Ex. LOT-00042")
            filtered = result.loc[result[f"prediction_{slug}"].eq(1)].copy() if only else result.copy()
            if query:
                filtered = filtered.loc[filtered[ID].str.contains(query, case=False, regex=False)]
            display = filtered[[ID, *([TARGET] if TARGET in filtered else []), f"prediction_{slug}", f"score_{slug}", "desaccord_modeles"]]
            st.dataframe(display, hide_index=True, width="stretch", height=300)
            st.caption(f"{len(filtered)} lots affichés. L’export complet contient tous les lots et trois modèles.")
        with right, st.container(border=True):
            heading("Distribution des scores", "Score de décision SVM" if slug == "svm" else "Probabilité non calibrée")
            fig = go.Figure(go.Histogram(x=result[f"score_{slug}"], nbinsx=30, marker_color=PURPLE, name="Lots"))
            fig.add_vline(x=meta["modeles"][slug]["seuil"], line_dash="dash", line_color="#B995DC", annotation_text="Seuil du modèle")
            fig.update_layout(xaxis_title="Score anomalie", yaxis_title="Nombre de lots", showlegend=False)
            st.plotly_chart(style_chart(fig), width="stretch", config={"displayModeBar": False})
            st.caption("Le score SVM n’est pas une probabilité; les scores des différents modèles ne se comparent pas directement.")
        download_csv(result, "Télécharger toutes les prédictions", "predictions_nouveaux_lots.csv", "full_predictions")
        with st.expander("Exporter uniquement les lots filtrés"):
            download_csv(filtered, "Télécharger la sélection", "predictions_selection.csv", "filtered_predictions")
    with second:
        missing = int(frame[FEATURES].isna().sum().sum())
        st.caption(f"{missing} cellules numériques à imputer; {audit['etiquettes_connues']} étiquettes disponibles. Les médianes enregistrées dans les modèles sont utilisées.")
        if audit["identifiants_generes"]:
            st.info("Des identifiants techniques NOUVEAU-… ont été générés pour cet import.")
        if audit["colonnes_ignorees"]:
            st.caption("Colonnes exclues des modèles : " + ", ".join(audit["colonnes_ignorees"]))
        table = distribution_comparison(frame, reference)
        heading("Vos données face à l’apprentissage", "Écart de médiane divisé par l’écart interquartile (IQR) de l’apprentissage")
        fig = go.Figure(go.Bar(x=table["Décalage médian / IQR"], y=table.Variable, orientation="h", marker_color=PURPLE))
        fig.add_vline(x=0, line_color="#BDA8D0")
        fig.update_layout(xaxis_title="Décalage médian / IQR", yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_chart(fig, 360), width="stretch", config={"displayModeBar": False})
        st.caption("Comparaison descriptive, sans test statistique de dérive. Un écart de distribution ne démontre pas une baisse de performance.")
        st.dataframe(table, hide_index=True, width="stretch")
    with third:
        evaluation_view(result, frame, reference)


def single(models, meta, reference):
    title("CONTRÔLE / LOT INDIVIDUEL", "Un lot. Huit mesures. Trois regards.", "Renseignez les mesures du lot pour obtenir les prédictions des trois modèles.")
    with st.form("single_batch"):
        batch = st.text_input("Identifiant du lot", value="MON-LOT-001")
        columns = st.columns(2)
        values = {}
        for i, feature in enumerate(FEATURES):
            with columns[i % 2]:
                values[feature] = st.number_input(FEATURE_LABELS[feature],
                    min_value=.001 if feature in [FEATURES[0], FEATURES[2], FEATURES[3]] else 0.,
                    max_value=100. if feature in [FEATURES[1], FEATURES[4]] else None,
                    value=float(reference[feature].median()), step=.01, format="%.3f")
        submitted = st.form_submit_button("Analyser ce lot", type="primary", width="stretch")
    st.caption("Les valeurs initiales sont les médianes de l’apprentissage actif, sans correspondre à un lot réel confirmé.")
    if submitted:
        try:
            frame, _ = validate_batches(pd.DataFrame([{ID: batch, **values}]))
            result = predict_batches(models, frame)
            st.subheader(f"Résultat · {batch}")
            for col, slug in zip(st.columns(3), MODEL_NAMES):
                with col, st.container(border=True):
                    heading(MODEL_NAMES[slug], "Recommandé par CV" if slug == meta["modele_recommande"] else "Comparaison")
                    alert = bool(result.iloc[0][f"prediction_{slug}"])
                    (st.warning if alert else st.success)("Lot à examiner" if alert else "Aucune alerte prédite")
                    st.metric("Score de décision" if slug == "svm" else "Probabilité non calibrée", f"{result.iloc[0][f'score_{slug}']:.4f}")
            download_csv(result, "Télécharger le résultat", "prediction_lot.csv", "single_result")
        except ValueError as exc:
            st.error(str(exc))


def improvement():
    title("LABORATOIRE / NOUVELLE EXPÉRIENCE", "Faites évoluer vos modèles.", "Enrichissez l’apprentissage avec des lots étiquetés et conservez une nouvelle version indépendante.")
    st.info("Utilisez des étiquettes confirmées. Une amélioration CV peut être optimiste et doit être confirmée sur un nouvel échantillon indépendant.")
    left, right = st.columns([1.7, 1])
    with left, st.container(border=True):
        heading("Données et validation", "CSV avec huit mesures et une étiquette confirmée 0 ou 1 pour chaque lot")
        upload = st.file_uploader("Nouvelles données étiquetées", type="csv", key="train_upload")
        decimal = st.selectbox("Décimales de l’apprentissage", ["Point (.)", "Virgule (,)"])
        include = st.toggle("Ajouter à l’apprentissage initial", value=True)
        folds = st.select_slider("Plis stratifiés", options=[3, 5, 7], value=5)
        mode = st.radio("Recherche", ["Rapide · pondération des classes", "Complète · 60 configurations"])
        frame, prepared = None, None
        if upload:
            try:
                frame, _ = validate_batches(read_csv_bytes(upload.getvalue(), "," if decimal.startswith("Virgule") else "."), require_labels=True)
                prepared = prepare_training(frame, include, folds)
                st.success(f"{len(frame)} nouveaux lots; ensemble prévu : {len(prepared)} lots, {folds} plis.")
            except ValueError as exc:
                st.error(str(exc))
        start = st.button("Lancer la nouvelle expérience", type="primary", width="stretch", disabled=prepared is None)
    with right, st.container(border=True):
        heading("Un protocole traçable", "Les modèles initiaux sont conservés.")
        st.markdown("- F1 des anomalies prioritaire, rappel secondaire.\n- Même CV pour les baselines et candidats.\n- Prétraitements ajustés dans les plis.\n- Test initial bloqué à l’apprentissage.\n- Aucun pseudo-étiquetage automatique.\n- Modèles, données et résultats sauvegardés séparément.")
        download_csv(pd.DataFrame(columns=[ID, DATE, *FEATURES, TARGET]), "Modèle CSV d’apprentissage", "modele_apprentissage.csv", "train_template")
        st.caption("Le mode rapide compare deux pondérations par modèle avec les autres réglages de baseline. Le mode complet utilise les 60 configurations du protocole initial.")
    if start and frame is not None:
        bar = st.progress(0., text="Préparation…")
        try:
            run = train_experiment(frame, include, folds, quick=mode.startswith("Rapide"), progress=lambda f, text: bar.progress(f, text=text))
            st.session_state.last_run = str(run)
            st.success("Expérience sauvegardée.")
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(f"Expérience interrompue : {exc}")
    if "last_run" in st.session_state and Path(st.session_state.last_run).is_dir():
        run = Path(st.session_state.last_run)
        table = pd.read_csv(run / "results/comparaison_validation_croisee.csv")
        table["gain_f1_cv"] = table.cv_f1_moyenne - table.baseline_f1
        heading("Résultats de l’expérience", "Témoin et candidats réajustés sur les mêmes données et plis; ce gain n’est pas une évaluation indépendante.")
        st.dataframe(table[["nom", "baseline_f1", "cv_f1_moyenne", "cv_recall_moyenne", "gain_f1_cv"]], hide_index=True, width="stretch")
        first, second = st.columns(2)
        with first:
            if st.button("Activer cette version", type="primary", width="stretch"):
                st.session_state.next_bundle = str(run)
                st.rerun()
        with second:
            st.download_button("Exporter cette expérience", zip_folder(run, exclude_experiments=False), run.name + ".zip", mime="application/zip", width="stretch", on_click="ignore")


def history(bundles, active):
    title("TRAÇABILITÉ / HISTORIQUE & EXPORTS", "Chaque version raconte ses résultats.", "Retrouvez vos expériences et téléchargez le projet, les modèles et leurs métadonnées.")
    rows = []
    for bundle in bundles:
        meta = read_json(bundle / "results/selection_modeles.json")
        chosen = meta["modele_recommande"]
        rows.append({"Version": "Référence initiale" if bundle == ROOT else bundle.name,
            "Lots d’apprentissage": meta["audit_train"]["forme"][0], "Modèle recommandé": MODEL_NAMES[chosen],
            "F1 CV": meta["modeles"][chosen]["cv"]["f1"]["moyenne"], "Active": bundle == active})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Les versions peuvent utiliser des données différentes. Comparez leurs scores avec prudence et utilisez des données indépendantes pour confirmer un gain.")
    left, right = st.columns(2)
    with left, st.container(border=True):
        heading("Projet complet", "Code, configuration, README, CSV synthétiques, modèles initiaux et résultats. Expériences exportées séparément.")
        st.download_button("Télécharger my-project.zip", zip_folder(), "my-project.zip", mime="application/zip", type="primary", width="stretch", on_click="ignore")
        st.download_button("Rapport initial (.md)", (ROOT / "results/rapport_resultats.md").read_bytes(), "rapport_resultats.md", mime="text/markdown", width="stretch", on_click="ignore")
    with right, st.container(border=True):
        heading("Modèles actifs", "Pipelines avec prétraitement inclus; utiliser les versions de bibliothèques fournies.")
        for slug, name in MODEL_NAMES.items():
            st.download_button(name + " · .joblib", (active / f"models/{slug}.joblib").read_bytes(), slug + ".joblib", key="dl_"+slug, width="stretch", on_click="ignore")
        if active != ROOT:
            st.download_button("Télécharger la version active", zip_folder(active, exclude_experiments=False), active.name+".zip", width="stretch", on_click="ignore")


def main():
    st.markdown(STYLE, unsafe_allow_html=True)
    bundles = [ROOT] + sorted([p for p in (ROOT / "experiments").glob("exp_*") if (p / "results/selection_modeles.json").is_file()], reverse=True)
    if "next_bundle" in st.session_state:
        st.session_state.active_bundle = st.session_state.pop("next_bundle")
    if st.session_state.get("active_bundle") not in [str(p) for p in bundles]:
        st.session_state.active_bundle = str(ROOT)
    pages = ["Vue d’ensemble", "Analyser un CSV", "Lot individuel", "Améliorer les modèles", "Historique & exports"]
    if "next_page" in st.session_state:
        st.session_state.navigation = st.session_state.pop("next_page")
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-icon">✚</div><div><div class="brand-name">PharmaCheck</div><div class="brand-small">Quality Intelligence</div></div></div>', unsafe_allow_html=True)
        st.caption("ESPACE DE TRAVAIL")
        page = st.radio("Navigation", pages, key="navigation", label_visibility="collapsed")
        st.divider()
        active = Path(st.selectbox("Version active", [str(p) for p in bundles], key="active_bundle", format_func=lambda p: "Référence initiale" if Path(p) == ROOT else Path(p).name))
        try:
            models, meta = cached_bundle(str(active), file_hash(active / "results/selection_modeles.json"))
            reference = pd.read_csv(training_path(active), dtype={ID: str})
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Version indisponible : {exc}")
            st.stop()
        st.caption(f"Recommandé : {MODEL_NAMES[meta['modele_recommande']]}")
        st.caption(f"Validation : {meta['validation']['plis']} plis · graine 42")
        st.divider()
        st.caption("PHARMACHECK STUDIO\n\nProjet académique · données initiales synthétiques\n\nReconstruction v2.1")
    st.markdown(f'<div class="crumb">PHARMACHECK &nbsp; / &nbsp; {html.escape(page.upper())}</div>', unsafe_allow_html=True)
    if page == pages[0]:
        overview(meta, reference)
    elif page == pages[1]:
        analyze(models, meta, reference)
    elif page == pages[2]:
        single(models, meta, reference)
    elif page == pages[3]:
        improvement()
    else:
        history(bundles, active)
    st.markdown('<div class="footer">PharmaCheck Studio · Détection d’anomalies de qualité assistée par apprentissage automatique.'
        '<br>Projet académique sur données initiales synthétiques. Une prédiction n’est pas une non-conformité pharmaceutique confirmée.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
