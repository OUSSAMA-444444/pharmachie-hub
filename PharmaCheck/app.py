"""PharmaCheck Studio : contrôle qualité et historique des analyses en local."""
from pathlib import Path
import base64
import html
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pharmacheck_utils import (
    ROOT, FEATURES, FEATURE_LABELS, ID, TARGET, MODEL_NAMES,
    file_hash, validate_batches,
)
from ui_backend import load_bundle, predict_batches, training_path
from analysis_history import (
    delete_history, history_csv, latest_history_deletion, read_history, record_analysis, restore_history,
)

st.set_page_config(
    page_title="PharmaCheck · Qualité Intelligence",
    page_icon=str(ROOT / "assets/favicon.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)
PURPLE = "#791bff"
PAGES = ("Vue d’ensemble", "Lot individuel", "Historique")
ICONS = {
    "flask": '<path d="M9 3h6m-5 0v6l-6 10a1.4 1.4 0 0 0 1.2 2h13.6a1.4 1.4 0 0 0 1.2-2L14 9V3M8 15h8m-6-3h4"/>',
    "cube": '<path d="m12 2 9 5v10l-9 5-9-5V7Zm0 10 9-5M12 12 3 7m9 5v10M7.5 4.5l9 5v5"/>',
    "layers": '<path d="m12 3 10 6-10 6L2 9Zm-10 11 10 6 10-6M2 19l10 6 10-6" transform="translate(0 -2)"/>',
    "award": '<circle cx="12" cy="8" r="6"/><circle cx="12" cy="8" r="3.5"/><path d="m8 13-2 9 6-3 6 3-2-9"/>',
    "star": '<path d="m12 2 3 6.5 7 1-5 5 1 7-6-3.5L6 22l1-7-5-5 7-1Z"/>',
    "chart": '<path d="M3 12h4v9H3Zm7-7h4v16h-4Zm7-4h4v20h-4Z"/>',
    "table": '<rect x="3" y="2" width="18" height="20" rx="3"/><path d="M3 8h18M8 8v14m5-10h4m-4 4h4m-4 3h4"/>',
    "pie": '<path d="M11 2a10 10 0 1 0 11 11H11Zm4 0v7h7a9 9 0 0 0-7-7Z"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',
    "compression": '<path d="M12 2v13m-5-5 5 5 5-5M4 17v4h16v-4"/>',
    "droplet": '<path d="M12 2S4 11 4 16a8 8 0 0 0 16 0c0-5-8-14-8-14Z"/><path d="M8 15c0 3 1 4 4 4"/>',
    "scale": '<path d="M12 3v18M5 21h14M3 6h18M6 6l-4 9h8Zm12 0-4 9h8Z"/>',
    "check": '<path d="m5 12 5 5L20 6"/>',
    "alert": '<path d="m12 3 10 18H2ZM12 9v5m0 3v1"/>',
    "bell": '<path d="M18 8a6 6 0 0 0-12 0c0 7-3 8-3 9h18c0-1-3-2-3-9M9 21h6"/><circle cx="18" cy="4" r="3" fill="#9c39ff" stroke="none"/>',
    "history": '<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 3"/>',
}


def icon(name, size=24):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>')


def markup(content):
    st.markdown(content, unsafe_allow_html=True)


def apply_style():
    banner = ROOT / "assets/hero-pharmaceutical.png"
    if not banner.exists():
        banner = ROOT / "assets/hero.svg"
    image = base64.b64encode(banner.read_bytes()).decode()
    mime = "image/png" if banner.suffix == ".png" else "image/svg+xml"
    css = (ROOT / "assets/style.css").read_text(encoding="utf-8")
    for index, name in enumerate(("overview", "batch", "history"), start=1):
        data = base64.b64encode((ROOT / f"assets/navigation/{name}.svg").read_bytes()).decode()
        css += (f'\n[data-testid=stSidebar] [role=radiogroup] label:nth-child({index}):before'
                f'{{background-image:url("data:image/svg+xml;base64,{data}")}}')
    markup(f'<style>{css}\n.hero{{background-image:linear-gradient(90deg,#401474 0%,#40147400 50%),url("data:{mime};base64,{image}")}}</style>')


def topbar(page):
    markup(
        '<div class="topbar"><div class="topbar-left">'
        f'<div class="page-title">{html.escape(page)}</div>'
        f'<div class="workspace">{icon("flask", 19)} Espace de travail · Qualité &amp; Contrôle</div>'
        '</div><div class="topbar-right">'
        f'{icon("bell", 21)}<span class="topbar-divider"></span>'
        '<div class="profile"><b>Contrôle qualité</b></div>'
        '</div></div>'
    )


def hero(title, subtitle):
    markup(f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>')


def heading(name, detail="", symbol="chart", form=False):
    visual = f'<div class="icon-tile">{icon(symbol, 33)}</div>' if form else icon(symbol, 28)
    markup(f'<div class="panel-heading{" form-heading" if form else ""}">{visual}'
           f'<div><div class="panel-title">{html.escape(name)}</div>'
           f'<div class="panel-sub">{html.escape(detail)}</div></div></div>')


def stat(label, value, detail, symbol, green=False):
    markup(f'<div class="stat"><div class="icon-tile{" green" if green else ""}">{icon(symbol, 28)}</div>'
           f'<div><div class="stat-label">{html.escape(label)}</div>'
           f'<div class="stat-value">{html.escape(str(value))}</div>'
           f'<div class="stat-detail">{html.escape(detail)}</div></div></div>')


@st.cache_resource(show_spinner=False)
def cached_bundle(path, signature):
    """Invalidate the cache when the signed model metadata changes."""
    return load_bundle(Path(path))


def overview(meta, reference):
    recommended = meta["modele_recommande"]
    hero("La qualité, sous un nouvel angle.",
         "Analysez vos lots, comparez les modèles et explorez votre détection d’anomalies.")
    cards = [
        ("Lots d’apprentissage actifs", f"{len(reference):,}".replace(",", " "),
         "Données de la version de référence", "cube", False),
        ("Modèles disponibles", f"{len(MODEL_NAMES):02d}", "Random Forest · XGBoost · SVM", "layers", False),
        ("Meilleur F1 en validation croisée", f"{meta['modeles'][recommended]['cv']['f1']['moyenne']:.3f}",
         "Classe anomalie · critère de sélection", "award", False),
        ("Modèle recommandé", MODEL_NAMES[recommended],
         "Sélection sur l’apprentissage uniquement", "star", True),
    ]
    for col, card in zip(st.columns(4, gap="small"), cards):
        with col:
            stat(*card)

    with st.container(border=True, key="panel_performance"):
        left, right = st.columns([1.5, 1])
        with left:
            heading("Performance des modèles", "Validation croisée de la version active · moyenne et écart-type entre plis.")
        with right:
            selected = st.radio(
                "Métrique de performance", ["F1 moyen", "Rappel moyen", "Précision moyenne"],
                index=0, horizontal=True, key="performance_metric", label_visibility="collapsed",
            )
        selected = selected or "F1 moyen"
        primary = {"F1 moyen": "f1", "Rappel moyen": "recall", "Précision moyenne": "precision"}[selected]
        secondary, secondary_label = ("precision", "Précision moyenne") if primary == "recall" else ("recall", "Rappel moyen")
        fig = go.Figure()
        for metric, label, color in [(primary, selected, PURPLE), (secondary, secondary_label, "#d4bcf7")]:
            fig.add_bar(
                name=label, x=list(MODEL_NAMES.values()),
                y=[meta["modeles"][s]["cv"][metric]["moyenne"] for s in MODEL_NAMES],
                error_y=dict(array=[meta["modeles"][s]["cv"][metric]["ecart_type"] for s in MODEL_NAMES],
                             thickness=1, width=3, color="#7a638f"),
                marker=dict(color=color, cornerradius=5), width=.25,
                hovertemplate="%{x}<br>Score : %{y:.4f}<extra>%{fullData.name}</extra>",
            )
        fig.update_layout(
            height=220, barmode="group", bargap=.5, bargroupgap=.04,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=15, t=34, b=0), font=dict(color="#40226c", family="Inter, Segoe UI, sans-serif", size=11),
            legend=dict(orientation="h", y=1.24, x=.03, font=dict(color="#20103e", size=11)),
        )
        fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color="#1c0c42", size=12))
        fig.update_yaxes(range=[0, 1], dtick=.2, tickformat=".0%", title="Score", gridcolor="#ece2fc", zeroline=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        markup(f'<div class="chart-note">{icon("info", 16)} Ces scores servent au réglage. Les écarts-types ne sont pas des intervalles de confiance ; une évaluation indépendante reste nécessaire.</div>')

    left, right = st.columns([1.75, 1], gap="small")
    with left, st.container(border=True, key="panel_benchmark"):
        comparison = pd.read_csv(ROOT / "results/comparaison_modeles.csv")
        total = sum(pd.read_csv(ROOT / "data/PharmaCheck_test_20.csv")[TARGET].value_counts())
        heading("Benchmark initial", f"Évaluation historique sur {total} lots synthétiques · résultats de la version de référence.", "table")
        rows = []
        for record in comparison.to_dict("records"):
            chosen = record["modele"] == recommended
            badge = '<span class="badge">✓ Recommandé</span>' if chosen else ""
            rows.append(f'<tr class="{"recommended" if chosen else ""}">'
                        f'<td>{html.escape(record["nom"])}</td><td>{record["f1"]:.4f}</td>'
                        f'<td>{record["recall"]:.1%}</td><td>{record["faux_negatifs"]}</td>'
                        f'<td>{record["faux_positifs"]}</td><td>{badge}</td></tr>')
        markup('<table class="benchmark"><thead><tr><th>Modèle</th><th>F1 test</th>'
               '<th>Rappel test</th><th>Anomalies manquées</th><th>Fausses alertes</th><th></th>'
               '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>')
    with right, st.container(border=True, key="panel_composition"):
        heading("Composition de l’apprentissage", "", "pie")
        counts = reference[TARGET].value_counts().reindex([0, 1], fill_value=0)
        normal, anomaly = counts.iloc[0] / len(reference), counts.iloc[1] / len(reference)
        markup('<div class="composition"><div class="composition-legend">'
               f'<div class="legend-row"><span class="swatch light"></span>Normaux<b>{normal:.0%}</b></div>'
               f'<div class="legend-row"><span class="swatch"></span>Anomalies<b>{anomaly:.0%}</b></div>'
               f'</div><div class="donut" style="--normal:{normal:.1%}" role="img" aria-label="{normal:.0%} normaux, {anomaly:.0%} anomalies">'
               f'<div class="donut-label"><b>{anomaly:.0%}</b>anomalies</div></div></div>')


def measure_input(feature, reference, symbol=None):
    label = FEATURE_LABELS[feature]
    if symbol:
        # Keep the native, accessible label while decorating the measurement card.
        markup(f'<div class="measure-label">{icon(symbol, 23)}{html.escape(label)}</div>')
    return st.number_input(
        label,
        min_value=.001 if feature in [FEATURES[0], FEATURES[2], FEATURES[3]] else 0.,
        max_value=100. if feature in [FEATURES[1], FEATURES[4]] else None,
        value=round(float(reference[feature].median()), 3),
        step=.01, format="%.3f", key=f"measure_{feature}",
        label_visibility="collapsed" if symbol else "visible",
    )


def single(models, meta, reference):
    hero("Un lot. Huit mesures. Trois regards.",
         "Renseignez les mesures du lot pour obtenir les prédictions de trois modèles et une analyse de risque claire.")
    with st.form("single_batch"):
        heading("Données du lot", "Saisissez l’identifiant du lot et ses mesures de production pour analyser sa conformité.", "cube", form=True)
        batch = st.text_input("Identifiant du lot", value="MON-LOT-001", key="batch_id")
        values = {}
        for col, feature, symbol in zip(st.columns(3, gap="small"), FEATURES[:3], ["compression", "droplet", "scale"]):
            with col, st.container(border=True, key=f"card_{feature}"):
                values[feature] = measure_input(feature, reference, symbol)
        with st.expander("Les 5 autres mesures de production · ajuster les valeurs de référence"):
            st.caption("Les huit mesures sont utilisées par les modèles. Les valeurs préremplies sont les médianes de l’apprentissage, à remplacer par vos mesures réelles.")
            cols = st.columns(3)
            for i, feature in enumerate(FEATURES[3:]):
                with cols[i % 3]:
                    values[feature] = measure_input(feature, reference)
        submitted = st.form_submit_button("Analyser ce lot", icon=":material/play_arrow:", type="primary", width="stretch")
    if submitted:
        # Never leave a previous successful prediction visible after an invalid submission.
        st.session_state.pop("single_analysis", None)
        try:
            frame, _ = validate_batches(pd.DataFrame([{ID: batch, **values}]))
            result = predict_batches(models, frame)
            st.session_state.single_analysis = {
                "batch": str(frame.iloc[0][ID]), "result": result,
            }
            try:
                record_analysis(frame, result, meta["modele_recommande"],
                                file_hash(ROOT / "results/selection_modeles.json"))
            except (OSError, sqlite3.Error) as exc:
                st.warning(f"Analyse terminée, mais l’enregistrement dans l’historique a échoué : {exc}")
            else:
                st.caption("Analyse enregistrée dans l’historique. Retrouvez toutes vos mesures et prédictions sur la page Historique.")
        except ValueError as exc:
            st.error(str(exc))
    analysis = st.session_state.get("single_analysis")
    if analysis is None:
        markup(f'<div class="result-placeholder">{icon("cube", 37)}<div><b>Votre prochain résultat commence ici.</b>'
               '<p>Renseignez les mesures, puis cliquez sur « Analyser ce lot ».</p></div></div>')
        return
    result = analysis["result"]
    row = result.iloc[0]
    recommended = meta["modele_recommande"]
    alert = bool(row[f"prediction_{recommended}"])
    status = "À examiner" if alert else "Conforme"
    markup(f'<div class="result{" alert" if alert else ""}" role="status">'
           f'<div class="result-symbol">{icon("alert" if alert else "check", 66)}</div><div>'
           f'<h2>Résultat final : <span>{status}</span></h2>'
           f'<p>{"Une anomalie est prédite pour ce lot." if alert else "Aucune anomalie prédite pour ce lot."}</p>'
           f'<small>Lot {html.escape(analysis["batch"])} · Évaluation par {html.escape(MODEL_NAMES[recommended])}, modèle recommandé.</small>'
           '</div></div>')
    if bool(row["desaccord_modeles"]):
        st.warning("Les modèles sont en désaccord. Le résultat final suit le modèle recommandé ; examinez les trois prédictions ci-dessous.")
    with st.expander("Voir les prédictions des trois modèles"):
        for col, slug in zip(st.columns(3), MODEL_NAMES):
            with col, st.container(border=True):
                heading(MODEL_NAMES[slug], "Modèle recommandé" if slug == recommended else "Modèle de comparaison", "layers")
                (st.warning if bool(row[f"prediction_{slug}"]) else st.success)(
                    "Anomalie prédite" if bool(row[f"prediction_{slug}"]) else "Aucune anomalie prédite")
                st.metric("Score de décision" if slug == "svm" else "Probabilité non calibrée", f"{row[f'score_{slug}']:.4f}")
        st.caption("Le score SVM n’est pas une probabilité ; les scores des trois modèles ne se comparent pas directement.")
        st.download_button("Télécharger le résultat", result.to_csv(index=False).encode("utf-8-sig"),
                           "prediction_lot.csv", mime="text/csv", on_click="ignore")
    st.caption("« Conforme » signifie ici qu’aucune anomalie n’est prédite par le modèle recommandé. Cette prédiction ne constitue pas une validation pharmaceutique.")


def remove_history_entries(analysis_ids):
    try:
        count = delete_history(analysis_ids)
    except (OSError, sqlite3.Error) as exc:
        st.session_state.history_error = f"Suppression impossible : {exc}"
    else:
        st.session_state.history_delete_selection = None
        st.session_state.history_notice = f"{count} analyse(s) supprimée(s)."


def undo_history_entries(deletion_id):
    try:
        count = restore_history(deletion_id)
    except (OSError, sqlite3.Error) as exc:
        st.session_state.history_error = f"Restauration impossible : {exc}"
    else:
        st.session_state.history_notice = f"{count} analyse(s) restaurée(s)."


def history():
    hero("Chaque analyse, une trace.",
         "Retrouvez les résultats de vos lots et téléchargez l’historique complet de vos analyses.")
    if notice := st.session_state.pop("history_notice", None):
        st.success(notice)
    if error := st.session_state.pop("history_error", None):
        st.error(error)
    try:
        records = read_history()
        deletion = latest_history_deletion()
    except (OSError, ValueError, sqlite3.Error) as exc:
        st.error(f"Historique indisponible : {exc}")
        return
    if deletion:
        st.button(f"Annuler la dernière suppression ({deletion[1]} analyse(s))", icon=":material/undo:",
                  key="undo_history_delete", on_click=undo_history_entries, args=(deletion[0],))
    cards = [
        ("Analyses enregistrées", str(len(records)), "Tous les contrôles sauvegardés", "history", False),
        ("Résultats conformes", str(records.resultat_final.eq("Conforme").sum()), "Aucune anomalie prédite", "check", True),
        ("Lots à examiner", str(records.resultat_final.eq("À examiner").sum()), "Anomalie prédite par le modèle recommandé", "alert", False),
    ]
    for col, card in zip(st.columns(3, gap="small"), cards):
        with col:
            stat(*card)
    with st.container(border=True, key="panel_history"):
        heading("Historique des résultats", "Analyses les plus récentes en premier · sauvegarde locale, conservée après un rechargement.", "history")
        if records.empty:
            st.info("Aucune analyse enregistrée pour le moment. Analysez un lot depuis la page Lot individuel pour commencer votre historique.")
        else:
            displayed = records[["analyse_le_utc", ID, "resultat_final", "modele_recommande", "desaccord_modeles"]].copy()
            displayed["analyse_le_utc"] = pd.to_datetime(displayed.analyse_le_utc, utc=True).dt.tz_convert("Africa/Casablanca").dt.strftime("%d/%m/%Y %H:%M:%S")
            displayed = displayed.rename(columns={
                "analyse_le_utc": "Date · Casablanca", ID: "Identifiant du lot", "resultat_final": "Résultat final",
                "modele_recommande": "Modèle recommandé", "desaccord_modeles": "Désaccord entre modèles",
            })
            displayed["Désaccord entre modèles"] = displayed["Désaccord entre modèles"].map({True: "Oui", False: "Non"})
            with st.container(key="history_table"):
                st.table(displayed.style.hide(axis="index"))
        st.download_button(
            "Télécharger l’historique (CSV)", history_csv(records), "historique_pharmacheck.csv",
            mime="text/csv", icon=":material/download:", key="download_history",
            disabled=records.empty, on_click="ignore", width="stretch",
        )
        if not records.empty:
            st.divider()
            labels = {
                row.id_analyse: f"{displayed.iloc[i]['Date · Casablanca']} · {row[ID]} · {row.resultat_final} · {row.id_analyse[:8]}"
                for i, (_, row) in enumerate(records.iterrows())
            }
            if st.session_state.get("history_delete_selection") not in labels:
                st.session_state.history_delete_selection = None
            selected = st.selectbox(
                "Analyse à supprimer", list(labels), index=None, format_func=labels.get,
                placeholder="Choisir une analyse…", key="history_delete_selection",
            )
            left, right = st.columns(2)
            left.button("Supprimer cette ligne", icon=":material/delete:", disabled=selected is None,
                        width="stretch", key="delete_history_row", on_click=remove_history_entries, args=([selected],))
            right.button("Vider l’historique", icon=":material/delete_sweep:", width="stretch", key="clear_history",
                         on_click=remove_history_entries, args=(None,))


def main():
    apply_style()
    # Reset routes removed from older app versions.
    if st.session_state.get("navigation") not in PAGES:
        st.session_state.navigation = PAGES[0]
    with st.sidebar:
        brand = base64.b64encode((ROOT / "assets/pharmacheck-brand.png").read_bytes()).decode()
        markup(f'<div class="brand"><img src="data:image/png;base64,{brand}" '
               'alt="PharmaCheck · Qualité Intelligence" /></div>')
        markup('<div class="side-label">ESPACE DE TRAVAIL</div>')
        page = st.radio("Navigation", PAGES, key="navigation", label_visibility="collapsed")
        markup('<div class="sidebar-note"><span>Un regard sur chaque lot.</span>8 mesures de production<br>3 modèles complémentaires</div>')
        markup(f'<div class="studio"><div class="studio-icon">{icon("flask", 24)}</div><div>PharmaCheck Studio'
               '<small>Projet académique — v2.1</small></div></div>')
    topbar(page)
    try:
        models, meta = cached_bundle(str(ROOT), file_hash(ROOT / "results/selection_modeles.json"))
        reference = pd.read_csv(training_path(ROOT), dtype={ID: str})
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Version indisponible : {exc}")
        st.stop()
    if page == PAGES[0]:
        overview(meta, reference)
    elif page == PAGES[1]:
        single(models, meta, reference)
    else:
        history()


if __name__ == "__main__":
    main()
