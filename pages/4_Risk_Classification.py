"""FloodGuard AI - Hydrological Risk Tier Classification."""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import DATA_DIR, FEATURES, RISK_CATEGORIES, TARGET_CLASSIFICATION, TEST_SPLIT_RATIO
from src.data_preprocessing import chronological_split, clean_data, engineer_features, load_data
from src.risk_classification import (
    assign_risk_labels,
    evaluate_classification,
    get_classification_report_df,
    get_confusion_matrix_df,
    train_risk_classifier,
)
from src.ui_theme import (
    apply_theme,
    render_sidebar,
    editorial_header,
    kpi_card,
    editorial_callout,
    render_html,
    style_plotly_chart,
    RISK_COLORS,
    RISK_PALETTE,
    TEXT_MUTED,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Risk Classification | FloodGuard AI",
    page_icon="☵",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
render_sidebar()


def _get_data():
    if "processed_df" in st.session_state:
        return st.session_state["processed_df"]
    demo_path = os.path.join(DATA_DIR, "demo_dataset.csv")
    if os.path.exists(demo_path):
        df = load_data(demo_path)
        df = clean_data(df)
        df = engineer_features(df)
        return df
    return None


def main():
    editorial_header(
        title="Hydrological Threat Classification",
        kicker="HAZARD TAXONOMY // MULTI-CLASS INFERENCE",
        subtitle=(
            "Categorization of flood threat severity into four standardized tiers (Low, Moderate, High, Critical) "
            "using ensemble decision trees evaluated with Precision-Recall AUC and confusion matrices."
        ),
    )

    df = _get_data()
    if df is None:
        editorial_callout(
            "Observations ledger unavailable. Initialize data via the Data Upload module.",
            label="TELEMETRY STATUS // MISSING",
            level="warning",
        )
        return

    if TARGET_CLASSIFICATION not in df.columns:
        df = assign_risk_labels(df)

    # -------------------------------------------------------------------------
    # Standardized Risk Tier Definitions
    # -------------------------------------------------------------------------
    render_html(
        f"""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; color: {TEXT_MUTED}; margin-bottom: 0.6rem;">
            // OPERATIONAL HYDROMETRIC THREAT TIERS
        </div>
        """
    )

    c_low, c_mod, c_high, c_crit = st.columns(4)
    with c_low:
        render_html(
            f"""
            <div style="border-left: 3px solid {RISK_COLORS['Low']}; background: #FFFFFF; border-top: 1px solid #D6D3CC; border-right: 1px solid #D6D3CC; border-bottom: 1px solid #D6D3CC; padding: 0.75rem 1rem;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: {RISK_COLORS['Low']}; font-weight: 600;">TIER I // LOW</div>
                <div style="font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 700; margin-top: 2px;">Normal Stage</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #666; margin-top: 2px;">Elevation &lt; 3.0 m</div>
            </div>
            """
        )
    with c_mod:
        render_html(
            f"""
            <div style="border-left: 3px solid {RISK_COLORS['Moderate']}; background: #FFFFFF; border-top: 1px solid #D6D3CC; border-right: 1px solid #D6D3CC; border-bottom: 1px solid #D6D3CC; padding: 0.75rem 1rem;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: {RISK_COLORS['Moderate']}; font-weight: 600;">TIER II // MODERATE</div>
                <div style="font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 700; margin-top: 2px;">Advisory Stage</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #666; margin-top: 2px;">3.0 m &le; Elevation &lt; 5.0 m</div>
            </div>
            """
        )
    with c_high:
        render_html(
            f"""
            <div style="border-left: 3px solid {RISK_COLORS['High']}; background: #FFFFFF; border-top: 1px solid #D6D3CC; border-right: 1px solid #D6D3CC; border-bottom: 1px solid #D6D3CC; padding: 0.75rem 1rem;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: {RISK_COLORS['High']}; font-weight: 600;">TIER III // HIGH</div>
                <div style="font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 700; margin-top: 2px;">Warning Stage</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #666; margin-top: 2px;">5.0 m &le; Elevation &lt; 7.0 m</div>
            </div>
            """
        )
    with c_crit:
        render_html(
            f"""
            <div style="border-left: 3px solid {RISK_COLORS['Critical']}; background: #FFFFFF; border-top: 1px solid #D6D3CC; border-right: 1px solid #D6D3CC; border-bottom: 1px solid #D6D3CC; padding: 0.75rem 1rem;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: {RISK_COLORS['Critical']}; font-weight: 600;">TIER IV // CRITICAL</div>
                <div style="font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 700; margin-top: 2px;">Inundation Hazard</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #666; margin-top: 2px;">Elevation &ge; 7.0 m</div>
            </div>
            """
        )

    render_html("<div style='height: 16px;'></div>")

    # -------------------------------------------------------------------------
    # Observation Distribution
    # -------------------------------------------------------------------------
    risk_counts = df[TARGET_CLASSIFICATION].value_counts().reindex(RISK_CATEGORIES).fillna(0)
    fig_dist = go.Figure()
    fig_dist.add_trace(
        go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            marker_color=[RISK_COLORS[c] for c in risk_counts.index],
            text=[f"{v:,} ({v/len(df)*100:.1f}%)" for v in risk_counts.values],
            textposition="auto",
        )
    )
    fig_dist.update_layout(
        title="Distribution of Historical Observations Across Threat Tiers",
        xaxis_title="Risk Tier",
        yaxis_title="Total Historical Days",
    )
    fig_dist = style_plotly_chart(fig_dist, height=320)
    st.plotly_chart(fig_dist, use_container_width=True)

    # -------------------------------------------------------------------------
    # Train Classification Pipeline
    # -------------------------------------------------------------------------
    st.markdown("---")
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        btn_train_clf = st.button("Train Threat Classifier")
    with col_info:
        st.caption("Fits a balanced Random Forest ensemble on chronological train partition (80%) without future data leakage.")

    if btn_train_clf:
        with st.spinner("Training classification forest and generating out-of-sample predictions..."):
            try:
                available_feats = [f for f in FEATURES if f in df.columns and f != "river_level_m"]
                df_model = df.dropna(subset=available_feats + [TARGET_CLASSIFICATION])

                train_df, test_df = chronological_split(df_model, TEST_SPLIT_RATIO)
                X_train = train_df[available_feats].values
                y_train = train_df[TARGET_CLASSIFICATION].values
                X_test = test_df[available_feats].values
                y_test = test_df[TARGET_CLASSIFICATION].values

                clf = train_risk_classifier(X_train, y_train, n_estimators=100)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)

                metrics = evaluate_classification(y_test, y_pred, y_prob, labels=RISK_CATEGORIES)

                st.session_state["risk_metrics"] = metrics
                st.session_state["risk_y_test"] = y_test
                st.session_state["risk_y_pred"] = y_pred
                st.session_state["risk_y_prob"] = y_prob

                editorial_callout("Threat classification model optimized and validated out-of-sample.", label="CLASSIFIER READY", level="info")
            except Exception as exc:
                st.error(f"Classification training failed: {exc}")
                return

    # -------------------------------------------------------------------------
    # Evaluation Metrics & Confusion Matrix
    # -------------------------------------------------------------------------
    if "risk_metrics" in st.session_state:
        metrics = st.session_state["risk_metrics"]
        y_test = st.session_state["risk_y_test"]
        y_pred = st.session_state["risk_y_pred"]

        st.markdown("---")
        render_html(
            f"""
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; color: {TEXT_MUTED}; margin-bottom: 0.75rem;">
                // OUT-OF-SAMPLE CLASSIFICATION METRICS
            </div>
            """
        )

        cols = st.columns(4)
        prec = metrics.get("precision", "N/A")
        rec = metrics.get("recall", "N/A")
        f1 = metrics.get("f1", "N/A")
        pr_auc = metrics.get("pr_auc", {}).get("macro_avg", "N/A") if isinstance(metrics.get("pr_auc"), dict) else "N/A"

        with cols[0]:
            kpi_card("Precision (Macro)", f"{prec:.4f}" if isinstance(prec, float) else prec, caption="Reliability of positive hazard warnings")
        with cols[1]:
            kpi_card("Recall (Macro)", f"{rec:.4f}" if isinstance(rec, float) else rec, caption="Coverage of actual hazardous events")
        with cols[2]:
            kpi_card("F1-Score (Macro)", f"{f1:.4f}" if isinstance(f1, float) else f1, caption="Harmonic balance across tiers")
        with cols[3]:
            kpi_card("PR-AUC (Macro)", f"{pr_auc:.4f}" if isinstance(pr_auc, float) else pr_auc, caption="Precision-Recall curve integral")

        # Visualizations
        render_html("<div style='height: 16px;'></div>")
        tab_cm, tab_report = st.tabs(["Confusion Matrix Heatmap", "Comprehensive Classification Report"])

        with tab_cm:
            cm_df = get_confusion_matrix_df(y_test, y_pred, labels=RISK_CATEGORIES)
            # Vintage slate/charcoal heatmap color scale
            fig_cm = px.imshow(
                cm_df.values,
                x=cm_df.columns.tolist(),
                y=cm_df.index.tolist(),
                text_auto=True,
                color_continuous_scale=[[0.0, "#FAF8F5"], [0.5, "#A8A29E"], [1.0, "#171717"]],
                title="Classification Confusion Matrix (Actual vs. Estimated)",
                labels=dict(x="Predicted Risk Tier", y="Ground Truth Tier", color="Observation Count"),
            )
            fig_cm = style_plotly_chart(fig_cm, height=400)
            st.plotly_chart(fig_cm, use_container_width=True)

        with tab_report:
            report_df = get_classification_report_df(y_test, y_pred, labels=RISK_CATEGORIES)
            st.dataframe(report_df, use_container_width=True)


try:
    main()
except Exception as exc:
    st.error(f"Runtime Exception in Risk Classification module: {exc}")
