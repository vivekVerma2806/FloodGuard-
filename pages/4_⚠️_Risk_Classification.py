"""FloodGuard AI — Risk Classification Page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import DATA_DIR, FEATURES, TARGET_CLASSIFICATION, RISK_CATEGORIES, TEST_SPLIT_RATIO
from src.data_preprocessing import load_data, clean_data, engineer_features, chronological_split
from src.risk_classification import (
    assign_risk_labels, train_risk_classifier,
    evaluate_classification, get_classification_report_df,
    get_confusion_matrix_df,
)

st.set_page_config(page_title="Risk Classification | FloodGuard AI", page_icon="⚠️", layout="wide")


def kpi_card(title, value, delta=None):
    delta_html = ""
    if delta is not None:
        delta_html = f'<p style="margin:0; color:#666; font-size:0.9rem;">{delta}</p>'
    st.markdown(
        f"""
        <div style="padding: 1rem; border-radius: 0.5rem; border: 1px solid #e0e0e0;
                    background: #f8f9fa; text-align: center;">
            <h4 style="margin:0; color:#666;">{title}</h4>
            <h2 style="margin:0; color:#1f77b4;">{value}</h2>
            {delta_html}
        </div>""",
        unsafe_allow_html=True,
    )


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
    st.title("⚠️ Risk Classification")
    st.markdown("Train a Random Forest classifier to categorise flood-risk levels.")

    st.info(
        "⚠️ **Note:** Risk labels are generated from demo thresholds on river level, "
        "not verified ground-truth labels. Treat results as illustrative.",
        icon="ℹ️",
    )
    st.markdown("---")

    df = _get_data()
    if df is None:
        st.warning("No data available. Load data from the **📊 Data Upload** page.")
        return

    # Ensure risk labels
    if TARGET_CLASSIFICATION not in df.columns:
        df = assign_risk_labels(df)

    # ── Risk distribution (always shown) ─────────────────────────────────
    st.subheader("📊 Risk Distribution")
    risk_counts = df[TARGET_CLASSIFICATION].value_counts().reindex(RISK_CATEGORIES).fillna(0)
    color_map = {"Low": "#2ca02c", "Moderate": "#ff7f0e", "High": "#d62728", "Critical": "#7f0000"}
    fig_pie = px.pie(
        names=risk_counts.index, values=risk_counts.values,
        title="Flood Risk Distribution",
        color=risk_counts.index,
        color_discrete_map=color_map,
        template="plotly_white",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # ── Train Classifier ─────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🚀 Train Risk Classifier", type="primary"):
        with st.spinner("Training classifier..."):
            try:
                available = [f for f in FEATURES if f in df.columns and f != "river_level_m"]
                df_model = df.dropna(subset=available + [TARGET_CLASSIFICATION])

                train_df, test_df = chronological_split(df_model, TEST_SPLIT_RATIO)
                X_train = train_df[available].values
                y_train = train_df[TARGET_CLASSIFICATION].values
                X_test = test_df[available].values
                y_test = test_df[TARGET_CLASSIFICATION].values

                clf = train_risk_classifier(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)

                metrics = evaluate_classification(y_test, y_pred, y_prob, labels=RISK_CATEGORIES)

                st.session_state["risk_metrics"] = metrics
                st.session_state["risk_y_test"] = y_test
                st.session_state["risk_y_pred"] = y_pred
                st.session_state["risk_y_prob"] = y_prob

                st.success("✅ Classifier trained!")
            except Exception as e:
                st.error(f"Training failed: {e}")
                return

    # ── Results ──────────────────────────────────────────────────────────
    if "risk_metrics" in st.session_state:
        metrics = st.session_state["risk_metrics"]
        y_test = st.session_state["risk_y_test"]
        y_pred = st.session_state["risk_y_pred"]

        st.markdown("---")
        st.subheader("📋 Metrics")

        cols = st.columns(4)
        prec = metrics.get("precision", "N/A")
        rec = metrics.get("recall", "N/A")
        f1 = metrics.get("f1", "N/A")
        pr_auc = metrics.get("pr_auc", {}).get("macro_avg", "N/A")

        with cols[0]:
            kpi_card("Precision (Macro)", f"{prec:.4f}" if isinstance(prec, float) else prec)
        with cols[1]:
            kpi_card("Recall (Macro)", f"{rec:.4f}" if isinstance(rec, float) else rec)
        with cols[2]:
            kpi_card("F1-Score (Macro)", f"{f1:.4f}" if isinstance(f1, float) else f1)
        with cols[3]:
            kpi_card("PR-AUC (Macro)", f"{pr_auc:.4f}" if isinstance(pr_auc, float) else pr_auc)

        # Confusion matrix
        st.markdown("")
        st.subheader("🔥 Confusion Matrix")
        try:
            cm_df = get_confusion_matrix_df(y_test, y_pred, labels=RISK_CATEGORIES)
            fig_cm = px.imshow(
                cm_df.values,
                x=cm_df.columns.tolist(),
                y=cm_df.index.tolist(),
                text_auto=True,
                color_continuous_scale="Blues",
                title="Confusion Matrix",
                labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render confusion matrix: {e}")

        # Classification report
        st.subheader("📄 Classification Report")
        try:
            report_df = get_classification_report_df(y_test, y_pred, labels=RISK_CATEGORIES)
            st.dataframe(report_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render report: {e}")


try:
    main()
except Exception as exc:
    st.error(f"An unexpected error occurred: {exc}")
