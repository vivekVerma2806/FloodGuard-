"""FloodGuard AI — Model Benchmarking Page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import DATA_DIR, FEATURES, TARGET_REGRESSION, TEST_SPLIT_RATIO
from src.data_preprocessing import load_data, clean_data, engineer_features, chronological_split
from src.ml_benchmarking import (
    compare_models, get_feature_importance,
    evaluate_regression,
)

st.set_page_config(page_title="Model Benchmarking | FloodGuard AI", page_icon="📈", layout="wide")


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
    st.title("📈 Model Benchmarking")
    st.markdown(
        "Compare Linear Regression, Random Forest, and Gradient Boosting models "
        "for river-level prediction."
    )
    st.markdown("---")

    df = _get_data()
    if df is None:
        st.warning("No data available. Load data from the **📊 Data Upload** page.")
        return

    # ── Station selector ─────────────────────────────────────────────────
    stations = sorted(df["station_name"].unique())
    selected = st.selectbox("Select monitoring station:", stations)
    sdf = df[df["station_name"] == selected].sort_values("date").reset_index(drop=True)

    available = [f for f in FEATURES if f in sdf.columns and f != TARGET_REGRESSION]
    if len(available) < 1:
        st.error("Insufficient features. Run feature engineering first.")
        return

    if st.button("🚀 Run Benchmark", type="primary"):
        with st.spinner("Training & evaluating models..."):
            try:
                sdf_clean = sdf.dropna(subset=available + [TARGET_REGRESSION])
                train_df, test_df = chronological_split(sdf_clean, TEST_SPLIT_RATIO)

                X_train = train_df[available].values
                y_train = train_df[TARGET_REGRESSION].values
                X_test = test_df[available].values
                y_test = test_df[TARGET_REGRESSION].values

                results_df, models_dict = compare_models(X_train, y_train, X_test, y_test)

                st.session_state["bench_results"] = results_df
                st.session_state["bench_models"] = models_dict
                st.session_state["bench_X_test"] = X_test
                st.session_state["bench_y_test"] = y_test
                st.session_state["bench_features"] = available
                st.session_state["bench_test_dates"] = test_df["date"].values

                st.success("✅ Benchmarking complete!")
            except Exception as e:
                st.error(f"Benchmarking failed: {e}")
                return

    # ── Results ──────────────────────────────────────────────────────────
    if "bench_results" in st.session_state:
        results_df = st.session_state["bench_results"]
        models_dict = st.session_state["bench_models"]
        X_test = st.session_state["bench_X_test"]
        y_test = st.session_state["bench_y_test"]
        features = st.session_state["bench_features"]

        st.markdown("---")
        st.subheader("📋 Comparison Table")
        st.dataframe(results_df.style.highlight_min(subset=["MAE", "RMSE"], color="#d4edda"),
                      use_container_width=True)

        # Bar chart
        st.subheader("📊 Model Comparison")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="MAE", x=results_df["Model"], y=results_df["MAE"]))
        fig_bar.add_trace(go.Bar(name="RMSE", x=results_df["Model"], y=results_df["RMSE"]))
        fig_bar.update_layout(
            barmode="group", template="plotly_white",
            title="MAE & RMSE by Model", yaxis_title="Error",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Feature importance
        st.subheader("🔑 Feature Importance")
        fi_tabs = st.tabs(["Random Forest", "Gradient Boosting"])
        for tab, model_name in zip(fi_tabs, ["Random Forest", "Gradient Boosting"]):
            with tab:
                if model_name in models_dict:
                    try:
                        trained_model = models_dict[model_name][0]
                        fi_df = get_feature_importance(trained_model, features)
                        fig_fi = px.bar(
                            fi_df.sort_values("Importance", ascending=True),
                            x="Importance", y="Feature", orientation="h",
                            title=f"{model_name} Feature Importance",
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_fi, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not compute feature importance: {e}")

        # Scatter: actual vs predicted
        st.subheader("🎯 Actual vs Predicted")
        scatter_tabs = st.tabs(list(models_dict.keys()))
        for tab, (mname, (model_obj, y_pred)) in zip(scatter_tabs, models_dict.items()):
            with tab:
                try:
                    fig_sc = px.scatter(
                        x=y_test, y=y_pred,
                        labels={"x": "Actual", "y": "Predicted"},
                        title=f"{mname}: Actual vs Predicted",
                        template="plotly_white",
                    )
                    # Perfect prediction line
                    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                    fig_sc.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
                                                mode="lines", name="Ideal",
                                                line=dict(dash="dash", color="gray")))
                    st.plotly_chart(fig_sc, use_container_width=True)
                except Exception as e:
                    st.warning(f"Scatter plot failed for {mname}: {e}")

        # Download
        st.markdown("---")
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Benchmark Results (CSV)", csv_bytes,
            file_name="benchmark_results.csv", mime="text/csv",
        )


try:
    main()
except Exception as exc:
    st.error(f"An unexpected error occurred: {exc}")
