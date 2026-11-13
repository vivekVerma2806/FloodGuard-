"""FloodGuard AI - Statistical Model Benchmarking."""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import DATA_DIR, FEATURES, TARGET_REGRESSION, TEST_SPLIT_RATIO
from src.data_preprocessing import chronological_split, clean_data, engineer_features, load_data
from src.ml_benchmarking import compare_models, evaluate_regression, get_feature_importance
from src.ui_theme import (
    apply_theme,
    render_sidebar,
    editorial_header,
    kpi_card,
    editorial_callout,
    render_html,
    style_plotly_chart,
    TEXT_MUTED,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Model Benchmarking | FloodGuard AI",
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
        title="Statistical Model Benchmarking",
        kicker="REGRESSION EVALUATION // COMPARATIVE AUDIT",
        subtitle=(
            "Rigorous empirical performance comparison of Ordinary Least Squares, "
            "Random Forest, and Gradient Boosting estimators against physical hydrological telemetries."
        ),
    )

    df = _get_data()
    if df is None:
        editorial_callout(
            "Hydrological observations unavailable. Load data via the Data Upload module first.",
            label="TELEMETRY STATUS // MISSING",
            level="warning",
        )
        return

    # -------------------------------------------------------------------------
    # Sentinel Basin Selection & Configuration
    # -------------------------------------------------------------------------
    col_stn, col_ratio = st.columns([2, 1])
    with col_stn:
        stations = sorted(df["station_name"].unique())
        selected = st.selectbox("Sentinel Gauging Station", stations)
    with col_ratio:
        test_pct = st.selectbox("Hold-out Test Proportion", ["20% (Strict Temporal)", "15%", "25%"], index=0)

    ratio_val = 0.20 if "20%" in test_pct else (0.15 if "15%" in test_pct else 0.25)
    sdf = df[df["station_name"] == selected].sort_values("date").reset_index(drop=True)

    available_features = [f for f in FEATURES if f in sdf.columns and f != TARGET_REGRESSION]
    if len(available_features) < 1:
        editorial_callout("Insufficient attributes for regression modeling. Run feature engineering.", label="ATTRIBUTE DEFICIT", level="warning")
        return

    render_html(
        f"""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {TEXT_MUTED}; margin-bottom: 1rem;">
            // PREDICTORS ({len(available_features)}): {', '.join(available_features)}
        </div>
        """
    )

    # -------------------------------------------------------------------------
    # Execution Button
    # -------------------------------------------------------------------------
    if st.button("Execute Comparative Benchmark"):
        with st.spinner("Fitting ensemble and linear estimators on training partition..."):
            try:
                sdf_clean = sdf.dropna(subset=available_features + [TARGET_REGRESSION])
                train_df, test_df = chronological_split(sdf_clean, ratio_val)

                X_train = train_df[available_features].values
                y_train = train_df[TARGET_REGRESSION].values
                X_test = test_df[available_features].values
                y_test = test_df[TARGET_REGRESSION].values

                results_df, models_dict = compare_models(X_train, y_train, X_test, y_test)

                st.session_state["bench_results"] = results_df
                st.session_state["bench_models"] = models_dict
                st.session_state["bench_X_test"] = X_test
                st.session_state["bench_y_test"] = y_test
                st.session_state["bench_features"] = available_features
                st.session_state["bench_test_dates"] = test_df["date"].values

                editorial_callout("Benchmarking complete. All estimators evaluated against identical chronological test window.", label="BENCHMARK EXECUTED", level="info")
            except Exception as exc:
                st.error(f"Benchmarking execution failed: {exc}")
                return

    # -------------------------------------------------------------------------
    # Comparative Results Display
    # -------------------------------------------------------------------------
    if "bench_results" in st.session_state:
        results_df = st.session_state["bench_results"]
        models_dict = st.session_state["bench_models"]
        X_test = st.session_state["bench_X_test"]
        y_test = st.session_state["bench_y_test"]
        features = st.session_state["bench_features"]

        st.markdown("---")
        render_html(
            f"""
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; color: {TEXT_MUTED}; margin-bottom: 0.75rem;">
                // STATISTICAL ESTIMATOR COMPARISON LEDGER
            </div>
            """
        )

        best_row = results_df.iloc[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Best Performing Model", best_row["Model"], caption="Minimized Root Mean Square Error")
        with c2:
            kpi_card("Minimum Test RMSE", f"{best_row['RMSE']:.4f}", caption="Deviation from gauge datum")
        with c3:
            kpi_card("Minimum Test MAE", f"{best_row['MAE']:.4f}", caption="Mean absolute error")

        render_html("<div style='height: 12px;'></div>")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # ---------------------------------------------------------------------
        # Visual Error Comparison
        # ---------------------------------------------------------------------
        render_html("<div style='height: 16px;'></div>")
        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                name="Mean Absolute Error (MAE)",
                x=results_df["Model"],
                y=results_df["MAE"],
                marker_color="#171717",
            )
        )
        fig_bar.add_trace(
            go.Bar(
                name="Root Mean Square Error (RMSE)",
                x=results_df["Model"],
                y=results_df["RMSE"],
                marker_color="#B45309",
            )
        )
        fig_bar.update_layout(
            barmode="group",
            title="Out-of-Sample Error Disparity by Algorithm",
            yaxis_title="Error Magnitude (m)",
        )
        fig_bar = style_plotly_chart(fig_bar, height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

        # ---------------------------------------------------------------------
        # Feature Importance & Fit Analysis
        # ---------------------------------------------------------------------
        st.markdown("---")
        tab_importance, tab_fit = st.tabs(["Gini Feature Importance Analysis", "Observed vs. Model Scatter Fit"])

        with tab_importance:
            col_rf, col_gb = st.columns(2)
            with col_rf:
                render_html(
                    f"""
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {TEXT_MUTED}; margin-bottom: 0.5rem;">
                        // RANDOM FOREST GINI WEIGHTS
                    </div>
                    """
                )
                if "Random Forest" in models_dict:
                    rf_model = models_dict["Random Forest"][0]
                    fi_rf = get_feature_importance(rf_model, features)
                    fig_fi_rf = px.bar(
                        fi_rf.sort_values("Importance", ascending=True),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        template="none",
                    )
                    fig_fi_rf.update_traces(marker_color="#171717")
                    fig_fi_rf = style_plotly_chart(fig_fi_rf, height=320)
                    st.plotly_chart(fig_fi_rf, use_container_width=True)

            with col_gb:
                render_html(
                    f"""
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {TEXT_MUTED}; margin-bottom: 0.5rem;">
                        // GRADIENT BOOSTING RESIDUAL WEIGHTS
                    </div>
                    """
                )
                if "Gradient Boosting" in models_dict:
                    gb_model = models_dict["Gradient Boosting"][0]
                    fi_gb = get_feature_importance(gb_model, features)
                    fig_fi_gb = px.bar(
                        fi_gb.sort_values("Importance", ascending=True),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        template="none",
                    )
                    fig_fi_gb.update_traces(marker_color="#B45309")
                    fig_fi_gb = style_plotly_chart(fig_fi_gb, height=320)
                    st.plotly_chart(fig_fi_gb, use_container_width=True)

        with tab_fit:
            scatter_tabs = st.tabs(list(models_dict.keys()))
            for s_tab, (mname, (model_obj, y_pred)) in zip(scatter_tabs, models_dict.items()):
                with s_tab:
                    fig_sc = go.Figure()
                    fig_sc.add_trace(
                        go.Scatter(
                            x=y_test,
                            y=y_pred,
                            mode="markers",
                            name="Predicted vs. Observed",
                            marker=dict(size=5, color="#171717", opacity=0.6),
                        )
                    )
                    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                    fig_sc.add_trace(
                        go.Scatter(
                            x=[mn, mx],
                            y=[mn, mx],
                            mode="lines",
                            name="Ideal 1:1 Identity",
                            line=dict(color="#B45309", dash="dash", width=1.5),
                        )
                    )
                    fig_sc.update_layout(
                        title=f"{mname} Linear Fit Agreement",
                        xaxis_title="Observed Gauge Stage (m)",
                        yaxis_title="Estimated Stage (m)",
                    )
                    fig_sc = style_plotly_chart(fig_sc, height=380)
                    st.plotly_chart(fig_sc, use_container_width=True)

        # Export benchmark results
        st.markdown("---")
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Benchmark Ledger (.CSV)",
            csv_bytes,
            file_name="floodguard_benchmark_ledger.csv",
            mime="text/csv",
        )


try:
    main()
except Exception as exc:
    st.error(f"Runtime Exception in Model Benchmarking module: {exc}")
