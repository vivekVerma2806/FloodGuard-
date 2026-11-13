"""FloodGuard AI — Model Evaluation & Report Page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import io

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

st.set_page_config(page_title="Model Evaluation | FloodGuard AI", page_icon="📋", layout="wide")


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


def _build_report_text(lstm_metrics, bench_df, risk_metrics):
    """Build a plain-text evaluation report."""
    lines = [
        "=" * 60,
        "FloodGuard AI — Model Evaluation Report",
        "=" * 60, "",
    ]

    if lstm_metrics:
        lines.append("── LSTM Flood Forecasting ──")
        lines.append(f"  MAE:  {lstm_metrics['mae']:.4f}")
        lines.append(f"  RMSE: {lstm_metrics['rmse']:.4f}")
        lines.append("")

    if bench_df is not None:
        lines.append("── Regression Benchmarking ──")
        lines.append(bench_df.to_string(index=False))
        lines.append("")

    if risk_metrics:
        lines.append("── Risk Classification ──")
        for k, v in risk_metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")

    if not lstm_metrics and bench_df is None and not risk_metrics:
        lines.append("No model results available yet. Run models from the relevant pages first.")

    lines.append("── Recommendations ──")
    lines.extend(_recommendations(lstm_metrics, bench_df, risk_metrics))
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def _recommendations(lstm, bench_df, risk):
    recs = []
    if bench_df is not None and len(bench_df) > 0:
        rmse_col = "RMSE" if "RMSE" in bench_df.columns else "rmse"
        model_col = "Model" if "Model" in bench_df.columns else "model"
        best = bench_df.loc[bench_df[rmse_col].idxmin()]
        recs.append(f"• Best regression model by RMSE: **{best[model_col]}** (RMSE={best[rmse_col]:.4f})")
        if best[rmse_col] > 1.0:
            recs.append("  → RMSE > 1.0 — consider adding more features or tuning hyperparameters.")
        else:
            recs.append("  → Acceptable error range for normalized targets.")

    if lstm:
        if lstm["rmse"] < 0.1:
            recs.append("• LSTM RMSE < 0.1 — strong fit (check for data leakage on real datasets).")
        elif lstm["rmse"] < 0.3:
            recs.append("• LSTM RMSE is moderate — consider longer training or more units.")
        else:
            recs.append("• LSTM RMSE is high — review data quality, sequence length, and features.")

    if risk:
        f1 = risk.get("f1") or risk.get("weighted_f1") or risk.get("macro_f1")
        if f1 and f1 > 0.9:
            recs.append("• Classification F1 > 0.90 — excellent (verify with hold-out data).")
        elif f1 and f1 > 0.7:
            recs.append("• Classification F1 is good but could improve with class balancing.")
        elif f1:
            recs.append("• Classification F1 is below 0.70 — investigate class imbalance or features.")

    if not recs:
        recs.append("• Run the models first to receive recommendations.")
    return recs


def main():
    st.title("📋 Model Evaluation Report")
    st.markdown("Consolidated view of all model performance metrics and actionable recommendations.")
    st.markdown("---")

    # Gather metrics from session state
    lstm_metrics = None
    if "lstm_y_test" in st.session_state and "lstm_y_pred" in st.session_state:
        y_t = st.session_state["lstm_y_test"]
        y_p = st.session_state["lstm_y_pred"].flatten()
        lstm_metrics = {
            "mae": float(np.mean(np.abs(y_t - y_p))),
            "rmse": float(np.sqrt(np.mean((y_t - y_p) ** 2))),
        }

    bench_df = st.session_state.get("bench_results")
    risk_metrics = st.session_state.get("risk_metrics")

    if lstm_metrics is None and bench_df is None and risk_metrics is None:
        st.info(
            "No model results found in this session. Train models from the "
            "**🌊 Flood Forecasting**, **📈 Model Benchmarking**, and "
            "**⚠️ Risk Classification** pages first."
        )
        return

    # ── LSTM Summary ─────────────────────────────────────────────────────
    if lstm_metrics:
        st.subheader("🌊 LSTM Forecasting")
        c1, c2 = st.columns(2)
        with c1:
            kpi_card("MAE", f"{lstm_metrics['mae']:.4f}")
        with c2:
            kpi_card("RMSE", f"{lstm_metrics['rmse']:.4f}")
        st.markdown("")

    # ── Benchmarking Summary ─────────────────────────────────────────────
    if bench_df is not None:
        st.subheader("📈 Regression Benchmarking")
        num_cols = [c for c in ["MAE", "RMSE", "mae", "rmse"] if c in bench_df.columns]
        m_col = "Model" if "Model" in bench_df.columns else "model"
        mae_col = "MAE" if "MAE" in bench_df.columns else "mae"
        rmse_col = "RMSE" if "RMSE" in bench_df.columns else "rmse"

        st.dataframe(
            bench_df.style.highlight_min(subset=num_cols, color="#d4edda") if num_cols else bench_df,
            use_container_width=True,
        )

        # Side-by-side bar
        fig = go.Figure()
        fig.add_trace(go.Bar(name="MAE", x=bench_df[m_col], y=bench_df[mae_col], marker_color="#1f77b4"))
        fig.add_trace(go.Bar(name="RMSE", x=bench_df[m_col], y=bench_df[rmse_col], marker_color="#ff7f0e"))
        fig.update_layout(barmode="group", template="plotly_white", title="Regression Error Comparison")
        st.plotly_chart(fig, use_container_width=True)

    # ── Risk Classification Summary ──────────────────────────────────────
    if risk_metrics:
        st.subheader("⚠️ Risk Classification")
        cols = st.columns(4)
        prec = risk_metrics.get("precision", "N/A")
        rec = risk_metrics.get("recall", "N/A")
        f1 = risk_metrics.get("f1", "N/A")
        pr_auc = risk_metrics.get("pr_auc", {}).get("macro_avg", "N/A") if isinstance(risk_metrics.get("pr_auc"), dict) else risk_metrics.get("pr_auc", "N/A")

        with cols[0]:
            kpi_card("Precision (Macro)", f"{prec:.4f}" if isinstance(prec, float) else prec)
        with cols[1]:
            kpi_card("Recall (Macro)", f"{rec:.4f}" if isinstance(rec, float) else rec)
        with cols[2]:
            kpi_card("F1-Score (Macro)", f"{f1:.4f}" if isinstance(f1, float) else f1)
        with cols[3]:
            kpi_card("PR-AUC (Macro)", f"{pr_auc:.4f}" if isinstance(pr_auc, float) else pr_auc)

    # ── Recommendations ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💡 Recommendations")
    for rec in _recommendations(lstm_metrics, bench_df, risk_metrics):
        st.markdown(rec)

    # ── Download report ──────────────────────────────────────────────────
    st.markdown("---")
    report_text = _build_report_text(lstm_metrics, bench_df, risk_metrics)
    st.download_button(
        "⬇️ Download Full Evaluation Report (TXT)",
        data=report_text.encode("utf-8"),
        file_name="floodguard_evaluation_report.txt",
        mime="text/plain",
    )

    with st.expander("Preview report"):
        st.code(report_text, language="text")


try:
    main()
except Exception as exc:
    st.error(f"An unexpected error occurred: {exc}")
