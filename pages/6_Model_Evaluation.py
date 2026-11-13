"""FloodGuard AI — Model Evaluation & Report Page."""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.ui_theme import (
    apply_theme,
    render_sidebar,
    editorial_header,
    editorial_callout,
    kpi_card,
    render_html,
    style_plotly_chart,
)

st.set_page_config(
    page_title="Model Evaluation | FloodGuard AI",
    page_icon="⌖",
    layout="wide",
)

apply_theme()
render_sidebar()


def _build_report_text(lstm_metrics, bench_df, risk_metrics):
    """Build a plain-text scientific evaluation report."""
    lines = [
        "================================================================================",
        " FLOODGUARD AI — HYDROLOGIC MODEL AUDIT & SYNTHESIS REPORT",
        " ARCHIVAL RESEARCH & PERFORMANCE LEDGER",
        "================================================================================",
        "",
    ]

    if lstm_metrics:
        lines.append("--- [PART I] DEEP SEQUENCE FORECASTING (MULTIVARIATE LSTM) ---")
        lines.append(f"  Mean Absolute Error (MAE):     {lstm_metrics['mae']:.4f}")
        lines.append(f"  Root Mean Squared Error (RMSE): {lstm_metrics['rmse']:.4f}")
        lines.append("")

    if bench_df is not None:
        lines.append("--- [PART II] REGRESSION BENCHMARKING (STAGE HEIGHT FORECASTING) ---")
        lines.append(bench_df.to_string(index=False))
        lines.append("")

    if risk_metrics:
        lines.append("--- [PART III] OPERATIONAL THREAT TIER CLASSIFICATION ---")
        for k, v in risk_metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:<28}: {v:.4f}")
            elif isinstance(v, dict):
                lines.append(f"  {k}:")
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, float):
                        lines.append(f"    - {sub_k:<24}: {sub_v:.4f}")
                    else:
                        lines.append(f"    - {sub_k:<24}: {sub_v}")
            else:
                lines.append(f"  {k:<28}: {v}")
        lines.append("")

    if not lstm_metrics and bench_df is None and not risk_metrics:
        lines.append("NOTICE: No trained artifacts found in session memory.")
        lines.append("Execute models from Sections II, III, or IV to populate performance benchmarks.")
        lines.append("")

    lines.append("--- [PART IV] SCIENTIFIC DEPLOYMENT ADVISORIES ---")
    for rec in _recommendations(lstm_metrics, bench_df, risk_metrics):
        clean_rec = rec.replace("**", "").replace("• ", "  [+] ").replace("  → ", "      > ")
        lines.append(clean_rec)
    lines.append("")
    lines.append("================================================================================")
    lines.append(" END OF EVALUATION DOSSIER")
    lines.append("================================================================================")
    return "\n".join(lines)


def _recommendations(lstm, bench_df, risk):
    recs = []
    if bench_df is not None and len(bench_df) > 0:
        rmse_col = "RMSE" if "RMSE" in bench_df.columns else "rmse"
        model_col = "Model" if "Model" in bench_df.columns else "model"
        best = bench_df.loc[bench_df[rmse_col].idxmin()]
        recs.append(f"• Optimal baseline regression algorithm: **{best[model_col]}** (RMSE = {best[rmse_col]:.4f}).")
        if best[rmse_col] > 1.0:
            recs.append("  → Residual error exceeds 1.0 metre threshold — recommend spatial covariate expansion.")
        else:
            recs.append("  → Stage error remains well within safe hydrological operational limits.")

    if lstm:
        if lstm["rmse"] < 0.15:
            recs.append("• LSTM recurrent model exhibits high temporal fidelity (RMSE < 0.15). Verify generalization across monsoon holdout epochs.")
        elif lstm["rmse"] < 0.35:
            recs.append("• LSTM demonstrates acceptable predictive alignment. Deeper hidden units or additional lagged windows may improve flood peak timing.")
        else:
            recs.append("• Elevated LSTM divergence (RMSE ≥ 0.35). Review input normalisation, sequence window depth, and extreme weather sensor dropouts.")

    if risk:
        f1 = risk.get("f1") or risk.get("weighted_f1") or risk.get("macro_f1")
        if f1 and f1 > 0.88:
            recs.append("• Multi-class classifier shows strong threat tier discrimination (Macro F1 > 0.88). Ready for field telemetry test runs.")
        elif f1 and f1 > 0.70:
            recs.append("• Solid threat discrimination. Minor boundary confusion between Moderate and High tiers; recommend focal loss or SMOTE re-weighting.")
        elif f1:
            recs.append("• Macro F1 below 0.70. Severe class imbalance or overlap detected in low-frequency Critical flood events.")

    if not recs:
        recs.append("• Initiate model training pipelines in Sections II through IV to compute comprehensive diagnostics.")
    return recs


def main():
    editorial_header(
        section="SECTION VI — SYNTHESIS & AUDIT",
        title="Model Evaluation Ledger",
        subtitle="Consolidated performance audit, comparative benchmark diagnostics, and field deployment advisories.",
    )

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
        editorial_callout(
            "No active model artifacts registered in the current session memory. "
            "Navigate to **Section II (Forecasting)**, **Section III (Benchmarking)**, or **Section IV (Classification)** "
            "to execute models and generate comparative telemetry.",
            level="info",
        )
        return

    # ── LSTM Summary ─────────────────────────────────────────────────────
    if lstm_metrics:
        render_html(
            """
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.12em;
                        text-transform:uppercase;color:#666666;margin:1.4rem 0 0.5rem 0;">
                I. MULTIVARIATE RECURRENT LSTM TELEMETRY
            </div>
            """
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("FORECAST MAE", f"{lstm_metrics['mae']:.4f}", "Absolute error (normalized)")
        with c2:
            kpi_card("FORECAST RMSE", f"{lstm_metrics['rmse']:.4f}", "Root mean square error")
        with c3:
            ratio = (lstm_metrics['rmse'] / (lstm_metrics['mae'] + 1e-6))
            kpi_card("RMSE / MAE RATIO", f"{ratio:.2f}", "Error distribution dispersion")

    # ── Benchmarking Summary ─────────────────────────────────────────────
    if bench_df is not None:
        render_html(
            """
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.12em;
                        text-transform:uppercase;color:#666666;margin:1.8rem 0 0.5rem 0;">
                II. REGRESSION BENCHMARKING AUDIT
            </div>
            """
        )

        m_col = "Model" if "Model" in bench_df.columns else "model"
        mae_col = "MAE" if "MAE" in bench_df.columns else "mae"
        rmse_col = "RMSE" if "RMSE" in bench_df.columns else "rmse"

        st.dataframe(bench_df, use_container_width=True, hide_index=True)

        # Plotly comparison bar
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="MAE (Mean Absolute Error)",
            x=bench_df[m_col],
            y=bench_df[mae_col],
            marker_color="#171717",
        ))
        fig.add_trace(go.Bar(
            name="RMSE (Root Mean Square Error)",
            x=bench_df[m_col],
            y=bench_df[rmse_col],
            marker_color="#737373",
        ))
        fig.update_layout(
            barmode="group",
            height=340,
            title=dict(text="Comparative Model Residual Error", font=dict(family="Playfair Display", size=16, color="#171717")),
            yaxis_title="Stage Error (Metres)",
            xaxis_title="Candidate Model Architecture",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        style_plotly_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Risk Classification Summary ──────────────────────────────────────
    if risk_metrics:
        render_html(
            """
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.12em;
                        text-transform:uppercase;color:#666666;margin:1.8rem 0 0.5rem 0;">
                III. OPERATIONAL THREAT TIER CLASSIFICATION
            </div>
            """
        )
        cols = st.columns(4)
        prec = risk_metrics.get("precision", "N/A")
        rec = risk_metrics.get("recall", "N/A")
        f1 = risk_metrics.get("f1", "N/A")
        pr_auc = risk_metrics.get("pr_auc", {}).get("macro_avg", "N/A") if isinstance(risk_metrics.get("pr_auc"), dict) else risk_metrics.get("pr_auc", "N/A")

        with cols[0]:
            kpi_card("PRECISION (MACRO)", f"{prec:.4f}" if isinstance(prec, float) else str(prec), "Positive predictive value")
        with cols[1]:
            kpi_card("RECALL (MACRO)", f"{rec:.4f}" if isinstance(rec, float) else str(rec), "Sensitivity across tiers")
        with cols[2]:
            kpi_card("F1-SCORE (MACRO)", f"{f1:.4f}" if isinstance(f1, float) else str(f1), "Harmonic mean metric")
        with cols[3]:
            kpi_card("PR-AUC (MACRO)", f"{pr_auc:.4f}" if isinstance(pr_auc, float) else str(pr_auc), "Area under PR curve")

    # ── Recommendations ──────────────────────────────────────────────────
    render_html(
        """
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.12em;
                    text-transform:uppercase;color:#666666;margin:2rem 0 0.5rem 0;">
            IV. FIELD ADVISORIES & SYNTHESIS RECOMMENDATIONS
        </div>
        """
    )
    for rec in _recommendations(lstm_metrics, bench_df, risk_metrics):
        render_html(
            f"""
            <div style="padding:0.75rem 1rem;background:#FFFFFF;border:1px solid #D6D3CC;margin-bottom:0.5rem;font-size:13.5px;color:#171717;">
                {rec}
            </div>
            """
        )

    # ── Download report ──────────────────────────────────────────────────
    render_html(
        """
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.12em;
                    text-transform:uppercase;color:#666666;margin:2rem 0 0.5rem 0;">
            V. REPORT SERIALIZATION & EXPORT
        </div>
        """
    )

    report_text = _build_report_text(lstm_metrics, bench_df, risk_metrics)
    st.download_button(
        "Download Field Evaluation Dossier (ASCII Plaintext)",
        data=report_text.encode("utf-8"),
        file_name="floodguard_evaluation_report.txt",
        mime="text/plain",
    )

    with st.expander("INSPECT SERIALIZED DOSSIER (ASCII)", expanded=False):
        st.code(report_text, language="text")


try:
    main()
except Exception as exc:
    st.error(f"Evaluation module exception: {exc}")
