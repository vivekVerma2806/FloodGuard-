"""FloodGuard AI — Model Evaluation & Report Page."""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import (
    DATA_DIR,
    FEATURES,
    RISK_CATEGORIES,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    TEST_SPLIT_RATIO,
)
from src.data_preprocessing import (
    chronological_split,
    clean_data,
    engineer_features,
    load_data,
)
from src.ml_benchmarking import compare_models
from src.risk_classification import assign_risk_labels, evaluate_classification, train_risk_classifier
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
        if lstm["rmse"] < 0.25:
            recs.append("• LSTM recurrent model exhibits high temporal fidelity (RMSE < 0.25). Verify generalization across monsoon holdout epochs.")
        elif lstm["rmse"] < 0.50:
            recs.append("• LSTM demonstrates acceptable predictive alignment. Deeper hidden units or additional lagged windows may improve flood peak timing.")
        else:
            recs.append("• Elevated LSTM divergence (RMSE ≥ 0.50). Review input normalisation, sequence window depth, and extreme weather sensor dropouts.")

    if risk:
        f1 = risk.get("f1") or risk.get("weighted_f1") or risk.get("macro_f1")
        if f1 and f1 >= 0.80:
            recs.append("• Multi-class classifier shows strong threat tier discrimination (Macro F1 ≥ 0.80). Ready for field telemetry test runs.")
        elif f1 and f1 >= 0.60:
            recs.append("• Solid threat discrimination (Macro F1 ≥ 0.60). Minor boundary confusion between Moderate and High tiers; recommend focal loss or SMOTE re-weighting.")
        elif f1:
            recs.append("• Macro F1 below 0.60. Severe class imbalance or overlap detected in low-frequency Critical flood events.")

    if not recs:
        recs.append("• Initiate model training pipelines in Sections II through IV to compute comprehensive diagnostics.")
    return recs


def _format_rec_html(rec: str) -> str:
    """Format recommendation bullet and sub-arrow with high contrast editorial styling."""
    import re

    rec_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", rec)
    rec_html = re.sub(r"`([^`]+)`", r"<code>\1</code>", rec_html)
    if rec_html.startswith("  → ") or rec_html.startswith("→ "):
        clean_sub = rec_html.replace("  → ", "").replace("→ ", "")
        return (
            f'<div style="padding:0.6rem 1rem 0.6rem 2.2rem;background:#FAF8F5;border-left:2.5px solid #171717;border-bottom:1px solid #E2DFD7;margin-bottom:0.4rem;font-size:13px;color:#333333;line-height:1.45;">'
            f'<span style="font-family:\'IBM Plex Mono\',monospace;font-weight:700;margin-right:8px;color:#171717;">↳</span>{clean_sub}'
            f'</div>'
        )
    clean_bullet = rec_html.replace("• ", "")
    return (
        f'<div style="padding:0.85rem 1.15rem;background:#FFFFFF;border:1px solid #D6D3CC;margin-bottom:0.4rem;font-size:13.5px;color:#171717;line-height:1.5;">'
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-weight:700;margin-right:8px;color:#171717;">•</span>{clean_bullet}'
        f'</div>'
    )


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
            "You can train models in Sections II, III, and IV, or execute an immediate baseline audit on archival telemetry using the button below.",
            label="SESSION TELEMETRY NOTICE",
            level="info",
        )
        if st.button("Generate Baseline Performance Audit from Archival Telemetry"):
            with st.spinner("Executing baseline benchmark and classification pipelines..."):
                try:
                    demo_csv = os.path.join(DATA_DIR, "demo_dataset.csv")
                    if os.path.exists(demo_csv):
                        df_raw = load_data(demo_csv)
                        df_clean = clean_data(df_raw)
                        df_feat = engineer_features(df_clean)
                        if TARGET_CLASSIFICATION not in df_feat.columns:
                            df_feat = assign_risk_labels(df_feat)

                        # 1. Regression Benchmarks on 1st station
                        stn_first = df_feat["station_name"].iloc[0]
                        sdf = df_feat[df_feat["station_name"] == stn_first].sort_values("date").reset_index(drop=True)
                        avail_feats = [f for f in FEATURES if f in sdf.columns and f != TARGET_REGRESSION]
                        sdf_clean = sdf.dropna(subset=avail_feats + [TARGET_REGRESSION])
                        tr_df, ts_df = chronological_split(sdf_clean, TEST_SPLIT_RATIO)

                        b_res, b_mods = compare_models(
                            tr_df[avail_feats].values,
                            tr_df[TARGET_REGRESSION].values,
                            ts_df[avail_feats].values,
                            ts_df[TARGET_REGRESSION].values,
                        )
                        st.session_state["bench_results"] = b_res
                        st.session_state["bench_models"] = b_mods

                        # 2. Classification
                        clf_feats = [f for f in FEATURES if f in df_feat.columns]
                        df_clf = df_feat.dropna(subset=clf_feats + [TARGET_CLASSIFICATION])
                        c_tr, c_ts = chronological_split(df_clf, TEST_SPLIT_RATIO)
                        clf_model = train_risk_classifier(c_tr[clf_feats].values, c_tr[TARGET_CLASSIFICATION].values, n_estimators=100)
                        y_pred_clf = clf_model.predict(c_ts[clf_feats].values)
                        y_prob_clf = clf_model.predict_proba(c_ts[clf_feats].values)
                        r_met = evaluate_classification(c_ts[TARGET_CLASSIFICATION].values, y_pred_clf, y_prob_clf, labels=RISK_CATEGORIES)
                        st.session_state["risk_metrics"] = r_met
                        st.session_state["risk_y_test"] = c_ts[TARGET_CLASSIFICATION].values
                        st.session_state["risk_y_pred"] = y_pred_clf
                        st.session_state["risk_y_prob"] = y_prob_clf

                        # 3. Baseline LSTM sequence evaluation
                        from src.data_preprocessing import normalize_features, create_sequences
                        from src.lstm_model import build_lstm_model, train_lstm_model, predict_lstm

                        seq_features = [f for f in FEATURES if f in tr_df.columns]
                        tr_norm, scaler = normalize_features(tr_df, seq_features)
                        ts_norm = ts_df.copy()
                        ts_norm[seq_features] = scaler.transform(ts_df[seq_features].values)
                        t_idx = seq_features.index(TARGET_REGRESSION)
                        X_sq_tr, y_sq_tr = create_sequences(
                            tr_norm[seq_features].values,
                            tr_norm[seq_features].values[:, t_idx],
                            seq_len=5,
                        )
                        X_sq_ts, y_sq_ts = create_sequences(
                            ts_norm[seq_features].values,
                            ts_norm[seq_features].values[:, t_idx],
                            seq_len=5,
                        )
                        lstm_net = build_lstm_model((5, len(seq_features)), units=32)
                        train_lstm_model(lstm_net, X_sq_tr, y_sq_tr, X_sq_ts, y_sq_ts, epochs=3, batch_size=32)
                        lstm_preds = predict_lstm(lstm_net, X_sq_ts)
                        st.session_state["lstm_y_test"] = y_sq_ts
                        st.session_state["lstm_y_pred"] = lstm_preds
                        st.session_state["lstm_model"] = lstm_net

                        st.rerun()
                except Exception as exc:
                    st.error(f"Failed to generate baseline performance audit: {exc}")
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
        render_html(_format_rec_html(rec))

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
