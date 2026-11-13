"""FloodGuard AI - Telemetry Ingestion & Quality Control."""

import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import DATA_DIR
from src.data_preprocessing import load_data, clean_data, engineer_features
from src.ui_theme import (
    apply_theme,
    render_sidebar,
    editorial_header,
    kpi_card,
    editorial_callout,
    render_html,
    TEXT_MUTED,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Ingestion | FloodGuard AI",
    page_icon="☵",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
render_sidebar()


def main():
    editorial_header(
        title="Telemetry Ingestion & Quality Control",
        kicker="DATA PIPELINE // SYSTEM GATEWAY",
        subtitle=(
            "Ingest raw multi-station hydrological CSV time-series, execute automated "
            "missing-value imputation, deduplicate timestamps, and derive antecedent precipitation features."
        ),
    )

    # -------------------------------------------------------------------------
    # Ingestion Source Selector
    # -------------------------------------------------------------------------
    render_html(
        f"""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; color: {TEXT_MUTED}; margin-bottom: 0.5rem;">
            // TELEMETRY SOURCE SELECTION
        </div>
        """
    )
    source = st.radio(
        "Telemetry source:",
        ["Baseline Archival Telemetry (Demo)", "Upload Custom Hydrological CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )
    df = None

    if source == "Upload Custom Hydrological CSV":
        sample_csv_path = os.path.join(DATA_DIR, "sample_custom_telemetry.csv")
        col_c_load, col_c_down = st.columns([1, 1])
        with col_c_load:
            btn_load_sample_csv = st.button("Load Included Sample Custom CSV (150 Records)")
        with col_c_down:
            if os.path.exists(sample_csv_path):
                with open(sample_csv_path, "rb") as f_samp_c:
                    st.download_button(
                        "Download Sample Custom CSV File",
                        data=f_samp_c.read(),
                        file_name="sample_custom_telemetry.csv",
                        mime="text/csv",
                    )

        uploaded = st.file_uploader("Upload station CSV", type=["csv"], label_visibility="collapsed")

        if btn_load_sample_csv:
            try:
                df = pd.read_csv(sample_csv_path, parse_dates=["date"])
                editorial_callout(
                    f"Successfully loaded included sample custom telemetry: {len(df):,} records across {df['station_id'].nunique()} basins.",
                    label="SAMPLE INGESTION // SUCCESS",
                    level="info",
                )
            except Exception as exc:
                st.error(f"Error reading sample CSV: {exc}")
                return
        elif uploaded is not None:
            try:
                df = pd.read_csv(uploaded, parse_dates=["date"])
                editorial_callout(
                    f"Successfully parsed external file: {uploaded.name} ({len(df):,} records, {df.shape[1]} columns).",
                    label="INGESTION // SUCCESS",
                    level="info",
                )
            except Exception as exc:
                st.error(f"Error reading uploaded file: {exc}")
                return
    else:
        demo_path = os.path.join(DATA_DIR, "demo_dataset.csv")
        if not os.path.exists(demo_path):
            editorial_callout(
                "Baseline dataset is uninitialized. Generate sample records via scripts/generate_demo_data.py.",
                label="SYSTEM NOTICE // MISSING FILE",
                level="warning",
            )
            return
        try:
            df = load_data(demo_path)
            render_html(
                f"""
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: #2E6930; margin-bottom: 1rem;">
                    [OK] Archival telemetry loaded: {len(df):,} daily records across 5 river basins.
                </div>
                """
            )
        except Exception as exc:
            st.error(f"Error loading demo dataset: {exc}")
            return

    if df is None:
        editorial_callout(
            "Select the baseline archival dataset or upload a valid hydrological CSV to begin data validation.",
            label="PIPELINE AWAITING INPUT",
            level="info",
        )
        return

    # -------------------------------------------------------------------------
    # Raw Data Ledger & Schema Audit
    # -------------------------------------------------------------------------
    render_html(
        f"""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; color: {TEXT_MUTED}; margin-top: 1rem; margin-bottom: 0.5rem;">
            // RAW TELEMETRY SCHEMA & SUMMARY AUDIT
        </div>
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Rows", f"{len(df):,}")
    with c2:
        kpi_card("Total Attributes", f"{df.shape[1]:02d}")
    with c3:
        kpi_card("Missing Cells", f"{int(df.isnull().sum().sum()):,}", caption="Unpopulated readings")
    with c4:
        kpi_card("Duplicate Rows", f"{int(df.duplicated().sum()):,}", caption="Repeated timestamp entries")

    with st.expander("Inspect Raw Observations Ledger (Top 50 Records)", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)

    with st.expander("Attribute Data Types & Non-Null Distribution", expanded=False):
        schema_df = pd.DataFrame({
            "Attribute": df.columns,
            "Storage Dtype": df.dtypes.astype(str).values,
            "Non-Null Count": df.notnull().sum().values,
            "Missing Count": df.isnull().sum().values,
        })
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Data Cleaning Pipeline
    # -------------------------------------------------------------------------
    render_html(
        f"""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; color: {TEXT_MUTED}; margin-bottom: 0.5rem;">
            // STEP 01: QUALITY AUDIT & RECTIFICATION
        </div>
        """
    )
    col_btn_clean, col_info_clean = st.columns([1, 3])
    with col_btn_clean:
        btn_clean = st.button("Execute Quality Cleanse")
    with col_info_clean:
        st.caption("Removes duplicate timestamps and performs group-based forward & backward filling per basin.")

    if btn_clean:
        with st.spinner("Processing telemetry quality cleanse..."):
            try:
                missing_before = int(df.isnull().sum().sum())
                dupes_before = int(df.duplicated().sum())

                df_clean = clean_data(df.copy())

                missing_after = int(df_clean.isnull().sum().sum())
                dupes_after = int(df_clean.duplicated().sum())

                st.session_state["clean_df"] = df_clean

                render_html("<div style='height: 10px;'></div>")
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    kpi_card("Pre-Clean Missing", missing_before)
                with r2:
                    kpi_card("Post-Clean Missing", missing_after, delta=f"{missing_after - missing_before:+d}")
                with r3:
                    kpi_card("Pre-Clean Dupes", dupes_before)
                with r4:
                    kpi_card("Post-Clean Dupes", dupes_after, delta=f"{dupes_after - dupes_before:+d}")

                editorial_callout(
                    f"Cleanse complete. {len(df_clean):,} validated observations retained. Zero missing values.",
                    label="AUDIT PASSED",
                    level="info",
                )
            except Exception as exc:
                st.error(f"Quality cleanse failed: {exc}")

    # -------------------------------------------------------------------------
    # Feature Engineering Pipeline
    # -------------------------------------------------------------------------
    st.markdown("---")
    render_html(
        f"""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; color: {TEXT_MUTED}; margin-bottom: 0.5rem;">
            // STEP 02: HYDROLOGICAL FEATURE SYNTHESIS
        </div>
        """
    )
    clean_target = st.session_state.get("clean_df")

    if clean_target is None:
        st.caption("Clean the telemetry data above prior to deriving lagged hydrological indicators.")
    else:
        col_btn_feat, col_info_feat = st.columns([1, 3])
        with col_btn_feat:
            btn_feat = st.button("Synthesize Features")
        with col_info_feat:
            st.caption("Derives prev_day_rainfall, 3-day rolling precipitation mean, river delta (level_change), and categorical risk tier.")

        if btn_feat:
            with st.spinner("Calculating moving windows and derivatives..."):
                try:
                    df_feat = engineer_features(clean_target.copy())
                    st.session_state["processed_df"] = df_feat
                    derived_cols = [c for c in df_feat.columns if c not in clean_target.columns]

                    editorial_callout(
                        f"Features synthesized successfully: {', '.join(derived_cols)}.",
                        label="SYNTHESIS COMPLETE",
                        level="info",
                    )
                    with st.expander("Inspect Synthesized Features Matrix", expanded=True):
                        st.dataframe(df_feat.head(40), use_container_width=True)
                except Exception as exc:
                    st.error(f"Feature engineering failed: {exc}")

    # -------------------------------------------------------------------------
    # Export Cleansed Dataset
    # -------------------------------------------------------------------------
    processed_target = st.session_state.get("processed_df", st.session_state.get("clean_df"))
    if processed_target is not None:
        st.markdown("---")
        csv_bytes = processed_target.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Calibrated Dataset (.CSV)",
            data=csv_bytes,
            file_name="floodguard_calibrated_telemetry.csv",
            mime="text/csv",
        )


try:
    main()
except Exception as exc:
    st.error(f"Runtime Exception in Data Ingestion module: {exc}")
