"""FloodGuard AI — Data Upload & Preprocessing Page."""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import DATA_DIR
from src.data_preprocessing import load_data, clean_data, engineer_features

st.set_page_config(page_title="Data Upload | FloodGuard AI", page_icon="📊", layout="wide")


def kpi_card(title: str, value, delta=None):
    delta_html = ""
    if delta is not None:
        color = "#28a745" if str(delta).startswith("-") or str(delta).startswith("+") else "#666"
        if str(delta).startswith("-"):
            color = "#28a745"
        delta_html = f'<p style="margin:0; color:{color}; font-size:0.9rem;">{delta}</p>'
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


def main():
    st.title("📊 Data Upload & Preprocessing")
    st.markdown("Upload a CSV file or load the built-in demo dataset, then clean and engineer features.")
    st.markdown("---")

    # ── Data Source ───────────────────────────────────────────────────────
    source = st.radio("Select data source:", ["Upload CSV", "Use demo dataset"], horizontal=True)
    df = None

    if source == "Upload CSV":
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded, parse_dates=["date"])
                st.success(f"✅ Uploaded **{uploaded.name}** — {len(df):,} rows, {df.shape[1]} columns.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
    else:
        demo_path = os.path.join(DATA_DIR, "demo_dataset.csv")
        if not os.path.exists(demo_path):
            st.warning("Demo dataset not found. Please generate it first (`scripts/generate_demo_data.py`).")
            return
        try:
            df = load_data(demo_path)
            st.success(f"✅ Demo dataset loaded — {len(df):,} rows, {df.shape[1]} columns.")
        except Exception as e:
            st.error(f"Error loading demo data: {e}")
            return

    if df is None:
        st.info("👆 Upload a CSV or select the demo dataset to get started.")
        return

    # ── Raw Data Preview ─────────────────────────────────────────────────
    st.subheader("🔍 Raw Data Preview")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Rows", f"{len(df):,}")
    with c2:
        st.metric("Columns", df.shape[1])

    with st.expander("Show raw data", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    with st.expander("Column data types"):
        st.dataframe(
            pd.DataFrame({"Column": df.columns, "Dtype": df.dtypes.astype(str).values}),
            use_container_width=True,
        )

    # ── Cleaning ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧹 Data Cleaning")
    if st.button("🧹 Clean Data", type="primary"):
        with st.spinner("Cleaning data..."):
            try:
                missing_before = int(df.isnull().sum().sum())
                dupes_before = int(df.duplicated().sum())

                df_clean = clean_data(df.copy())

                missing_after = int(df_clean.isnull().sum().sum())
                dupes_after = int(df_clean.duplicated().sum())

                st.session_state["clean_df"] = df_clean
                st.success("✅ Data cleaned!")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    kpi_card("Missing (Before)", missing_before)
                with col2:
                    kpi_card("Missing (After)", missing_after, delta=f"{missing_after - missing_before:+d}")
                with col3:
                    kpi_card("Duplicates (Before)", dupes_before)
                with col4:
                    kpi_card("Duplicates (After)", dupes_after, delta=f"{dupes_after - dupes_before:+d}")

                with st.expander("Preview cleaned data"):
                    st.dataframe(df_clean.head(50), use_container_width=True)
            except Exception as e:
                st.error(f"Cleaning failed: {e}")

    # ── Feature Engineering ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚙️ Feature Engineering")
    clean_df = st.session_state.get("clean_df")
    if clean_df is None:
        st.info("Please clean the data first.")
    else:
        if st.button("⚙️ Engineer Features", type="primary"):
            with st.spinner("Engineering features..."):
                try:
                    df_feat = engineer_features(clean_df.copy())
                    st.session_state["processed_df"] = df_feat
                    new_cols = [c for c in df_feat.columns if c not in clean_df.columns]
                    st.success(f"✅ Features added: {', '.join(new_cols)}")

                    with st.expander("Preview engineered data"):
                        st.dataframe(df_feat.head(50), use_container_width=True)
                except Exception as e:
                    st.error(f"Feature engineering failed: {e}")

    # ── Download ─────────────────────────────────────────────────────────
    st.markdown("---")
    processed = st.session_state.get("processed_df", st.session_state.get("clean_df"))
    if processed is not None:
        csv_bytes = processed.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Processed Data (CSV)",
            data=csv_bytes,
            file_name="floodguard_processed.csv",
            mime="text/csv",
        )


try:
    main()
except Exception as exc:
    st.error(f"An unexpected error occurred: {exc}")
