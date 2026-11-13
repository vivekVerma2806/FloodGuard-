"""FloodGuard AI — Overview Dashboard (Main Entry Point)."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, DEMO_STATIONS

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloodGuard AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helper ───────────────────────────────────────────────────────────────────
def kpi_card(title: str, value, delta=None):
    delta_html = ""
    if delta is not None:
        color = "#28a745" if float(delta) >= 0 else "#dc3545"
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


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.title("🌊 FloodGuard AI")
    st.markdown(
        "**AI-powered flood monitoring, forecasting & risk assessment dashboard.**  \n"
        "Upload your own data or explore the built-in demo dataset covering five "
        "Indian river-basin monitoring stations (2019-2023)."
    )
    st.markdown("---")

    # Load demo dataset
    demo_path = os.path.join(DATA_DIR, "demo_dataset.csv")
    if not os.path.exists(demo_path):
        st.warning(
            "⚠️ Demo dataset not found. Please run "
            "`scripts/generate_demo_data.py` first, or upload data via the "
            "**📊 Data Upload** page."
        )
        return

    try:
        df = pd.read_csv(demo_path, parse_dates=["date"])
    except Exception as e:
        st.error(f"Error loading demo dataset: {e}")
        return

    # ── KPI Cards ────────────────────────────────────────────────────────
    st.subheader("📌 Key Statistics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Records", f"{len(df):,}")
    with c2:
        kpi_card("Monitoring Stations", df["station_id"].nunique())
    with c3:
        kpi_card("Avg Rainfall (mm)", f"{df['rainfall_mm'].mean():.1f}")
    with c4:
        kpi_card("Avg River Level (m)", f"{df['river_level_m'].mean():.2f}")

    st.markdown("")  # spacer

    # ── Overview Charts ──────────────────────────────────────────────────
    st.subheader("📈 Overview Charts")
    tab1, tab2 = st.tabs(["Rainfall Trend", "River Level Distribution"])

    with tab1:
        # Aggregate daily rainfall across all stations
        daily_rain = (
            df.groupby("date")["rainfall_mm"]
            .mean()
            .reset_index()
            .rename(columns={"rainfall_mm": "avg_rainfall_mm"})
        )
        fig_rain = px.line(
            daily_rain,
            x="date",
            y="avg_rainfall_mm",
            title="Average Daily Rainfall Across Stations",
            labels={"date": "Date", "avg_rainfall_mm": "Rainfall (mm)"},
            template="plotly_white",
        )
        fig_rain.update_traces(line_color="#1f77b4")
        st.plotly_chart(fig_rain, use_container_width=True)

    with tab2:
        fig_level = px.histogram(
            df,
            x="river_level_m",
            color="station_name",
            nbins=50,
            title="River Level Distribution by Station",
            labels={"river_level_m": "River Level (m)", "station_name": "Station"},
            template="plotly_white",
            barmode="overlay",
            opacity=0.65,
        )
        st.plotly_chart(fig_level, use_container_width=True)

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/flood.png",
            width=64,
        )
        st.markdown("### FloodGuard AI")
        st.markdown(
            "An end-to-end ML pipeline for flood forecasting, risk "
            "classification, and geospatial analysis."
        )
        st.markdown("---")
        st.markdown("**Navigation**")
        st.markdown(
            "Use the sidebar pages to:\n"
            "1. 📊 Upload & preprocess data\n"
            "2. 🌊 LSTM flood forecasting\n"
            "3. 📈 Benchmark ML models\n"
            "4. ⚠️ Risk classification\n"
            "5. 🗺️ Geospatial risk map\n"
            "6. 📋 Model evaluation report"
        )
        st.markdown("---")
        st.caption("Built with Streamlit · Plotly · TensorFlow · scikit-learn")


try:
    main()
except Exception as exc:
    st.error(f"An unexpected error occurred: {exc}")
