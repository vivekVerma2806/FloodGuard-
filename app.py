"""FloodGuard AI - Spatiotemporal Risk Intelligence (Overview)."""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, DEMO_STATIONS
from src.ui_theme import (
    apply_theme,
    render_sidebar,
    editorial_header,
    kpi_card,
    editorial_callout,
    render_html,
    style_plotly_chart,
    BORDER_COLOR,
    CARD_BG,
    TEXT_PRIMARY,
    TEXT_MUTED,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FloodGuard AI — Spatiotemporal Risk Intelligence",
    page_icon="☵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global monochrome editorial styling
apply_theme()
render_sidebar()


def main():
    editorial_header(
        title="Hydrometric Basin Observatory",
        kicker="ARCHIVAL TELEMETRY // EXECUTIVE SYNTHESIS",
        subtitle=(
            "Continuous multi-station hydrological observations, antecedent precipitation records, "
            "and spatiotemporal risk indices across five sentinel river basins."
        ),
    )

    # Ingest baseline observations
    demo_path = os.path.join(DATA_DIR, "demo_dataset.csv")
    if not os.path.exists(demo_path):
        editorial_callout(
            "Baseline hydrological dataset is uninitialized. Generate the sample telemetry via "
            "`scripts/generate_demo_data.py` or ingest records via the Data Upload module.",
            label="TELEMETRY STATUS // MISSING",
            level="warning",
        )
        return

    try:
        df = pd.read_csv(demo_path, parse_dates=["date"])
    except Exception as exc:
        st.error(f"Failed to parse hydrological observations: {exc}")
        return

    # -------------------------------------------------------------------------
    # Key Scientific Metrics
    # -------------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Observations", f"{len(df):,}", caption="Daily telemetry samples (2019-2023)")
    with c2:
        kpi_card("Sentinel Basins", f"{df['station_id'].nunique():02d}", caption="Gauging stations monitored")
    with c3:
        kpi_card("Mean Precipitation", f"{df['rainfall_mm'].mean():.1f} mm", caption="Basin-wide daily average")
    with c4:
        kpi_card("Mean River Stage", f"{df['river_level_m'].mean():.2f} m", caption="Surface elevation gauge datum")

    render_html("<div style='height: 18px;'></div>")

    # -------------------------------------------------------------------------
    # Scientific Visualizations
    # -------------------------------------------------------------------------
    tab_rainfall, tab_stage, tab_stations = st.tabs([
        "Precipitation Time-Series",
        "Stage Elevation Distribution",
        "Sentinel Station Ledger",
    ])

    with tab_rainfall:
        daily_rain = (
            df.groupby("date")["rainfall_mm"]
            .mean()
            .reset_index()
            .rename(columns={"rainfall_mm": "avg_rainfall_mm"})
        )
        # 30-day moving average for hydrological trendline
        daily_rain["rolling_30d"] = daily_rain["avg_rainfall_mm"].rolling(window=30, min_periods=1).mean()

        fig_rain = go.Figure()
        fig_rain.add_trace(
            go.Scatter(
                x=daily_rain["date"],
                y=daily_rain["avg_rainfall_mm"],
                mode="lines",
                name="Daily Mean Precipitation",
                line=dict(color="#B5B2A9", width=1),
                opacity=0.65,
            )
        )
        fig_rain.add_trace(
            go.Scatter(
                x=daily_rain["date"],
                y=daily_rain["rolling_30d"],
                mode="lines",
                name="30-Day Moving Mean",
                line=dict(color="#171717", width=2),
            )
        )
        fig_rain.update_layout(
            title="Basin-Wide Daily Precipitation Time-Series (2019 – 2023)",
            xaxis_title="Calendar Date",
            yaxis_title="Precipitation (mm)",
        )
        fig_rain = style_plotly_chart(fig_rain, height=400)
        st.plotly_chart(fig_rain, use_container_width=True)

    with tab_stage:
        # Monochrome shades for the stations
        station_colors = ["#171717", "#4B5563", "#78716C", "#A8A29E", "#D6D3CC"]
        stations = sorted(df["station_name"].unique())

        fig_level = go.Figure()
        for idx, station in enumerate(stations):
            stn_data = df[df["station_name"] == station]["river_level_m"].dropna()
            fig_level.add_trace(
                go.Histogram(
                    x=stn_data,
                    name=station.split(" - ")[0],
                    nbinsx=40,
                    opacity=0.65,
                    marker=dict(color=station_colors[idx % len(station_colors)], line=dict(color="#171717", width=0.5)),
                )
            )

        fig_level.update_layout(
            barmode="overlay",
            title="River Gauge Stage Distribution by Sentinel Basin",
            xaxis_title="River Stage Elevation (m)",
            yaxis_title="Observation Frequency",
        )
        fig_level = style_plotly_chart(fig_level, height=400)
        st.plotly_chart(fig_level, use_container_width=True)

    with tab_stations:
        render_html(
            f"""
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.1em; color: {TEXT_MUTED}; margin-bottom: 0.75rem;">
                // SENTINEL MONITORING NETWORK METADATA & RECENT OBSERVATIONS
            </div>
            """
        )
        latest_summary = []
        for stn in DEMO_STATIONS:
            stn_df = df[df["station_id"] == stn["id"]].sort_values("date")
            latest_row = stn_df.iloc[-1] if len(stn_df) > 0 else {}
            latest_summary.append({
                "Station ID": stn["id"],
                "Basin Name": stn["name"],
                "Coordinates": f"{stn['lat']:.4f}° N, {stn['lon']:.4f}° E",
                "Mean Level (m)": f"{stn_df['river_level_m'].mean():.2f}" if len(stn_df) > 0 else "N/A",
                "Max Recorded (m)": f"{stn_df['river_level_m'].max():.2f}" if len(stn_df) > 0 else "N/A",
                "Latest Observation": str(latest_row.get("date", "N/A"))[:10],
                "Current Stage": f"{latest_row.get('river_level_m', 0.0):.2f} m",
            })
        st.dataframe(pd.DataFrame(latest_summary), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # Technical Abstract
    # -------------------------------------------------------------------------
    editorial_callout(
        "FloodGuard AI operates as an analytical bridge between physical hydrology and deep learning. "
        "Observations undergo strict temporal validation to prevent predictive lookahead bias, feeding "
        "multivariate LSTM networks and ensemble regressors to establish calibrated hydrological lead-time.",
        label="RESEARCH BULLETIN // METHODOLOGY & SCOPE",
        level="info",
    )


try:
    main()
except Exception as exc:
    st.error(f"Runtime Exception in Overview module: {exc}")
