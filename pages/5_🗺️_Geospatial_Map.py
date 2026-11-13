"""FloodGuard AI — Geospatial Risk Map Page."""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import DATA_DIR, TARGET_CLASSIFICATION, RISK_CATEGORIES
from src.data_preprocessing import load_data, clean_data, engineer_features
from src.risk_classification import assign_risk_labels
from src.geospatial import (
    get_risk_color, csv_to_geodataframe, load_geojson, create_risk_map,
)

st.set_page_config(page_title="Geospatial Map | FloodGuard AI", page_icon="🗺️", layout="wide")


def main():
    st.title("🗺️ Geospatial Risk Map")
    st.markdown("Visualize flood-risk levels across monitoring stations on an interactive map.")

    st.warning(
        "📌 **Demo locations** — these are synthetic monitoring stations and do **not** "
        "represent verified real-world installations.",
        icon="⚠️",
    )
    st.markdown("---")

    # ── Data source selector ─────────────────────────────────────────────
    source = st.radio(
        "Map data source:",
        ["Demo GeoJSON", "From CSV (latest data per station)", "Upload custom GeoJSON"],
        horizontal=True,
    )

    gdf = None

    if source == "Demo GeoJSON":
        geojson_path = os.path.join(DATA_DIR, "demo_geospatial.geojson")
        if not os.path.exists(geojson_path):
            st.warning("Demo GeoJSON not found.")
            return
        try:
            gdf = load_geojson(geojson_path)
            st.success(f"✅ Loaded GeoJSON with {len(gdf)} features.")
        except Exception as e:
            st.error(f"Error loading GeoJSON: {e}")
            return

        # Try to add risk info from CSV
        risk_col = "risk_zone" if "risk_zone" in gdf.columns else None
        level_col = "river_level_m" if "river_level_m" in gdf.columns else None
        rain_col = "rainfall_mm" if "rainfall_mm" in gdf.columns else None
        name_col = "station_name" if "station_name" in gdf.columns else None

    elif source == "From CSV (latest data per station)":
        df = None
        if "processed_df" in st.session_state:
            df = st.session_state["processed_df"]
        else:
            demo_csv = os.path.join(DATA_DIR, "demo_dataset.csv")
            if os.path.exists(demo_csv):
                df = load_data(demo_csv)
                df = clean_data(df)
                df = engineer_features(df)

        if df is None:
            st.warning("No data available. Load data from the **📊 Data Upload** page.")
            return

        # Ensure risk labels
        if TARGET_CLASSIFICATION not in df.columns:
            df = assign_risk_labels(df)

        # Latest record per station
        latest = df.sort_values("date").groupby("station_id").tail(1).reset_index(drop=True)

        if "latitude" not in latest.columns or "longitude" not in latest.columns:
            st.error("Data must contain 'latitude' and 'longitude' columns.")
            return

        try:
            gdf = csv_to_geodataframe(latest, lat_col="latitude", lon_col="longitude")
            st.success(f"✅ Created GeoDataFrame with {len(gdf)} stations.")
        except Exception as e:
            st.error(f"Error creating GeoDataFrame: {e}")
            return

        risk_col = TARGET_CLASSIFICATION
        level_col = "river_level_m"
        rain_col = "rainfall_mm"
        name_col = "station_name"

    else:  # Upload custom GeoJSON
        uploaded = st.file_uploader("Upload a GeoJSON file", type=["geojson", "json"])
        if uploaded is None:
            st.info("👆 Upload a GeoJSON file to visualize.")
            return
        try:
            import tempfile
            tmp = os.path.join(DATA_DIR, "_uploaded.geojson")
            with open(tmp, "wb") as f:
                f.write(uploaded.read())
            gdf = load_geojson(tmp)
            st.success(f"✅ Loaded GeoJSON with {len(gdf)} features.")
        except Exception as e:
            st.error(f"Error loading GeoJSON: {e}")
            return

        risk_col = "risk_zone" if "risk_zone" in gdf.columns else None
        level_col = "river_level_m" if "river_level_m" in gdf.columns else None
        rain_col = "rainfall_mm" if "rainfall_mm" in gdf.columns else None
        name_col = "station_name" if "station_name" in gdf.columns else None

    if gdf is None or len(gdf) == 0:
        st.warning("No geospatial data to display.")
        return

    # ── Render map ───────────────────────────────────────────────────────
    st.subheader("🗺️ Interactive Map")
    try:
        # Ensure fallback columns if missing
        if "flood_risk" not in gdf.columns and "risk_zone" in gdf.columns:
            gdf["flood_risk"] = gdf["risk_zone"]
        elif "flood_risk" not in gdf.columns:
            gdf["flood_risk"] = "Moderate"

        if "river_level_m" not in gdf.columns:
            gdf["river_level_m"] = 0.0
        if "rainfall_mm" not in gdf.columns:
            gdf["rainfall_mm"] = 0.0
        if "station_name" not in gdf.columns:
            gdf["station_name"] = [f"Station {i+1}" for i in range(len(gdf))]

        m = create_risk_map(
            gdf,
            risk_col="flood_risk",
            level_col="river_level_m",
            rainfall_col="rainfall_mm",
            name_col="station_name",
        )
        components.html(m._repr_html_(), height=600, scrolling=True)
    except Exception as e:
        st.error(f"Map rendering failed: {e}")

    # ── Legend ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Legend")
    legend_cols = st.columns(len(RISK_CATEGORIES))
    for col, cat in zip(legend_cols, RISK_CATEGORIES):
        color = get_risk_color(cat)
        col.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<div style="width:20px;height:20px;border-radius:50%;background:{color};"></div>'
            f'<span>{cat}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Station details ──────────────────────────────────────────────────
    if name_col and name_col in gdf.columns:
        with st.expander("Station details table"):
            display_cols = [c for c in [name_col, risk_col, level_col, rain_col] if c and c in gdf.columns]
            st.dataframe(gdf[display_cols] if display_cols else gdf, use_container_width=True)


try:
    main()
except Exception as exc:
    st.error(f"An unexpected error occurred: {exc}")
