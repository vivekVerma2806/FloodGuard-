"""FloodGuard AI — Geospatial Risk Map Page."""

import os
import sys
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import DATA_DIR, TARGET_CLASSIFICATION, RISK_CATEGORIES
from src.ui_theme import (
    apply_theme,
    render_sidebar,
    editorial_header,
    editorial_callout,
    kpi_card,
    render_html,
    RISK_COLORS,
)
from src.data_preprocessing import load_data, clean_data, engineer_features
from src.risk_classification import assign_risk_labels
from src.geospatial import (
    get_risk_color,
    csv_to_geodataframe,
    load_geojson,
    create_risk_map,
)

st.set_page_config(
    page_title="Geospatial Atlas | FloodGuard AI",
    page_icon="⌖",
    layout="wide",
)

apply_theme()
render_sidebar()


def main():
    editorial_header(
        section="SECTION V — CARTOGRAPHY & SPATIAL RISK",
        title="Geospatial Risk Atlas",
        subtitle="Spatiotemporal surveillance map and hydrographic threat tier classification across telemetry stations.",
    )

    editorial_callout(
        "**Cartographic Notice:** All monitoring coordinates depict synthetic hydrological observation posts deployed for algorithmic verification and baseline risk auditing.",
        level="warning",
    )

    # ── Ingestion source selector ────────────────────────────────────────
    render_html(
        """
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.12em;
                    text-transform:uppercase;color:#666666;margin:1.2rem 0 0.4rem 0;">
            SELECT GEOSPATIAL DATA INGESTION SOURCE
        </div>
        """
    )

    source = st.radio(
        "Map data source:",
        ["Archived Sentinel GeoJSON", "Telemetry CSV (Latest Station Readings)", "Custom GeoJSON Ingestion"],
        horizontal=True,
        label_visibility="collapsed",
    )

    gdf = None
    risk_col = "flood_risk"
    level_col = "river_level_m"
    rain_col = "rainfall_mm"
    name_col = "station_name"

    if source == "Archived Sentinel GeoJSON":
        geojson_path = os.path.join(DATA_DIR, "demo_geospatial.geojson")
        if not os.path.exists(geojson_path):
            st.error("Sentinel GeoJSON archive not found at repository data path.")
            return
        try:
            gdf = load_geojson(geojson_path)
            render_html(
                f"""
                <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#2E6930;
                            padding:0.4rem 0.6rem;background:#F0FDF4;border:1px solid #BBF7D0;margin-bottom:1rem;">
                    RECORD INGESTION: {len(gdf)} GEOSPATIAL FEATURES LOADED FROM ARCHIVE
                </div>
                """
            )
        except Exception as e:
            st.error(f"Failed to ingest GeoJSON: {e}")
            return

        if "risk_zone" in gdf.columns:
            gdf["flood_risk"] = gdf["risk_zone"]

    elif source == "Telemetry CSV (Latest Station Readings)":
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
            editorial_callout("No telemetry data found in session. Visit Section I (Data Ingestion) first.", level="warning")
            return

        if TARGET_CLASSIFICATION not in df.columns:
            df = assign_risk_labels(df)

        latest = df.sort_values("date").groupby("station_id").tail(1).reset_index(drop=True)

        if "latitude" not in latest.columns or "longitude" not in latest.columns:
            st.error("Telemetry schema requires 'latitude' and 'longitude' coordinates.")
            return

        try:
            gdf = csv_to_geodataframe(latest, lat_col="latitude", lon_col="longitude")
            render_html(
                f"""
                <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#2E6930;
                            padding:0.4rem 0.6rem;background:#F0FDF4;border:1px solid #BBF7D0;margin-bottom:1rem;">
                    TELEMETRY MAPPED: {len(gdf)} ACTIVE MONITORING STATIONS LOCATED
                </div>
                """
            )
        except Exception as e:
            st.error(f"Error projecting coordinates to GeoDataFrame: {e}")
            return

        risk_col = TARGET_CLASSIFICATION

    else:
        sample_custom_path = os.path.join(DATA_DIR, "sample_custom_upload.geojson")
        col_up_btn, col_down_btn = st.columns([1, 1])
        with col_up_btn:
            btn_load_sample = st.button("Load Included Sample Custom GeoJSON (6 Basins)")
        with col_down_btn:
            if os.path.exists(sample_custom_path):
                with open(sample_custom_path, "r", encoding="utf-8") as f_samp:
                    st.download_button(
                        "Download Sample Custom GeoJSON File",
                        data=f_samp.read(),
                        file_name="sample_custom_upload.geojson",
                        mime="application/geo+json",
                    )

        uploaded = st.file_uploader("Upload custom GeoJSON archive", type=["geojson", "json"])

        if btn_load_sample or "loaded_custom_sample" in st.session_state:
            st.session_state["loaded_custom_sample"] = True
            try:
                gdf = load_geojson(sample_custom_path)
                render_html(
                    f"""
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#2E6930;
                                padding:0.4rem 0.6rem;background:#F0FDF4;border:1px solid #BBF7D0;margin-bottom:1rem;">
                        DEMO INGESTION: {len(gdf)} CUSTOM GEOSPATIAL STATIONS PARSED (Narmada, Mahanadi, Yamuna, Kaveri, Tapti, Beas)
                    </div>
                    """
                )
            except Exception as e:
                st.error(f"Error reading sample custom GeoJSON: {e}")
                return

        elif uploaded is not None:
            try:
                tmp = os.path.join(DATA_DIR, "_uploaded.geojson")
                with open(tmp, "wb") as f:
                    f.write(uploaded.read())
                gdf = load_geojson(tmp)
                render_html(
                    f"""
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#2E6930;
                                padding:0.4rem 0.6rem;background:#F0FDF4;border:1px solid #BBF7D0;margin-bottom:1rem;">
                        CUSTOM INGESTION: {len(gdf)} GEOSPATIAL FEATURES PARSED FROM UPLOADED FILE
                    </div>
                    """
                )
            except Exception as e:
                st.error(f"Error reading uploaded GeoJSON: {e}")
                return
        else:
            editorial_callout("Click 'Load Included Sample Custom GeoJSON' above or drag & drop any standard GeoJSON file.", level="info")
            return

        if gdf is not None and "risk_zone" in gdf.columns:
            gdf["flood_risk"] = gdf["risk_zone"]

    if gdf is None or len(gdf) == 0:
        st.warning("No geospatial observations to render.")
        return

    # Ensure required columns exist
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

    # ── Summary metrics strip ────────────────────────────────────────────
    render_html(
        """
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.12em;
                    text-transform:uppercase;color:#666666;margin:1.4rem 0 0.4rem 0;">
            HYDROLOGIC OBSERVATION SUMMARY
        </div>
        """
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("SURVEILLANCE POSTS", f"{len(gdf):,}", "Active stations")
    with col2:
        mean_stage = float(gdf["river_level_m"].mean())
        kpi_card("MEAN STAGE LEVEL", f"{mean_stage:.2f} m", "River gauge avg")
    with col3:
        max_rain = float(gdf["rainfall_mm"].max())
        kpi_card("PEAK 24H RAINFALL", f"{max_rain:.1f} mm", "Maximum station reading")
    with col4:
        crit_count = int((gdf["flood_risk"].str.lower() == "critical").sum())
        kpi_card("CRITICAL TIER STATIONS", f"{crit_count}", "Immediate threshold exceedance")

    # ── Render Map ───────────────────────────────────────────────────────
    render_html(
        """
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.12em;
                    text-transform:uppercase;color:#666666;margin:1.6rem 0 0.5rem 0;">
            INTERACTIVE HYDROLOGIC ATLAS & TELEMETRY NETWORK
        </div>
        """
    )

    try:
        m = create_risk_map(
            gdf,
            risk_col="flood_risk",
            level_col="river_level_m",
            rainfall_col="rainfall_mm",
            name_col="station_name",
        )
        # Wrap the folium html inside an editorial styled container
        render_html(
            """
            <div style="border:1px solid #171717;background:#FFFFFF;padding:2px;">
            """
        )
        components.html(m._repr_html_(), height=580, scrolling=False)
        render_html("</div>")
    except Exception as e:
        st.error(f"Cartographic rendering failed: {e}")

    # ── Cartographic Legend ──────────────────────────────────────────────
    render_html(
        f"""
        <div style="margin-top:1.5rem;padding:0.9rem 1.2rem;background:#FFFFFF;border:1px solid #D6D3CC;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:0.14em;
                        text-transform:uppercase;color:#666666;margin-bottom:0.6rem;">
                CARTOGRAPHIC RISK CLASSIFICATION LEGEND
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:2rem;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="width:12px;height:12px;background:{RISK_COLORS['Low']};border:1px solid #171717;display:inline-block;"></span>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#171717;">LOW RISK (SAFE EQUILIBRIUM)</span>
                </div>
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="width:12px;height:12px;background:{RISK_COLORS['Moderate']};border:1px solid #171717;display:inline-block;"></span>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#171717;">MODERATE (ADVISORY MONITORING)</span>
                </div>
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="width:12px;height:12px;background:{RISK_COLORS['High']};border:1px solid #171717;display:inline-block;"></span>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#171717;">HIGH (FLOOD WARNING ACTIVE)</span>
                </div>
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="width:12px;height:12px;background:{RISK_COLORS['Critical']};border:1px solid #171717;display:inline-block;"></span>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#171717;">CRITICAL (CATASTROPHIC INUNDATION)</span>
                </div>
            </div>
        </div>
        """
    )

    # ── Station Details Table ────────────────────────────────────────────
    render_html(
        """
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.12em;
                    text-transform:uppercase;color:#666666;margin:1.8rem 0 0.5rem 0;">
            TELEMETRY STATION INVENTORY & STAGE LEDGER
        </div>
        """
    )

    cols_to_show = [c for c in ["station_name", "flood_risk", "river_level_m", "rainfall_mm", "latitude", "longitude"] if c in gdf.columns]
    table_df = gdf[cols_to_show].copy() if cols_to_show else gdf.copy()
    if "river_level_m" in table_df.columns:
        table_df["river_level_m"] = table_df["river_level_m"].round(2)
    if "rainfall_mm" in table_df.columns:
        table_df["rainfall_mm"] = table_df["rainfall_mm"].round(1)

    st.dataframe(table_df, use_container_width=True, hide_index=True)


try:
    main()
except Exception as exc:
    st.error(f"Cartographic module exception: {exc}")
