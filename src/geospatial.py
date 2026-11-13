"""FloodGuard AI - Geospatial Visualization Module.

Provides utilities for converting tabular data to GeoDataFrames, loading
GeoJSON layers, and creating interactive Folium maps with risk-coloured
markers, legends, and optional choropleth overlays.
"""

import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DEMO_STATIONS, RISK_CATEGORIES

# ── colour mapping ──────────────────────────────────────────────────────────

_RISK_COLORS = {
    'Low': 'green',
    'Moderate': 'orange',
    'High': 'red',
    'Critical': 'darkred',
}


def get_risk_color(risk: str) -> str:
    """Return a map-marker colour string for a flood-risk category.

    Args:
        risk: One of ``'Low'``, ``'Moderate'``, ``'High'``, ``'Critical'``.

    Returns:
        Colour name understood by Folium (e.g. ``'green'``, ``'darkred'``).
        Falls back to ``'gray'`` for unrecognised categories.
    """
    return _RISK_COLORS.get(risk, 'gray')


# ── data conversion ─────────────────────────────────────────────────────────

def csv_to_geodataframe(
    df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
) -> gpd.GeoDataFrame:
    """Convert a regular DataFrame with lat/lon columns to a GeoDataFrame.

    Args:
        df: Source DataFrame.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.

    Returns:
        A ``GeoDataFrame`` with ``Point`` geometry in EPSG:4326.

    Raises:
        KeyError: If *lat_col* or *lon_col* is missing.
        ValueError: If coordinates contain non-numeric data.
        RuntimeError: On any other failure.
    """
    try:
        missing = [c for c in (lat_col, lon_col) if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing column(s) {missing} in DataFrame. "
                f"Available: {list(df.columns)}"
            )

        if not pd.api.types.is_numeric_dtype(df[lat_col]) or not pd.api.types.is_numeric_dtype(df[lon_col]):
            raise ValueError(
                f"Columns '{lat_col}' and '{lon_col}' must be numeric."
            )

        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs='EPSG:4326',
        )
        return gdf

    except (KeyError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to convert DataFrame to GeoDataFrame: {e}") from e


def load_geojson(filepath: str) -> gpd.GeoDataFrame:
    """Load a GeoJSON file and return it as a GeoDataFrame.

    Args:
        filepath: Path to the ``.geojson`` file.

    Returns:
        A ``GeoDataFrame`` parsed from the file.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        RuntimeError: On parsing or I/O errors.
    """
    try:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"GeoJSON file not found: {filepath}")

        gdf = gpd.read_file(filepath)
        return gdf

    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load GeoJSON from '{filepath}': {e}") from e


# ── interactive maps ─────────────────────────────────────────────────────────

def _build_legend_html() -> str:
    """Return an HTML legend block styled like a vintage cartographic index."""
    items = ''.join(
        f'<li style="display:flex;align-items:center;margin-bottom:4px;">'
        f'<span style="background:{_RISK_COLORS[cat]};width:10px;height:10px;display:inline-block;margin-right:8px;border:1px solid #171717;border-radius:1px;"></span>'
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#171717;letter-spacing:0.04em;">{cat.upper()}</span></li>'
        for cat in RISK_CATEGORIES
    )
    return (
        '<div style="position:fixed;bottom:25px;left:25px;z-index:1000;'
        'background:#FAF8F5;border:1px solid #171717;padding:10px 14px;border-radius:1px;'
        'box-shadow:0 1px 3px rgba(0,0,0,0.1);font-family:\'IBM Plex Mono\',monospace;">'
        '<div style="font-size:9px;letter-spacing:0.14em;text-transform:uppercase;color:#666;border-bottom:1px solid #D6D3CC;padding-bottom:4px;margin-bottom:6px;">// CARTOGRAPHIC RISK TIER</div>'
        f'<ul style="list-style:none;padding:0;margin:0;">{items}</ul>'
        '</div>'
    )


def create_risk_map(
    gdf: gpd.GeoDataFrame,
    risk_col: str = 'flood_risk',
    level_col: str = 'river_level_m',
    rainfall_col: str = 'rainfall_mm',
    name_col: str = 'station_name',
) -> folium.Map:
    """Create an interactive Folium map colour-coded by flood-risk category.

    The function aggregates data per station (mean water level and rainfall,
    latest risk category) and places ``CircleMarker``s with pop-ups showing
    station name, water level, rainfall, and risk category.

    Args:
        gdf: GeoDataFrame with ``Point`` geometry and the columns listed
             below.
        risk_col: Column containing risk-category strings.
        level_col: Column with river water-level values (metres).
        rainfall_col: Column with rainfall values (mm).
        name_col: Column with human-readable station names.

    Returns:
        A ``folium.Map`` object ready for display or ``.save()``.

    Raises:
        KeyError: If required columns are missing from *gdf*.
        RuntimeError: On any other failure.
    """
    try:
        required = [risk_col, level_col, rainfall_col, name_col]
        missing = [c for c in required if c not in gdf.columns]
        if missing:
            raise KeyError(
                f"GeoDataFrame is missing required column(s): {missing}"
            )

        # Aggregate per station: mean numeric, latest risk
        agg_dict = {
            level_col: 'mean',
            rainfall_col: 'mean',
            risk_col: 'last',
            'geometry': 'first',
        }
        station_gdf = gdf.groupby(name_col, as_index=False).agg(agg_dict)
        station_gdf = gpd.GeoDataFrame(station_gdf, geometry='geometry', crs='EPSG:4326')

        # Centre map on data centroid
        centre_lat = station_gdf.geometry.y.mean()
        centre_lon = station_gdf.geometry.x.mean()
        m = folium.Map(location=[centre_lat, centre_lon], zoom_start=5, tiles='OpenStreetMap')

        # Feature group for markers
        fg = folium.FeatureGroup(name='Flood Risk Stations')

        for _, row in station_gdf.iterrows():
            risk = row[risk_col]
            color = get_risk_color(risk)
            popup_html = (
                f'<div style="font-family:\'IBM Plex Mono\',Consolas,monospace;font-size:12px;color:#171717;background:#FAF8F5;padding:6px 8px;line-height:1.5;border:1px solid #171717;">'
                f'<div style="font-weight:700;font-size:13px;border-bottom:1px solid #171717;padding-bottom:3px;margin-bottom:5px;letter-spacing:0.02em;">{row[name_col]}</div>'
                f'<div>STAGE LEVEL : <b>{row[level_col]:.2f} m</b></div>'
                f'<div>PRECIPITATION: <b>{row[rainfall_col]:.1f} mm</b></div>'
                f'<div style="margin-top:5px;padding-top:3px;border-top:1px dashed #D6D3CC;">THREAT TIER : <b style="color:{color};">{risk.upper()}</b></div>'
                f'</div>'
            )
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=10,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=row[name_col],
            ).add_to(fg)

        fg.add_to(m)

        # Legend
        legend = folium.Element(_build_legend_html())
        m.get_root().html.add_child(legend)

        # Layer control
        folium.LayerControl().add_to(m)

        return m

    except KeyError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to create risk map: {e}") from e


def create_choropleth_map(gdf: gpd.GeoDataFrame, value_col: str) -> folium.Map:
    """Create a choropleth map from polygon geometries, or fall back to markers.

    If the GeoDataFrame contains ``Polygon`` / ``MultiPolygon`` geometries
    the function renders a Folium choropleth shaded by *value_col*.
    Otherwise it produces a simple ``CircleMarker`` map.

    Args:
        gdf: GeoDataFrame with geometry and a numeric *value_col*.
        value_col: Column whose values drive the colour scale.

    Returns:
        A ``folium.Map`` object.

    Raises:
        KeyError: If *value_col* is missing.
        RuntimeError: On any other failure.
    """
    try:
        if value_col not in gdf.columns:
            raise KeyError(
                f"Column '{value_col}' not found. Available: {list(gdf.columns)}"
            )

        centre_lat = gdf.geometry.centroid.y.mean()
        centre_lon = gdf.geometry.centroid.x.mean()
        m = folium.Map(location=[centre_lat, centre_lon], zoom_start=5, tiles='OpenStreetMap')

        geom_types = set(gdf.geometry.geom_type)
        has_polygons = bool(geom_types & {'Polygon', 'MultiPolygon'})

        if has_polygons:
            # Build GeoJSON string for choropleth
            geojson_data = gdf.to_json()

            # Ensure an id column exists for the key_on binding
            if 'id' not in gdf.columns:
                gdf = gdf.copy()
                gdf['id'] = gdf.index.astype(str)

            folium.Choropleth(
                geo_data=geojson_data,
                data=gdf,
                columns=['id', value_col],
                key_on='feature.id',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.3,
                legend_name=value_col,
            ).add_to(m)
        else:
            # Fallback: circle-marker map sized by value_col
            min_val = gdf[value_col].min()
            max_val = gdf[value_col].max()
            val_range = max_val - min_val if max_val != min_val else 1.0

            for _, row in gdf.iterrows():
                normalised = (row[value_col] - min_val) / val_range
                radius = 5 + normalised * 15
                folium.CircleMarker(
                    location=[row.geometry.centroid.y, row.geometry.centroid.x],
                    radius=radius,
                    color='steelblue',
                    fill=True,
                    fill_color='steelblue',
                    fill_opacity=0.6,
                    tooltip=f"{value_col}: {row[value_col]:.2f}",
                ).add_to(m)

        folium.LayerControl().add_to(m)
        return m

    except KeyError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to create choropleth map: {e}") from e
