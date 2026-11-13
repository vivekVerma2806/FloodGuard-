"""Tests for src/geospatial.py."""

import os
import sys

import folium
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.geospatial import (
    create_risk_map,
    csv_to_geodataframe,
    get_risk_color,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetRiskColor:
    @pytest.mark.parametrize(
        "risk, expected",
        [
            ("Low", "green"),
            ("Moderate", "orange"),
            ("High", "red"),
            ("Critical", "darkred"),
        ],
    )
    def test_get_risk_color(self, risk, expected):
        assert get_risk_color(risk) == expected

    def test_get_risk_color_unknown_returns_gray(self):
        assert get_risk_color("Unknown") == "gray"


class TestCsvToGeoDataFrame:
    def test_csv_to_geodataframe(self):
        """Check geometry column exists after conversion."""
        df = pd.DataFrame(
            {
                "station_id": ["A", "B"],
                "latitude": [19.99, 26.14],
                "longitude": [73.78, 91.73],
                "flood_risk": ["Low", "High"],
            }
        )
        gdf = csv_to_geodataframe(df)

        assert "geometry" in gdf.columns
        assert len(gdf) == 2
        # Each geometry should be a Point
        assert all(gdf.geometry.geom_type == "Point")


class TestCreateRiskMap:
    def test_create_risk_map(self):
        """Check returns a folium.Map."""
        df = pd.DataFrame(
            {
                "station_name": ["StationA", "StationB", "StationC"],
                "latitude": [19.99, 26.14, 25.31],
                "longitude": [73.78, 91.73, 82.97],
                "rainfall_mm": [10.0, 30.0, 50.0],
                "river_level_m": [2.0, 5.5, 8.0],
                "flood_risk": ["Low", "High", "Critical"],
            }
        )
        gdf = csv_to_geodataframe(df)
        m = create_risk_map(gdf)

        assert isinstance(m, folium.Map)
