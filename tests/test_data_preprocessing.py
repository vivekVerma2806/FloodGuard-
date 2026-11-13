"""Tests for src/data_preprocessing.py."""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path so `config` resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_preprocessing import (
    chronological_split,
    clean_data,
    create_sequences,
    engineer_features,
    load_data,
    normalize_features,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_sample_df(rows: int = 30) -> pd.DataFrame:
    """Return a small DataFrame that mimics the demo CSV schema."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "station_id": ["STN001"] * rows,
            "station_name": ["Test Station"] * rows,
            "latitude": [19.9975] * rows,
            "longitude": [73.7898] * rows,
            "rainfall_mm": np.random.uniform(0, 50, rows),
            "river_level_m": np.random.uniform(1, 8, rows),
            "temperature_c": np.random.uniform(20, 40, rows),
            "humidity_pct": np.random.uniform(40, 100, rows),
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_load_data(self, tmp_path):
        """Create a temp CSV, load it, and verify a DataFrame is returned."""
        df = _make_sample_df(10)
        csv_path = tmp_path / "sample.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_data(str(csv_path))

        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == 10
        assert "date" in loaded.columns
        assert pd.api.types.is_datetime64_any_dtype(loaded["date"])

    def test_load_data_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path.csv")


class TestCleanData:
    def test_clean_data_removes_duplicates_and_fills_nan(self):
        """Create a df with NaN and duplicates; verify cleaning."""
        df = _make_sample_df(20)
        # Inject duplicates
        df = pd.concat([df, df.iloc[:2]], ignore_index=True)
        # Inject NaN
        df.loc[3, "rainfall_mm"] = np.nan
        df.loc[5, "river_level_m"] = np.nan

        cleaned = clean_data(df)

        # Duplicates removed
        assert len(cleaned) <= 20
        assert cleaned.duplicated().sum() == 0
        # NaN filled
        assert cleaned["rainfall_mm"].isna().sum() == 0
        assert cleaned["river_level_m"].isna().sum() == 0


class TestEngineerFeatures:
    def test_engineer_features_adds_columns(self):
        """Verify new columns are added correctly."""
        df = _make_sample_df(20)
        df["date"] = pd.to_datetime(df["date"])
        result = engineer_features(df)

        expected_new_cols = [
            "prev_day_rainfall",
            "rainfall_3d_avg",
            "level_change",
            "flood_risk",
        ]
        for col in expected_new_cols:
            assert col in result.columns, f"Missing column: {col}"

        # flood_risk should only contain valid risk categories
        valid = {"Low", "Moderate", "High", "Critical"}
        assert set(result["flood_risk"].unique()).issubset(valid)


class TestNormalizeFeatures:
    def test_normalize_features_range(self):
        """Check normalised values are in [0, 1] range."""
        df = _make_sample_df(20)
        features = ["rainfall_mm", "river_level_m", "temperature_c"]

        normed, scaler = normalize_features(df, features)

        for col in features:
            assert normed[col].min() >= 0.0 - 1e-9
            assert normed[col].max() <= 1.0 + 1e-9

        # Scaler should be fitted
        assert hasattr(scaler, "data_min_")


class TestCreateSequences:
    def test_create_sequences_shapes(self):
        """Check output shapes of sliding-window sequences."""
        n_samples, n_features, seq_len = 50, 3, 5
        data = np.random.rand(n_samples, n_features)
        target = np.random.rand(n_samples)

        X, y = create_sequences(data, target, seq_len)

        expected_n = n_samples - seq_len
        assert X.shape == (expected_n, seq_len, n_features)
        assert y.shape == (expected_n,)


class TestChronologicalSplit:
    def test_chronological_split_no_leakage(self):
        """Verify test dates are strictly after train dates."""
        df = _make_sample_df(50)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        train, test = chronological_split(df, test_ratio=0.2)

        assert len(train) > 0
        assert len(test) > 0
        assert len(train) + len(test) == len(df)
        # No temporal leakage
        assert train["date"].max() < test["date"].min()
