"""FloodGuard AI — Data loading, cleaning, feature engineering & sequence creation."""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import logging

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import RISK_THRESHOLDS, RISK_CATEGORIES  # noqa: E402

logger = logging.getLogger(__name__)


# ============================= load_data ====================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load a CSV dataset and parse the *date* column as datetime.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with ``date`` parsed as ``datetime64``.
    """
    try:
        df = pd.read_csv(filepath, parse_dates=['date'])
        logger.info("Loaded %d rows from %s", len(df), filepath)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Error loading data from {filepath}: {e}") from e


# ============================= clean_data ===================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and forward-/backward-fill missing values per station.

    Steps
    -----
    1. Drop exact duplicate rows.
    2. Sort by ``station_id`` and ``date``.
    3. Group by ``station_id``, apply forward fill then backward fill on
       numeric columns.
    4. Ensure correct dtypes for key columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    try:
        initial_len = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info("Removed %d duplicate rows", initial_len - len(df))

        df = df.sort_values(['station_id', 'date']).reset_index(drop=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = (
            df.groupby('station_id')[numeric_cols]
            .transform(lambda g: g.ffill().bfill())
        )

        # Ensure dtypes
        df['date'] = pd.to_datetime(df['date'])
        for col in ['rainfall_mm', 'river_level_m', 'temperature_c', 'humidity_pct']:
            if col in df.columns:
                df[col] = df[col].astype(float)

        logger.info("Cleaned data: %d rows remaining", len(df))
        return df

    except Exception as e:
        raise RuntimeError(f"Error during data cleaning: {e}") from e


# ========================== engineer_features ===============================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features per station group.

    New columns
    -----------
    * ``prev_day_rainfall`` — previous day's rainfall (shift 1).
    * ``rainfall_3d_avg`` — 3-day rolling mean of rainfall.
    * ``level_change`` — day-over-day change in river level.
    * ``flood_risk`` — categorical risk label based on ``RISK_THRESHOLDS``.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with at least ``rainfall_mm`` and ``river_level_m``.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new feature columns appended.
    """
    try:
        df = df.sort_values(['station_id', 'date']).reset_index(drop=True)

        df['prev_day_rainfall'] = (
            df.groupby('station_id')['rainfall_mm'].shift(1)
        )
        df['rainfall_3d_avg'] = (
            df.groupby('station_id')['rainfall_mm']
            .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        )
        df['level_change'] = (
            df.groupby('station_id')['river_level_m'].diff()
        )

        # Assign flood_risk based on RISK_THRESHOLDS
        def _assign_risk(level: float) -> str:
            for category in RISK_CATEGORIES:
                lo, hi = RISK_THRESHOLDS[category]
                if lo <= level < hi:
                    return category
            return RISK_CATEGORIES[-1]  # fallback to highest

        df['flood_risk'] = df['river_level_m'].apply(_assign_risk)

        # Fill NaN in new columns for first rows of each group
        df['prev_day_rainfall'] = df['prev_day_rainfall'].fillna(0.0)
        df['level_change'] = df['level_change'].fillna(0.0)

        logger.info("Engineered features added; columns: %s", list(df.columns))
        return df

    except Exception as e:
        raise RuntimeError(f"Error during feature engineering: {e}") from e


# ========================== normalize_features ==============================

def normalize_features(
    df: pd.DataFrame,
    features: list,
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Apply MinMaxScaler to the specified feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    features : list
        Column names to normalise.

    Returns
    -------
    Tuple[pd.DataFrame, MinMaxScaler]
        A copy of *df* with normalised feature columns, and the fitted scaler.
    """
    try:
        scaler = MinMaxScaler()
        df_out = df.copy()
        df_out[features] = scaler.fit_transform(df_out[features].values)
        logger.info("Normalised %d features with MinMaxScaler", len(features))
        return df_out, scaler
    except Exception as e:
        raise RuntimeError(f"Error during normalisation: {e}") from e


# ============================ create_sequences ==============================

def create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    seq_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences for LSTM training.

    Parameters
    ----------
    data : np.ndarray
        2-D array of shape ``(n_samples, n_features)``.
    target : np.ndarray
        1-D array of shape ``(n_samples,)`` with the prediction target.
    seq_length : int
        Number of time-steps in each input window.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``X`` of shape ``(n_sequences, seq_length, n_features)`` and
        ``y`` of shape ``(n_sequences,)``.
    """
    try:
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length])
            y.append(target[i + seq_length])
        X = np.array(X)
        y = np.array(y)
        logger.info("Created %d sequences of length %d", len(X), seq_length)
        return X, y
    except Exception as e:
        raise RuntimeError(f"Error creating sequences: {e}") from e


# ========================= chronological_split ==============================

def chronological_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame into train and test sets (no shuffle).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame sorted chronologically.
    test_ratio : float, optional
        Fraction of rows to reserve for the test set (default 0.2).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)``
    """
    try:
        split_idx = int(len(df) * (1 - test_ratio))
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        logger.info(
            "Chronological split: train=%d, test=%d (ratio=%.2f)",
            len(train_df), len(test_df), test_ratio,
        )
        return train_df, test_df
    except Exception as e:
        raise RuntimeError(f"Error during chronological split: {e}") from e
