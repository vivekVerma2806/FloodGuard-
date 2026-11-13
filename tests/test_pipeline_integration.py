"""End-to-end pipeline integration tests for FloodGuard AI."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DATA_DIR, FEATURES, TARGET_REGRESSION, TARGET_CLASSIFICATION, RISK_CATEGORIES
from src.data_preprocessing import (
    load_data, clean_data, engineer_features,
    normalize_features, create_sequences, chronological_split
)
from src.lstm_model import build_lstm_model, train_lstm_model, predict_lstm
from src.ml_benchmarking import compare_models, evaluate_regression
from src.risk_classification import (
    assign_risk_labels, train_risk_classifier,
    evaluate_classification, get_confusion_matrix_df
)
from src.geospatial import csv_to_geodataframe, create_risk_map, get_risk_color


@pytest.fixture(scope="module")
def prepared_data():
    """Prepare cleaned and engineered data for testing."""
    demo_csv = os.path.join(DATA_DIR, "demo_dataset.csv")
    assert os.path.exists(demo_csv), "Demo CSV dataset must exist"
    df = load_data(demo_csv)
    df_clean = clean_data(df)
    df_feat = engineer_features(df_clean)
    return df_feat


class TestEndToEndPipeline:

    def test_pipeline_data_integrity(self, prepared_data):
        """Verify data cleaning and feature engineering integrity."""
        assert prepared_data.isnull().sum().sum() == 0
        assert prepared_data.duplicated().sum() == 0
        assert "prev_day_rainfall" in prepared_data.columns
        assert "rainfall_3d_avg" in prepared_data.columns
        assert "level_change" in prepared_data.columns
        assert "flood_risk" in prepared_data.columns
        assert set(prepared_data["flood_risk"].unique()).issubset(set(RISK_CATEGORIES))

    def test_pipeline_chronological_split(self, prepared_data):
        """Verify that chronological splitting prevents lookahead bias."""
        station_df = prepared_data[prepared_data["station_id"] == "STN001"].sort_values("date").reset_index(drop=True)
        train_df, test_df = chronological_split(station_df, test_ratio=0.2)
        assert train_df["date"].max() < test_df["date"].min()

    def test_pipeline_lstm_flow(self, prepared_data):
        """Verify the full LSTM flow from sequence generation to inference."""
        station_df = prepared_data[prepared_data["station_id"] == "STN001"].sort_values("date").reset_index(drop=True)
        train_df, test_df = chronological_split(station_df, test_ratio=0.2)

        feat_cols = [c for c in FEATURES if c in station_df.columns]
        train_norm, scaler = normalize_features(train_df, feat_cols)
        test_norm = test_df.copy()
        test_norm[feat_cols] = scaler.transform(test_df[feat_cols].values)

        target_idx = feat_cols.index(TARGET_REGRESSION)
        X_tr, y_tr = create_sequences(train_norm[feat_cols].values, train_norm[feat_cols].values[:, target_idx], 5)
        X_te, y_te = create_sequences(test_norm[feat_cols].values, test_norm[feat_cols].values[:, target_idx], 5)

        model = build_lstm_model(input_shape=(5, len(feat_cols)), units=16)
        history = train_lstm_model(model, X_tr[:50], y_tr[:50], X_te[:20], y_te[:20], epochs=2, batch_size=16)

        assert "loss" in history.history
        preds = predict_lstm(model, X_te[:20])
        assert preds.shape == (20,)
        assert not np.isnan(preds).any()

    def test_pipeline_benchmarking_flow(self, prepared_data):
        """Verify that regression benchmarking executes and returns valid metrics."""
        station_df = prepared_data[prepared_data["station_id"] == "STN001"].sort_values("date").reset_index(drop=True)
        train_df, test_df = chronological_split(station_df, test_ratio=0.2)

        reg_cols = [c for c in FEATURES if c in station_df.columns and c != TARGET_REGRESSION]
        X_tr = train_df[reg_cols].values[:100]
        y_tr = train_df[TARGET_REGRESSION].values[:100]
        X_te = test_df[reg_cols].values[:30]
        y_te = test_df[TARGET_REGRESSION].values[:30]

        df_bench, models_dict = compare_models(X_tr, y_tr, X_te, y_te)
        assert len(df_bench) == 3
        assert set(df_bench["Model"]).issuperset({"Linear Regression", "Random Forest", "Gradient Boosting"})
        assert all(df_bench["MAE"] > 0)
        assert all(df_bench["RMSE"] > 0)

    def test_pipeline_classification_flow(self, prepared_data):
        """Verify risk classification training and metric extraction."""
        cls_cols = [c for c in FEATURES if c in prepared_data.columns and c != TARGET_REGRESSION]
        train_df, test_df = chronological_split(prepared_data, test_ratio=0.2)

        X_tr = train_df[cls_cols].values[:200]
        y_tr = train_df[TARGET_CLASSIFICATION].values[:200]
        X_te = test_df[cls_cols].values[:50]
        y_te = test_df[TARGET_CLASSIFICATION].values[:50]

        clf = train_risk_classifier(X_tr, y_tr, n_estimators=20)
        preds = clf.predict(X_te)
        probs = clf.predict_proba(X_te)

        metrics = evaluate_classification(y_te, preds, probs, labels=RISK_CATEGORIES)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "pr_auc" in metrics

    def test_pipeline_geospatial_flow(self, prepared_data):
        """Verify that geodataframe creation and folium map rendering succeed."""
        latest = prepared_data.sort_values("date").groupby("station_id").tail(1).reset_index(drop=True)
        gdf = csv_to_geodataframe(latest, lat_col="latitude", lon_col="longitude")
        assert len(gdf) == 5
        m = create_risk_map(gdf, risk_col="flood_risk", level_col="river_level_m", rainfall_col="rainfall_mm", name_col="station_name")
        html = m._repr_html_()
        assert len(html) > 500
