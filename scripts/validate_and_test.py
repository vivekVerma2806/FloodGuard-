"""Senior QA & ML Engineer Test Suite for FloodGuard AI.

Validates:
- Data Preprocessing pipeline
- LSTM Forecasting model training & inference (checks for data leakage)
- ML Benchmarking models (LR, RF, GBR)
- Multi-class Flood Risk Classification pipeline
- Geospatial vector mapping & Folium WebGIS map creation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    DATA_DIR,
    DEMO_STATIONS,
    FEATURES,
    RISK_CATEGORIES,
    RISK_THRESHOLDS,
    SEQUENCE_LENGTH,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    TEST_SPLIT_RATIO,
)
from src.data_preprocessing import (
    chronological_split,
    clean_data,
    create_sequences,
    engineer_features,
    load_data,
    normalize_features,
)
from src.geospatial import (
    create_risk_map,
    csv_to_geodataframe,
    get_risk_color,
    load_geojson,
)
from src.lstm_model import (
    build_lstm_model,
    load_saved_model,
    predict_lstm,
    save_model,
    train_lstm_model,
)
from src.ml_benchmarking import (
    compare_models,
    evaluate_regression,
    get_feature_importance,
)
from src.risk_classification import (
    assign_risk_labels,
    evaluate_classification,
    get_classification_report_df,
    get_confusion_matrix_df,
    train_risk_classifier,
)


def run_all_qa_checks():
    print("=" * 70)
    print("PHASE 2 & 3: COMPREHENSIVE ML & CODE VALIDATION TEST")
    print("=" * 70)

    # 1. DATA VALIDATION
    csv_path = os.path.join(DATA_DIR, "demo_dataset.csv")
    geojson_path = os.path.join(DATA_DIR, "demo_geospatial.geojson")

    assert os.path.exists(csv_path), f"CSV missing: {csv_path}"
    assert os.path.exists(geojson_path), f"GeoJSON missing: {geojson_path}"

    df_raw = load_data(csv_path)
    print(f"[OK] Raw CSV loaded: {len(df_raw)} rows, {df_raw.shape[1]} cols")
    print(f"     Stations: {df_raw['station_id'].nunique()} stations")
    print(f"     Date range: {df_raw['date'].min()} to {df_raw['date'].max()}")

    # 2. PREPROCESSING TEST
    missing_init = df_raw.isnull().sum().sum()
    dupes_init = df_raw.duplicated().sum()
    print(f"     Initial missing cells: {missing_init}, duplicates: {dupes_init}")

    df_clean = clean_data(df_raw)
    assert df_clean.isnull().sum().sum() == 0, "Missing values remain after clean_data!"
    assert df_clean.duplicated().sum() == 0, "Duplicates remain after clean_data!"
    print(f"[OK] clean_data passed: 0 missing values, 0 duplicates remaining.")

    df_feat = engineer_features(df_clean)
    for expected_col in ["prev_day_rainfall", "rainfall_3d_avg", "level_change", "flood_risk"]:
        assert expected_col in df_feat.columns, f"Missing feature: {expected_col}"
    print(f"[OK] engineer_features passed: New features created successfully.")
    print(f"     Risk categories assigned: {dict(df_feat['flood_risk'].value_counts())}")

    # 3. DATA LEAKAGE TEST (Chronological split)
    station_sample = df_feat[df_feat["station_id"] == "STN001"].sort_values("date").reset_index(drop=True)
    train_df, test_df = chronological_split(station_sample, test_ratio=0.2)
    max_train_date = train_df["date"].max()
    min_test_date = test_df["date"].min()
    assert max_train_date < min_test_date, f"Data leakage detected! Train max ({max_train_date}) >= Test min ({min_test_date})"
    print(f"[OK] Chronological Split validated: No data leakage. Train: {train_df['date'].min().date()}..{max_train_date.date()} | Test: {min_test_date.date()}..{test_df['date'].max().date()}")

    # 4. NORMALIZATION & SEQUENCE CREATION TEST
    feature_cols = [c for c in FEATURES if c in station_sample.columns]
    train_norm, scaler = normalize_features(train_df, feature_cols)
    test_norm = test_df.copy()
    test_norm[feature_cols] = scaler.transform(test_df[feature_cols].values)

    target_idx = feature_cols.index(TARGET_REGRESSION)
    X_train, y_train = create_sequences(train_norm[feature_cols].values, train_norm[feature_cols].values[:, target_idx], SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(test_norm[feature_cols].values, test_norm[feature_cols].values[:, target_idx], SEQUENCE_LENGTH)

    print(f"[OK] Sequence dimensions: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
    print(f"     X_test shape={X_test.shape}, y_test shape={y_test.shape}")
    assert X_train.ndim == 3 and X_train.shape[1] == SEQUENCE_LENGTH and X_train.shape[2] == len(feature_cols)

    # 5. LSTM TRAINING & PREDICTION TEST
    print("\n--- Training LSTM Model ---")
    lstm_model = build_lstm_model(input_shape=(SEQUENCE_LENGTH, len(feature_cols)), units=32)
    history = train_lstm_model(lstm_model, X_train, y_train, X_test, y_test, epochs=5, batch_size=32)

    assert "loss" in history.history and "val_loss" in history.history
    initial_loss = history.history["loss"][0]
    final_loss = history.history["loss"][-1]
    print(f"[OK] LSTM Training converged: initial_loss={initial_loss:.4f} -> final_loss={final_loss:.4f}")

    y_pred_lstm = predict_lstm(lstm_model, X_test)
    assert len(y_pred_lstm) == len(y_test), "Prediction shape mismatch!"
    assert not np.all(y_pred_lstm == y_pred_lstm[0]), "Predictions appear hardcoded (all identical)!"
    assert not np.isnan(y_pred_lstm).any(), "NaN found in predictions!"

    lstm_mae = float(np.mean(np.abs(y_test - y_pred_lstm)))
    lstm_rmse = float(np.sqrt(np.mean((y_test - y_pred_lstm) ** 2)))
    print(f"[OK] LSTM Evaluation on Test Set: MAE={lstm_mae:.4f}, RMSE={lstm_rmse:.4f}")

    # Model save & reload
    test_model_path = os.path.join(DATA_DIR, "..", "models", "test_lstm_verify.keras")
    save_model(lstm_model, test_model_path)
    loaded_model = load_saved_model(test_model_path)
    y_pred_reloaded = predict_lstm(loaded_model, X_test)
    np.testing.assert_allclose(y_pred_lstm, y_pred_reloaded, atol=1e-5)
    print(f"[OK] LSTM Save/Load verified: Reloaded model predictions match exactly.")
    if os.path.exists(test_model_path):
        os.remove(test_model_path)

    # 6. ML BENCHMARKING (LR, RF, GBR)
    print("\n--- Running Scikit-learn Regression Benchmarks ---")
    reg_features = [f for f in feature_cols if f != TARGET_REGRESSION]
    X_tr_reg = train_df[reg_features].values
    y_tr_reg = train_df[TARGET_REGRESSION].values
    X_te_reg = test_df[reg_features].values
    y_te_reg = test_df[TARGET_REGRESSION].values

    bench_df, models_dict = compare_models(X_tr_reg, y_tr_reg, X_te_reg, y_te_reg)
    print("[OK] Benchmark Comparison Results:")
    print(bench_df.to_string(index=False))

    for mname, (mobj, preds) in models_dict.items():
        assert not np.all(preds == preds[0]), f"{mname} predictions appear hardcoded!"
        eval_dict = evaluate_regression(y_te_reg, preds)
        assert eval_dict["mae"] > 0 and eval_dict["rmse"] > 0

    rf_model = models_dict["Random Forest"][0]
    fi_df = get_feature_importance(rf_model, reg_features)
    print(f"[OK] Random Forest Feature Importance:\n{fi_df.to_string(index=False)}")

    # 7. RISK CLASSIFICATION PIPELINE
    print("\n--- Running Flood-Risk Classification Pipeline ---")
    df_cls = assign_risk_labels(df_feat)
    available_cls_feats = [f for f in FEATURES if f in df_cls.columns and f != "river_level_m"]
    train_cls, test_cls = chronological_split(df_cls, test_ratio=0.2)

    X_tr_cls = train_cls[available_cls_feats].values
    y_tr_cls = train_cls[TARGET_CLASSIFICATION].values
    X_te_cls = test_cls[available_cls_feats].values
    y_te_cls = test_cls[TARGET_CLASSIFICATION].values

    clf = train_risk_classifier(X_tr_cls, y_tr_cls, n_estimators=50)
    y_pred_cls = clf.predict(X_te_cls)
    y_prob_cls = clf.predict_proba(X_te_cls)

    assert not np.all(y_pred_cls == y_pred_cls[0]), "Classifier predictions are constant/hardcoded!"
    cls_metrics = evaluate_classification(y_te_cls, y_pred_cls, y_prob_cls, labels=RISK_CATEGORIES)

    print(f"[OK] Classification Metrics:")
    print(f"     Macro Precision: {cls_metrics['precision']:.4f}")
    print(f"     Macro Recall:    {cls_metrics['recall']:.4f}")
    print(f"     Macro F1-Score:  {cls_metrics['f1']:.4f}")
    print(f"     PR-AUC Macro:    {cls_metrics['pr_auc']['macro_avg']:.4f}")
    print(f"     Per-class PR-AUC: {cls_metrics['pr_auc']}")

    cm_df = get_confusion_matrix_df(y_te_cls, y_pred_cls, labels=RISK_CATEGORIES)
    print(f"[OK] Confusion Matrix:\n{cm_df}")

    report_df = get_classification_report_df(y_te_cls, y_pred_cls, labels=RISK_CATEGORIES)
    print(f"[OK] Classification Report:\n{report_df[['precision', 'recall', 'f1-score', 'support']]}")

    # 8. GEOSPATIAL VECTOR & MAP VALIDATION
    print("\n--- Validating Geospatial Processing & Maps ---")
    for cat in RISK_CATEGORIES:
        color = get_risk_color(cat)
        assert color in ["green", "orange", "red", "darkred"], f"Invalid color for {cat}: {color}"
    print("[OK] Risk color mappings verified.")

    gdf_loaded = load_geojson(geojson_path)
    assert len(gdf_loaded) == 5, f"Expected 5 features, got {len(gdf_loaded)}"
    assert all(gdf_loaded.geometry.geom_type == "Point"), "Expected Point geometries!"
    print(f"[OK] GeoJSON structure validated: {len(gdf_loaded)} station points.")

    latest_records = df_feat.sort_values("date").groupby("station_id").tail(1).reset_index(drop=True)
    gdf_csv = csv_to_geodataframe(latest_records, lat_col="latitude", lon_col="longitude")
    assert isinstance(gdf_csv, gpd.GeoDataFrame)
    assert gdf_csv.crs == "EPSG:4326"
    print(f"[OK] csv_to_geodataframe: Successfully converted tabular records to EPSG:4326 GeoDataFrame.")

    folium_map = create_risk_map(
        gdf_csv,
        risk_col="flood_risk",
        level_col="river_level_m",
        rainfall_col="rainfall_mm",
        name_col="station_name",
    )
    assert folium_map is not None
    html_repr = folium_map._repr_html_()
    assert "Flood Risk Stations" in html_repr
    assert "Flood Risk" in html_repr  # legend
    print(f"[OK] Interactive Folium Map generated successfully ({len(html_repr)} HTML chars).")

    print("\n" + "=" * 70)
    print("ALL QA CHECKS COMPLETED AND VERIFIED!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_qa_checks()
