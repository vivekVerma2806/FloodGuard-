# FloodGuard AI - Architecture Plan

## Directory Structure
```
FloodGuard-/
├── app.py                          # Main Streamlit entry point
├── requirements.txt
├── README.md
├── .gitignore
├── config.py                       # Global config constants
├── data/
│   ├── demo_dataset.csv            # Generated demo dataset
│   └── demo_geospatial.geojson     # Demo geospatial data
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data loading, cleaning, feature engineering
│   ├── lstm_model.py               # LSTM flood forecasting model
│   ├── ml_benchmarking.py          # Linear Regression, RF, GBR comparison
│   ├── risk_classification.py      # Flood risk classification pipeline
│   └── geospatial.py               # GeoPandas/Folium visualization
├── pages/
│   ├── __init__.py
│   ├── 1_📊_Data_Upload.py
│   ├── 2_🌊_Flood_Forecasting.py
│   ├── 3_📈_Model_Benchmarking.py
│   ├── 4_⚠️_Risk_Classification.py
│   ├── 5_🗺️_Geospatial_Map.py
│   └── 6_📋_Model_Evaluation.py
├── models/                         # Saved model artifacts (gitignored except .gitkeep)
│   └── .gitkeep
└── tests/
    ├── __init__.py
    ├── test_data_preprocessing.py
    ├── test_lstm_model.py
    ├── test_ml_benchmarking.py
    ├── test_risk_classification.py
    └── test_geospatial.py
```

## Module Contracts

### config.py
```python
RANDOM_SEED = 42
SEQUENCE_LENGTH = 10
TEST_SPLIT_RATIO = 0.2
FEATURES = ['rainfall_mm', 'river_level_m', 'temperature_c', 'humidity_pct', 'prev_day_rainfall', 'rainfall_3d_avg', 'level_change']
TARGET_REGRESSION = 'river_level_m'
TARGET_CLASSIFICATION = 'flood_risk'
RISK_CATEGORIES = ['Low', 'Moderate', 'High', 'Critical']
RISK_THRESHOLDS = {'Low': 3.0, 'Moderate': 5.0, 'High': 7.0}  # river_level_m thresholds
MODEL_DIR = 'models'
DATA_DIR = 'data'
```

### data/demo_dataset.csv columns
date, station_id, station_name, latitude, longitude, rainfall_mm, river_level_m, temperature_c, humidity_pct

- 5 years of daily data (2019-01-01 to 2023-12-31)
- 5 stations with realistic Indian river basin names
- Seasonal patterns (monsoon Jun-Sep has higher rainfall/levels)
- Synthetic but physically plausible values

### src/data_preprocessing.py exports
- load_data(filepath: str) -> pd.DataFrame
- clean_data(df: pd.DataFrame) -> pd.DataFrame  # missing values, duplicates
- engineer_features(df: pd.DataFrame) -> pd.DataFrame  # adds prev_day_rainfall, rainfall_3d_avg, level_change, flood_risk
- normalize_features(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, MinMaxScaler]
- create_sequences(data: np.ndarray, target: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]
- chronological_split(df: pd.DataFrame, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]

### src/lstm_model.py exports
- build_lstm_model(input_shape: tuple, units: int = 64) -> tf.keras.Model
- train_lstm_model(model, X_train, y_train, X_val, y_val, epochs, batch_size) -> History
- predict_lstm(model, X) -> np.ndarray
- save_model(model, path: str)
- load_model(path: str) -> tf.keras.Model

### src/ml_benchmarking.py exports
- train_linear_regression(X_train, y_train) -> model
- train_random_forest(X_train, y_train) -> model
- train_gradient_boosting(X_train, y_train) -> model
- evaluate_regression(y_true, y_pred) -> dict  # {'mae': float, 'rmse': float}
- compare_models(results: dict) -> pd.DataFrame  # model_name -> metrics

### src/risk_classification.py exports
- assign_risk_labels(df: pd.DataFrame) -> pd.DataFrame  # adds flood_risk column
- train_risk_classifier(X_train, y_train) -> model
- evaluate_classification(y_true, y_pred, labels: list) -> dict  # precision, recall, f1, pr_auc, confusion_matrix, report
- get_classification_report_df(y_true, y_pred, labels: list) -> pd.DataFrame

### src/geospatial.py exports
- create_risk_map(gdf: gpd.GeoDataFrame) -> folium.Map
- load_geojson(filepath: str) -> gpd.GeoDataFrame
- csv_to_geodataframe(df: pd.DataFrame, lat_col, lon_col) -> gpd.GeoDataFrame
- get_risk_color(risk: str) -> str

### app.py (Overview page)
- KPI cards: total stations, avg risk score, model accuracy, latest prediction
- Summary charts
- Navigation links

### Pages
Each page imports from src/ modules. No business logic in pages.

