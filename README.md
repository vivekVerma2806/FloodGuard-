# 🌊 FloodGuard AI – Spatiotemporal Flood Forecasting & Risk Analytics

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-1.x-139C5A?style=for-the-badge)](https://geopandas.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/Tests-36%20Passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)

An end-to-end, production-grade Machine Learning and Spatiotemporal Analytics system for river-level forecasting, flood risk classification, model benchmarking, and interactive geospatial mapping.

Built for hydrological monitoring and disaster resilience analysis, **FloodGuard AI** integrates deep recurrent neural networks (LSTM), tree-based ensemble learning, rigorous chronological cross-validation, and interactive WebGIS mapping into a responsive Streamlit dashboard.

---

## 📌 Project Architecture

```
FloodGuard-/
├── app.py                          # Streamlit application entry point (Overview & KPIs)
├── config.py                       # Centralized system constants, paths, and thresholds
├── requirements.txt                # Production dependency specifications
├── .gitignore                      # Git exclusion rules (virtualenvs, models, caches)
├── ARCHITECTURE.md                 # System architecture contracts & specifications
├── README.md                       # Comprehensive project documentation
├── data/
│   ├── demo_dataset.csv            # 5-year daily multi-station hydrological dataset
│   └── demo_geospatial.geojson     # Geospatial GeoJSON vector layer of monitoring stations
├── models/
│   └── .gitkeep                    # Target directory for serialized models
├── pages/
│   ├── 1_📊_Data_Upload.py          # CSV upload, missing value imputation, duplicate removal
│   ├── 2_🌊_Flood_Forecasting.py    # Multivariate LSTM forecasting & loss curves
│   ├── 3_📈_Model_Benchmarking.py   # Sklearn benchmarking (LR, RF, GBR) vs LSTM
│   ├── 4_⚠️_Risk_Classification.py   # Multi-class flood risk pipeline (PR-AUC, F1, Confusion Matrix)
│   ├── 5_🗺️_Geospatial_Map.py       # Interactive Folium map with color-coded risk markers
│   └── 6_📋_Model_Evaluation.py     # Unified performance audit & actionable recommendations
├── scripts/
│   └── generate_demo_data.py       # Synthetic data generation engine with physical river dynamics
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Cleaning, rolling-window features, normalization, sequences
│   ├── lstm_model.py               # Deep LSTM architecture (Keras/TensorFlow) with EarlyStopping
│   ├── ml_benchmarking.py          # Benchmark regression pipelines & feature importance
│   ├── risk_classification.py      # Random Forest classifier, PR-AUC, confusion matrix
│   └── geospatial.py               # GeoPandas vector processing & Folium WebGIS visualization
└── tests/
    ├── __init__.py
    ├── test_data_preprocessing.py  # Unit tests for preprocessing & sequence creation
    ├── test_lstm_model.py          # Unit tests for LSTM architecture & model persistence
    ├── test_ml_benchmarking.py     # Unit tests for baseline regressors & metrics
    ├── test_risk_classification.py # Unit tests for risk labeling & classification metrics
    └── test_geospatial.py          # Unit tests for vector transformation & map creation
```

---

## 🚀 Key Features

### 1. Hydrological Data Preprocessing & Pipeline
- **Missing Value Handling & Imputation**: Group-based forward and backward filling per station to preserve physical continuity.
- **Deduplication**: Automatic detection and removal of repeated temporal records.
- **Feature Engineering**:
  - `prev_day_rainfall`: 1-day shifted antecedent precipitation.
  - `rainfall_3d_avg`: 3-day rolling cumulative precipitation moving average.
  - `level_change`: Daily derivative of water level ($\Delta h = h_t - h_{t-1}$).
- **Data Leakage Prevention**: Strict chronological partitioning (`chronological_split`) ensures validation and test data strictly succeed training timelines without lookahead bias.
- **Sliding-Window Sequence Generation**: Sliding time-window matrices for recurrent neural networks ($N, T, D$).

### 2. Deep Multivariate LSTM Flood Forecasting
- **Architecture**:
  - `Input(shape=(seq_len, n_features))`
  - `LSTM(64 units, return_sequences=True)`
  - `Dropout(0.20)`
  - `LSTM(32 units)`
  - `Dropout(0.20)`
  - `Dense(1)`
- **Optimization**: Adam optimizer with Mean Squared Error (MSE) loss and Mean Absolute Error (MAE) metrics.
- **Early Stopping**: Regularized training with `val_loss` patience=10 and `restore_best_weights=True`.
- **Model Serialization**: Native `.keras` format model saving and on-demand reloading.

### 3. Machine Learning Model Benchmarking
Compare traditional regression algorithms against recurrent neural architectures:
- **Linear Regression** (Ordinary Least Squares baseline)
- **Random Forest Regressor** (100 estimators, bootstrap ensemble)
- **Gradient Boosting Regressor** (Sequential residual boosting)
- **Evaluation Metrics**: Exact Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Interpretability**: Gini feature importance extraction across meteorological predictors.

### 4. Multi-Class Flood Risk Classification
Classifies hydrological states into four operational threat tiers:
| Tier | River Water Level ($h$) | Visual Color | Severity |
| :--- | :--- | :--- | :--- |
| **Low** | $h < 3.0\,\text{m}$ | 🟢 Green | Normal river stage |
| **Moderate** | $3.0\,\text{m} \le h < 5.0\,\text{m}$ | 🟠 Orange | Advisory stage |
| **High** | $5.0\,\text{m} \le h < 7.0\,\text{m}$ | 🔴 Red | Flood warning stage |
| **Critical** | $h \ge 7.0\,\text{m}$ | 🛑 Dark Red | Severe inundation hazard |

- **Metrics Computed**: Precision (Macro), Recall (Macro), F1-Score (Macro), and Precision-Recall AUC (PR-AUC per class and macro-average).
- **Diagnostics**: Heatmap Confusion Matrix and tabular Scikit-Learn Classification Reports.

### 5. Interactive Geospatial Risk Mapping (WebGIS)
- Powered by **GeoPandas**, **Shapely**, and **Folium**.
- Visualizes station coordinates with dynamic `CircleMarker` icons styled by severity tier.
- Interactive popups display Station Name, Water Level ($m$), Rainfall ($mm$), and Warning Tier.
- Dual ingestion support: Point geometry from CSV Coordinates or native vector **GeoJSON** layers.
- Embedded map controls, risk legend, and responsive layer toggles.

---

## 🛠️ Tech Stack & Dependencies

| Category | Tools & Libraries |
| :--- | :--- |
| **Programming Language** | Python 3.11+ |
| **Deep Learning** | TensorFlow 2.x, Keras 3.x |
| **Machine Learning** | Scikit-Learn |
| **Data Manipulation** | Pandas, NumPy |
| **Geospatial & Vector** | GeoPandas, Shapely, PyProj |
| **Dashboard & UI** | Streamlit |
| **Data Visualization** | Plotly, Folium |
| **Testing & Quality** | Pytest, Flake8 |

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/vivekVerma2806/FloodGuard-.git
cd FloodGuard-
```

### 2. Set Up a Virtual Environment
Using Python 3.11:
```bash
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

*(Alternatively, if using [uv](https://github.com/astral-sh/uv)):*
```bash
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate the Demo Dataset (Included by default)
```bash
python scripts/generate_demo_data.py
```
This generates 5 years (1,826 days $\times$ 5 stations = 9,130 records + injected duplicates and missing values) of realistic seasonal hydrological observations in `data/demo_dataset.csv`.

---

## 🖥️ Running the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

The application will start and open automatically in your default browser at:
```
http://localhost:8501
```

### Dashboard Walkthrough:
1. **Overview**: Key metrics, 5-year precipitation trends, and river level distributions across river basins.
2. **1. Data Upload**: Ingest custom CSV datasets or load the demo dataset; trigger cleaning and feature engineering pipelines.
3. **2. Flood Forecasting**: Train the multivariate LSTM model with real-time loss tracking and actual vs. predicted curves.
4. **3. Model Benchmarking**: Train and compare Linear Regression, Random Forest, and Gradient Boosting; view feature importance.
5. **4. Risk Classification**: Train multi-class Random Forest risk classifier, inspect PR-AUC and confusion matrices.
6. **5. Geospatial Map**: View interactive Folium map with threat-coded stations, popups, and layer toggles.
7. **6. Model Evaluation**: Consolidated performance scorecards, automated diagnostics, and downloadable text reports.

---

## 🧪 Automated Testing

The project includes a comprehensive test suite covering all modules:
```bash
pytest tests/ -v
```

### Test Suite Summary:
```
tests/test_data_preprocessing.py    ...... [ 19%]
tests/test_geospatial.py            ....... [ 38%]
tests/test_lstm_model.py            ...... [ 55%]
tests/test_ml_benchmarking.py       ...... [ 72%]
tests/test_pipeline_integration.py  ...... [ 88%]
tests/test_risk_classification.py   ....   [100%]

====================== 36 passed in 26.67s ======================
```
- Preprocessing validation (missing value imputation, chronological integrity, sliding-window geometry).
- LSTM lifecycle testing (construction, gradient descent execution, prediction shapes, serialization).
- Benchmark validation (error calculations, feature importance extraction).
- Classification metric validation (PR-AUC, confusion matrix shape integrity).
- Geospatial mapping tests (geometry creation, projection validation, color assignment).

---

## ⚠️ Notes & Disclaimer
- **Demonstration Data**: The included dataset and geographical coordinates reflect synthetic hydrological patterns modeled after major Indian river basins (Godavari, Brahmaputra, Ganga, Krishna, Mahanadi) for technical evaluation and interview demonstrations.
- **Operational Use**: While the algorithms (LSTM, RF, GBR) and mathematical pipelines represent real machine learning implementations, predictions from synthetic data must not be deployed for live disaster safety operations without calibration against official sensor telemetries (e.g., Central Water Commission / IMD).

---

## 👤 Author
**Vivek Verma**  
- GitHub: [@vivekVerma2806](https://github.com/vivekVerma2806)
- Repository: [FloodGuard-](https://github.com/vivekVerma2806/FloodGuard-)
