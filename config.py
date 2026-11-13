"""FloodGuard AI - Global Configuration Constants."""

import os

# Reproducibility
RANDOM_SEED = 42

# Data
SEQUENCE_LENGTH = 10
TEST_SPLIT_RATIO = 0.2
VALIDATION_SPLIT_RATIO = 0.1

# Feature columns used for modeling
FEATURES = [
    'rainfall_mm', 'river_level_m', 'temperature_c', 'humidity_pct',
    'prev_day_rainfall', 'rainfall_3d_avg', 'level_change'
]
TARGET_REGRESSION = 'river_level_m'
TARGET_CLASSIFICATION = 'flood_risk'

# Risk categories and thresholds (river_level_m based)
RISK_CATEGORIES = ['Low', 'Moderate', 'High', 'Critical']
RISK_THRESHOLDS = {
    'Low': (0, 3.0),
    'Moderate': (3.0, 5.0),
    'High': (5.0, 7.0),
    'Critical': (7.0, float('inf'))
}

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# LSTM hyperparameters
LSTM_UNITS = 64
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# Demo stations (Indian river basins)
DEMO_STATIONS = [
    {'id': 'STN001', 'name': 'Godavari Basin - Nashik', 'lat': 19.9975, 'lon': 73.7898},
    {'id': 'STN002', 'name': 'Brahmaputra Basin - Guwahati', 'lat': 26.1445, 'lon': 91.7362},
    {'id': 'STN003', 'name': 'Ganga Basin - Varanasi', 'lat': 25.3176, 'lon': 82.9739},
    {'id': 'STN004', 'name': 'Krishna Basin - Vijayawada', 'lat': 16.5062, 'lon': 80.6480},
    {'id': 'STN005', 'name': 'Mahanadi Basin - Cuttack', 'lat': 20.4625, 'lon': 85.8830},
]

