"""Generate realistic synthetic flood monitoring dataset for FloodGuard AI demo."""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DEMO_STATIONS, RANDOM_SEED

np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Station metadata
# ---------------------------------------------------------------------------
STATION_FACTORS = {
    'STN001': 0.90,
    'STN002': 1.00,
    'STN003': 1.10,
    'STN004': 0.95,
    'STN005': 1.05,
}

# ---------------------------------------------------------------------------
# Seasonal parameters: (rainfall_mean, rainfall_std, rainfall_clip,
#                        level_mean, level_std, level_clip,
#                        temp_mean, temp_std, humidity_base)
# ---------------------------------------------------------------------------
SEASON_PARAMS = {
    'monsoon': {
        'rainfall': (45, 25, (0, 200)),
        'level':    (5.5, 2.0, (1, 12)),
        'temp':     (27, 3),       # slightly cooler
        'hum_base': 75,
    },
    'post_monsoon': {
        'rainfall': (15, 10, (0, 80)),
        'level':    (3.5, 1.0, (0.5, 8)),
        'temp':     (28, 4),
        'hum_base': 60,
    },
    'winter': {
        'rainfall': (5, 5, (0, 30)),
        'level':    (2.0, 0.5, (0.5, 5)),
        'temp':     (22, 3),
        'hum_base': 50,
    },
    'pre_monsoon': {
        'rainfall': (10, 8, (0, 50)),
        'level':    (2.5, 0.8, (0.5, 6)),
        'temp':     (35, 3),
        'hum_base': 45,
    },
}


def get_season(month: int) -> str:
    if month in (6, 7, 8, 9):
        return 'monsoon'
    elif month in (10, 11):
        return 'post_monsoon'
    elif month in (12, 1, 2):
        return 'winter'
    else:
        return 'pre_monsoon'


def generate_station_data(station: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate daily data for one station across all dates."""
    factor = STATION_FACTORS[station['id']]
    n = len(dates)

    records = []
    for i, dt in enumerate(dates):
        season = get_season(dt.month)
        sp = SEASON_PARAMS[season]

        rain_mean, rain_std, rain_clip = sp['rainfall']
        lvl_mean, lvl_std, lvl_clip = sp['level']
        temp_mean, temp_std = sp['temp']
        hum_base = sp['hum_base']

        rainfall = np.clip(np.random.normal(rain_mean, rain_std), *rain_clip) * factor
        level = np.clip(np.random.normal(lvl_mean, lvl_std), *lvl_clip) * factor
        temp = np.clip(np.random.normal(temp_mean, temp_std), 20, 40)
        # Humidity correlated with rainfall
        humidity = np.clip(hum_base + rainfall * 0.3 + np.random.normal(0, 5), 40, 100)

        records.append({
            'date': dt.strftime('%Y-%m-%d'),
            'station_id': station['id'],
            'station_name': station['name'],
            'latitude': station['lat'],
            'longitude': station['lon'],
            'rainfall_mm': round(rainfall, 2),
            'river_level_m': round(level, 2),
            'temperature_c': round(temp, 2),
            'humidity_pct': round(humidity, 2),
        })

    return pd.DataFrame(records)


def main():
    dates = pd.date_range('2019-01-01', '2023-12-31', freq='D')
    print(f"Generating data for {len(dates)} days × {len(DEMO_STATIONS)} stations …")

    frames = []
    for stn in DEMO_STATIONS:
        df_stn = generate_station_data(stn, dates)
        frames.append(df_stn)

    df = pd.concat(frames, ignore_index=True)
    print(f"Raw rows: {len(df)}")

    # --- Inject ~2 % missing values (NaN) in numeric columns ---
    numeric_cols = ['rainfall_mm', 'river_level_m', 'temperature_c', 'humidity_pct']
    n_total_cells = len(df) * len(numeric_cols)
    n_missing = int(n_total_cells * 0.02)
    rng = np.random.default_rng(RANDOM_SEED)
    missing_rows = rng.choice(len(df), size=n_missing, replace=True)
    missing_cols = rng.choice(numeric_cols, size=n_missing)
    for r, c in zip(missing_rows, missing_cols):
        df.at[r, c] = np.nan
    print(f"Injected {n_missing} NaN cells (~2 %)")

    # --- Add ~0.5 % duplicate rows ---
    n_dups = int(len(df) * 0.005)
    dup_idx = rng.choice(len(df), size=n_dups, replace=False)
    dups = df.iloc[dup_idx].copy()
    df = pd.concat([df, dups], ignore_index=True)
    print(f"Added {n_dups} duplicate rows (~0.5 %)")

    # Shuffle lightly so duplicates aren't all at the end
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'demo_dataset.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {os.path.abspath(out_path)}")
    print(df.head(5).to_string())


if __name__ == '__main__':
    main()
