
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import tide_utils

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def rebuild_v8_dataset():
    logger.info("🚀 Building Dataset V8 (10-Year Historical Data)...")
    
    # 1. Load Rain Data
    rain_path = os.path.join(config.BASE_DIR, "data", "rainfall_history_10y.csv")
    if not os.path.exists(rain_path):
        logger.error(f"Rainfall history {rain_path} not found!")
        return
        
    df = pd.read_csv(rain_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 2. Generate Tide Data
    logger.info("🌊 Generating historical tide peaks...")
    tide_model = tide_utils.load_tide_model()
    if not tide_model:
        logger.error("Tide model not found! Train it first.")
        return
        
    # We need to predict tides at high resolution (e.g., hourly) 
    # then take the max for each day to get 'pasut_msl_max'
    full_dates = pd.date_range(start=df['date'].min(), end=df['date'].max() + pd.Timedelta(days=1), freq='h')
    tide_levels = tide_utils.predict_tide(tide_model, full_dates)
    
    tide_df = pd.DataFrame({
        'timestamp': full_dates,
        'tide': tide_levels
    })
    tide_df['date'] = tide_df['timestamp'].dt.date
    daily_tide = tide_df.groupby('date')['tide'].max().reset_index()
    daily_tide.columns = ['date', 'pasut_msl_max']
    daily_tide['date'] = pd.to_datetime(daily_tide['date'])
    
    # 3. Merge Rain and Tide
    df = pd.merge(df, daily_tide, on='date', how='inner')
    
    # 4. Feature Engineering
    logger.info("🛠️ Feature Engineering...")
    # Map column names for compatibility
    df = df.rename(columns={
        'local_precipitation_sum': 'rain_sum_imputed',
        'local_precipitation_hours': 'rain_duration'
    })
    
    # rain_intensity_max (estimated: sum / duration if duration > 0)
    df['rain_intensity_max'] = df.apply(lambda row: row['rain_sum_imputed'] / row['rain_duration'] if row['rain_duration'] > 0 else 0, axis=1)
    
    # Lag Features (1-7 days)
    for i in range(1, 8):
        df[f'rain_lag{i}'] = df['rain_sum_imputed'].shift(i).fillna(0)
        
    # Cumulative Rain
    df['rain_cumsum_3d'] = df['rain_sum_imputed'].rolling(window=3, min_periods=1).sum()
    df['rain_cumsum_7d'] = df['rain_sum_imputed'].rolling(window=7, min_periods=1).sum()
    
    # API 7-day
    k = config.API_DECAY_FACTOR
    df['api_7day'] = df['rain_sum_imputed']
    for i in range(1, 8):
        df['api_7day'] += (k**i) * df[f'rain_lag{i}']
        
    # Soil Moisture Proxy (Since we only have precipitation, we use API as proxy or set default)
    # Note: V7 model needs soil_moisture_surface_mean
    # We'll scale API to a range 0.2 - 0.9 as proxy for soil moisture
    api_max = df['api_7day'].max()
    df['soil_moisture_surface_mean'] = 0.2 + (df['api_7day'] / api_max) * 0.7
    df['soil_moisture_root_mean'] = df['soil_moisture_surface_mean'] # Simplified
    df['soil_saturation_index'] = df['soil_moisture_surface_mean']
    
    # Interaction & Seasonality
    df['tide_rain_interaction'] = df['pasut_msl_max'] * df['rain_sum_imputed']
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['is_rainy_season'] = df['month'].isin([11, 12, 1, 2, 3, 4]).astype(int)
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    df['is_high_tide'] = (df['pasut_msl_max'] > 2.5).astype(int)
    df['is_heavy_rain'] = (df['rain_sum_imputed'] > 50).astype(int)
    
    # Upstream Rain (Hulu)
    df['upstream_rain_6h'] = df['upstream_precipitation_sum'].fillna(0)
    
    # 5. Labeling
    logger.info("🏷️ Labeling data (Heuristic + Known Floods)...")
    df['label'] = 0 # Default: Aman
    df['status_siaga'] = 'Aman'
    
    # --- HEURISTIC 1: Air Meluap (Label 1) ---
    # Rain > 40mm or (Rain > 25mm and Tide > 2.8m)
    mask_meluap = (df['rain_sum_imputed'] > 40) | ((df['rain_sum_imputed'] > 25) & (df['pasut_msl_max'] > 2.8))
    df.loc[mask_meluap, 'label'] = 1
    df.loc[mask_meluap, 'status_siaga'] = 'Air Meluap'
    
    # --- HEURISTIC 2: Banjir (Label 2) ---
    # Extreme cases: Rain > 100mm or (Rain > 70mm and Tide > 3.0m)
    mask_banjir = (df['rain_sum_imputed'] > 100) | ((df['rain_sum_imputed'] > 70) & (df['pasut_msl_max'] > 3.0))
    df.loc[mask_banjir, 'label'] = 2
    df.loc[mask_banjir, 'status_siaga'] = 'Banjir'
    
    # --- HARD-CODED KNOWN FLOOD DATES (BPBD/Media) ---
    known_floods = [
         # 2017: April 4-7
         '2017-04-04', '2017-04-05', '2017-04-06', '2017-04-07',
         # 2018: March (Approx mid-month based on news)
         '2018-03-15', '2018-03-16', '2018-03-17',
         # 2019: June 4-22 (Lebaran Flood)
         '2019-06-04', '2019-06-05', '2019-06-06', '2019-06-07', '2019-06-08', '2019-06-09',
         '2019-06-10', '2019-06-11', '2019-06-12', '2019-06-13', '2019-06-14',
         # 2019: Dec 23
         '2019-12-23',
         # 2020: Jan 11-16
         '2020-01-11', '2020-01-12', '2020-01-13', '2020-01-14', '2020-01-15', '2020-01-16',
         # 2020: May 22 (Idul Fitri Flood)
         '2020-05-22', '2020-05-23', '2020-05-24',
         # 2020: Dec 1
         '2020-12-01'
    ]
    for d_str in known_floods:
        target_date = pd.to_datetime(d_str)
        df.loc[df['date'] == target_date, 'label'] = 2
        df.loc[df['date'] == target_date, 'status_siaga'] = 'Banjir'
        
    # 6. Merge with existing labeled data (Overwrite for better quality)
    labeled_path = os.path.join(config.BASE_DIR, "data", "dataset_banjir_samarinda_final.csv")
    if os.path.exists(labeled_path):
        logger.info(f"Merging with verified labels from {labeled_path}...")
        df_ver = pd.read_csv(labeled_path)
        df_ver['tanggal'] = pd.to_datetime(df_ver['tanggal'])
        
        # We only want dates and labels from the verified set
        ver_map = df_ver.set_index('tanggal')['status_siaga'].to_dict()
        
        # Overwrite
        for d, s in ver_map.items():
            if d in df['date'].values:
                df.loc[df['date'] == d, 'status_siaga'] = s
                # Sync numeric label
                l_map = {'Aman': 0, 'Air Meluap': 1, 'Banjir': 2, 'Siaga': 2, 'Waspada': 1}
                df.loc[df['date'] == d, 'label'] = l_map.get(s, 0)

    # 7. Final Polish
    # Add dummy columns for V7 compatibility if needed
    df['prev_flood_30d'] = df['label'].apply(lambda x: 1 if x == 2 else 0).rolling(window=30, min_periods=1).sum()
    df['prev_meluap_30d'] = df['label'].apply(lambda x: 1 if x >= 1 else 0).rolling(window=30, min_periods=1).sum()
    df['drain_capacity_index'] = df['rain_cumsum_7d'] / 200.0
    
    # Reorder and rename for final save
    df = df.rename(columns={'date': 'tanggal'})
    
    output_path = os.path.join(config.BASE_DIR, "data", "dataset_banjir_v8_10years.csv")
    df.to_csv(output_path, index=False)
    
    logger.info("=" * 60)
    logger.info("DATASET V8 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Class distribution:\n{df['status_siaga'].value_counts()}")
    logger.info(f"Output saved to: {output_path}")

if __name__ == "__main__":
    rebuild_v8_dataset()
