"""
rebuild_dataset_v6.py - Enhanced Data Preprocessing with 10 New Features

Key improvements over V4:
1. Adds 10 new features for real-world flood conditions
2. Uses Open-Meteo Archive API for historical data (2020-2025)
3. Total features: 35 (25 existing + 10 new)
4. Maintains compatibility with existing architecture
"""

import pandas as pd
import numpy as np
import os
import sys
import requests
from datetime import datetime, timedelta

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_historical_weather(start_date, end_date):
    """
    Fetch historical hourly weather data from Open-Meteo Archive API.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["precipitation", "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "wind_speed_10m"],
        "timezone": config.TIMEZONE
    }
    
    try:
        logger.info(f"Fetching historical data from {start_date} to {end_date}...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        hourly = data['hourly']
        df = pd.DataFrame({
            'date': pd.to_datetime(hourly['time']),
            'precipitation': hourly['precipitation'],
            'soil_moisture_surface': hourly['soil_moisture_0_to_1cm'],
            'soil_moisture_root': hourly['soil_moisture_1_to_3cm'],
            'wind_speed': hourly['wind_speed_10m']
        })
        
        logger.info(f"Fetched {len(df)} hourly records")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching historical weather: {e}")
        return pd.DataFrame()


def fetch_upstream_rain(start_date, end_date):
    """
    Fetch historical rain from upstream location (Badak Baru).
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Get upstream coordinates from config
    upstream_coord = config.UPSTREAM_LOCATIONS.get("Hulu Karang Mumus (Badak Baru)")
    if not upstream_coord:
        logger.warning("Upstream location not found. Skipping upstream rain.")
        return pd.DataFrame()
    
    lat, lon = upstream_coord
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["precipitation"],
        "timezone": config.TIMEZONE
    }
    
    try:
        logger.info(f"Fetching upstream rain data...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        hourly = data['hourly']
        df = pd.DataFrame({
            'date': pd.to_datetime(hourly['time']),
            'upstream_precipitation': hourly['precipitation']
        })
        
        logger.info(f"Fetched {len(df)} upstream rain records")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching upstream rain: {e}")
        return pd.DataFrame()


def add_new_features_v6(df_daily: pd.DataFrame, df_hourly: pd.DataFrame, df_upstream: pd.DataFrame) -> pd.DataFrame:
    """
    Add 10 new features for V6 model.
    
    Args:
        df_daily: Daily aggregated dataset
        df_hourly: Hourly weather data
        df_upstream: Upstream hourly rain data
    """
    logger.info("Creating 10 new features for V6...")
    
    # Merge hourly data for feature calculation
    df_daily['tanggal'] = pd.to_datetime(df_daily['tanggal'])
    df_daily['date_only'] = df_daily['tanggal'].dt.date
    
    # Group hourly by date for aggregations
    df_hourly['date_only'] = pd.to_datetime(df_hourly['date']).dt.date
    
    # --- NEW FEATURE 1: rain_intensity_3h ---
    # Max 3-hour rolling sum of precipitation
    df_hourly['rain_3h'] = df_hourly['precipitation'].rolling(window=3, min_periods=1).sum()
    rain_intensity_3h = df_hourly.groupby('date_only')['rain_3h'].max().reset_index()
    rain_intensity_3h.columns = ['date_only', 'rain_intensity_3h']
    df_daily = df_daily.merge(rain_intensity_3h, on='date_only', how='left')
    
    # --- NEW FEATURE 2: rain_burst_count ---
    # Count of hourly precipitation > 10mm in a day
    df_hourly['is_burst'] = (df_hourly['precipitation'] > 10).astype(int)
    rain_burst_count = df_hourly.groupby('date_only')['is_burst'].sum().reset_index()
    rain_burst_count.columns = ['date_only', 'rain_burst_count']
    df_daily = df_daily.merge(rain_burst_count, on='date_only', how='left')
    
    # --- NEW FEATURE 3: soil_saturation_trend ---
    # Change in soil saturation over 3 days
    df_daily['soil_saturation_trend'] = df_daily['soil_saturation_index'].diff(periods=3).fillna(0)
    
    # --- NEW FEATURE 4: tide_rain_sync ---
    # Flag: high tide (>2.5m) AND heavy rain (>50mm) on same day
    df_daily['tide_rain_sync'] = ((df_daily['pasut_msl_max'] > 2.5) & 
                                   (df_daily['rain_sum_imputed'] > 50)).astype(int)
    
    # --- NEW FEATURE 5: consecutive_rain_days ---
    # Number of consecutive days with rain (>0mm)
    df_daily['has_rain'] = (df_daily['rain_sum_imputed'] > 0).astype(int)
    df_daily['consecutive_rain_days'] = df_daily['has_rain'].groupby(
        (df_daily['has_rain'] != df_daily['has_rain'].shift()).cumsum()
    ).cumsum()
    df_daily['consecutive_rain_days'] = df_daily['consecutive_rain_days'] * df_daily['has_rain']
    
    # --- NEW FEATURE 6: hour_risk_factor ---
    # Risk factor based on hour of day (peak flooding hours: 22:00-06:00)
    # For daily data, use a proxy: if any heavy rain occurred at night
    df_hourly['hour'] = pd.to_datetime(df_hourly['date']).dt.hour
    df_hourly['is_night'] = df_hourly['hour'].apply(lambda h: 1 if h >= 22 or h <= 6 else 0)
    df_hourly['night_rain'] = df_hourly['is_night'] * df_hourly['precipitation']
    night_rain_total = df_hourly.groupby('date_only')['night_rain'].sum().reset_index()
    night_rain_total.columns = ['date_only', 'night_rain_total']
    df_daily = df_daily.merge(night_rain_total, on='date_only', how='left')
    df_daily['hour_risk_factor'] = 1.0 + (df_daily['night_rain_total'] / (df_daily['rain_sum_imputed'] + 1e-5)) * 0.2
    df_daily['hour_risk_factor'] = df_daily['hour_risk_factor'].clip(0.8, 1.2)
    
    # --- NEW FEATURE 7: drain_capacity_index ---
    # Proxy for drainage saturation: rain_cumsum_7d / 200mm capacity
    df_daily['drain_capacity_index'] = df_daily['rain_cumsum_7d'] / 200.0
    df_daily['drain_capacity_index'] = df_daily['drain_capacity_index'].clip(0, 2.0)
    
    # --- NEW FEATURE 8: upstream_rain_6h ---
    # Upstream rain 6 hours before (shifted by 6 hours)
    if not df_upstream.empty:
        df_upstream['date_only'] = pd.to_datetime(df_upstream['date']).dt.date
        # Shift upstream by 6 hours to account for travel time
        df_upstream['date_shifted'] = pd.to_datetime(df_upstream['date']) + timedelta(hours=config.UPSTREAM_LAG_HOURS)
        df_upstream['date_only_shifted'] = df_upstream['date_shifted'].dt.date
        
        # Aggregate upstream rain by shifted date
        upstream_daily = df_upstream.groupby('date_only_shifted')['upstream_precipitation'].sum().reset_index()
        upstream_daily.columns = ['date_only', 'upstream_rain_6h']
        df_daily = df_daily.merge(upstream_daily, on='date_only', how='left')
    else:
        df_daily['upstream_rain_6h'] = 0
    
    # --- NEW FEATURE 9: wind_speed_max ---
    # Max wind speed in the day (storm indicator)
    wind_max = df_hourly.groupby('date_only')['wind_speed'].max().reset_index()
    wind_max.columns = ['date_only', 'wind_speed_max']
    df_daily = df_daily.merge(wind_max, on='date_only', how='left')
    
    # --- NEW FEATURE 10: rainfall_acceleration ---
    # Change in rain intensity (today - yesterday)
    df_daily['rainfall_acceleration'] = df_daily['rain_intensity_max'].diff().fillna(0)
    
    # Fill NaN values for new features
    new_features = [
        'rain_intensity_3h', 'rain_burst_count', 'soil_saturation_trend',
        'tide_rain_sync', 'consecutive_rain_days', 'hour_risk_factor',
        'drain_capacity_index', 'upstream_rain_6h', 'wind_speed_max',
        'rainfall_acceleration'
    ]
    
    for feat in new_features:
        if feat in df_daily.columns:
            df_daily[feat] = df_daily[feat].fillna(0)
    
    # Drop temporary columns
    df_daily = df_daily.drop(['date_only', 'has_rain', 'night_rain_total'], axis=1, errors='ignore')
    
    logger.info(f"Added {len(new_features)} new features")
    return df_daily


def rebuild_dataset_v6():
    """
    Main function to rebuild dataset V6 with enhanced features.
    """
    
    # 1. Load existing V4 dataset as base
    data_path = os.path.join(config.BASE_DIR, "data", "dataset_banjir_v4_processed.csv")
    logger.info(f"Loading base data from {data_path}...")
    df_base = pd.read_csv(data_path)
    
    logger.info(f"Base dataset shape: {df_base.shape}")
    logger.info(f"Date range: {df_base['tanggal'].min()} to {df_base['tanggal'].max()}")
    
    # 2. Fetch historical hourly weather data
    start_date = "2020-12-01"
    end_date = "2025-12-31"
    
    df_hourly = fetch_historical_weather(start_date, end_date)
    df_upstream = fetch_upstream_rain(start_date, end_date)
    
    if df_hourly.empty:
        logger.error("Failed to fetch hourly weather data. Cannot proceed.")
        return None
    
    # 3. Add new V6 features
    df_final = add_new_features_v6(df_base, df_hourly, df_upstream)
    
    # 4. Define final feature set (35 features)
    features_v6 = [
        # Original V4 features (25)
        'rain_sum_imputed', 'rain_intensity_max',
        'soil_moisture_surface_mean', 'soil_moisture_root_mean', 'soil_saturation_index',
        'pasut_msl_max',
        'rain_lag1', 'rain_lag2', 'rain_lag3', 'rain_lag4', 'rain_lag5', 'rain_lag6', 'rain_lag7',
        'rain_cumsum_3d', 'rain_cumsum_7d',
        'tide_rain_interaction', 'is_high_tide', 'is_heavy_rain',
        'api_7day',
        'month_sin', 'month_cos', 'is_rainy_season', 'is_weekend',
        'prev_flood_30d', 'prev_meluap_30d',
        
        # New V6 features (10)
        'rain_intensity_3h', 'rain_burst_count', 'soil_saturation_trend',
        'tide_rain_sync', 'consecutive_rain_days', 'hour_risk_factor',
        'drain_capacity_index', 'upstream_rain_6h', 'wind_speed_max',
        'rainfall_acceleration'
    ]
    
    # Ensure all features exist
    for feat in features_v6:
        if feat not in df_final.columns:
            logger.warning(f"Feature {feat} not found. Setting to 0.")
            df_final[feat] = 0
    
    # 5. Select final columns
    final_cols = ['tanggal'] + features_v6 + ['label', 'status_siaga']
    df_output = df_final[final_cols].copy()
    
    # 6. Save processed dataset
    output_path = os.path.join(config.BASE_DIR, "data", "dataset_banjir_v6_enhanced.csv")
    df_output.to_csv(output_path, index=False)
    logger.info(f"Saved V6 dataset to {output_path}")
    
    # 7. Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DATASET V6 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(df_output)}")
    logger.info(f"Total features: {len(features_v6)} (25 old + 10 new)")
    logger.info(f"\nClass distribution:")
    logger.info(f"  Aman (0):       {(df_output['label'] == 0).sum()}")
    logger.info(f"  Air Meluap (1): {(df_output['label'] == 1).sum()}")
    logger.info(f"  Banjir (2):     {(df_output['label'] == 2).sum()}")
    logger.info(f"\nDate range: {df_output['tanggal'].min()} to {df_output['tanggal'].max()}")
    logger.info(f"\nNew features added:")
    for i, feat in enumerate(features_v6[25:], 1):
        logger.info(f"  {i}. {feat}")

    return df_output, features_v6


if __name__ == "__main__":
    rebuild_dataset_v6()
