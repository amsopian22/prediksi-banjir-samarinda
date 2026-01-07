
import sys
import os
import pandas as pd
import numpy as np

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_utils
import data_ingestion
import config
from feature_extraction import SpatialFeatureExtractor
import datetime

def debug_7day_forecast():
    print("🔍 DIAGNOSTIC: 7-Day Forecast Evaluation")
    
    # 1. Load Resources
    print("\n[1/4] Loading Resources...")
    model_pack = model_utils.load_model()
    tide_predictor = data_ingestion.TidePredictor()
    wf = data_ingestion.WeatherFetcher()
    
    # 2. Fetch Weather Forecast
    print("\n[2/4] Fetching Weather Forecast...")
    lat, lon = config.LATITUDE, config.LONGITUDE
    weather_df = wf.fetch_weather_data(lat, lon)
    
    if weather_df.empty:
        print("❌ Weather data empty")
        return
        
    # 3. Process 7-Day Data
    print("\n[3/4] Processing 7-Day Forecast Data...")
    
    # Predict Tide
    weather_df['est'] = tide_predictor.predict_hourly(weather_df['date'])
    weather_df = weather_df.rename(columns={'date': 'time'})
    hourly_df = weather_df
    
    hourly_df['day_date'] = hourly_df['time'].dt.date
    today_date = pd.Timestamp.now(tz=config.TIMEZONE).date()
    
    # Calculate daily sums for lags
    daily_sums = hourly_df.groupby('day_date')['precipitation'].sum()
    
    # Lags Lookup
    lags_lookup = {}
    unique_dates = hourly_df['day_date'].unique()
    for d in unique_dates:
        lags_lookup[d] = {}
        for lag_num in range(1, 8):
            lag_date = d - datetime.timedelta(days=lag_num)
            lags_lookup[d][f'hujan_lag{lag_num}'] = daily_sums.get(lag_date, 0)

    # Filter for future dates
    future_groups = [
        (d, g) for d, g in hourly_df.groupby('day_date') 
        if d >= today_date
    ]
    
    print(f"\nFound {len(future_groups)} days for forecast.")
    
    print("\n[4/4] Analyzing Daily Inputs (Next 7 Days)...")
    print("-" * 100)
    print(f"{'Date':<15} | {'Rain(mm)':<10} | {'Tide(m)':<10} | {'Cumsum7d':<10} | {'Prob(%)':<8} | {'Status':<10} | {'Model Prob':<10}")
    print("-" * 100)

    for date_val, group in future_groups[:7]:
        daily_rain = group['precipitation'].sum()
        daily_max_tide = group['est'].max()
        
        # Prepare Inputs
        lags = lags_lookup.get(date_val, {})
        lag1 = lags.get('hujan_lag1', 0)
        lag2 = lags.get('hujan_lag2', 0)
        lag3 = lags.get('hujan_lag3', 0)
        lag4 = lags.get('hujan_lag4', 0)
        lag5 = lags.get('hujan_lag5', 0)
        lag6 = lags.get('hujan_lag6', 0)
        
        rain_cumsum_7d = daily_rain + lag1 + lag2 + lag3 + lag4 + lag5 + lag6
        
        import datetime as dt
        date_obj = pd.to_datetime(date_val)
        month = date_obj.month
        import math
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)
        
        d_input = {
            "rain_sum_imputed": daily_rain,
            "rain_intensity_max": group['precipitation'].max(),
            "soil_moisture_surface_mean": 0.5,
            "soil_moisture_root_mean": 0.5,
            "pasut_msl_max": daily_max_tide,
            "hujan_lag1": lag1, 
            "hujan_lag2": lag2,
            "hujan_lag3": lag3,
            "hujan_lag4": lag4,
            "hujan_lag5": lag5,
            "hujan_lag6": lag6,
            "hujan_lag7": lags.get('hujan_lag7', 0),
            "is_weekend": 1 if date_obj.dayofweek >= 5 else 0,
            "is_rainy_season": 1,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "rain_cumsum_3d": daily_rain + lag1 + lag2,
            "rain_cumsum_7d": rain_cumsum_7d,
            "upstream_rain_6h": 0,
            "prev_flood_30d": 0,
            "prev_meluap_30d": 0,
            "drain_capacity_index": rain_cumsum_7d / 200.0,
            "tide_rain_sync": 1 if (daily_max_tide > 2.5 and daily_rain > 50) else 0,
            "rain_intensity_3h": daily_rain / 4.0 if daily_rain > 10 else 0,
            "rainfall_acceleration": 0,
            "rain_burst_count": 0,
            "hour_risk_factor": 1.0,
        }
        
        # Test with Offset
        d_input_offset = d_input.copy()
        offset = config.TIDE_DATUM_OFFSET
        d_input_offset["pasut_msl_max"] = max(0, daily_max_tide - offset)
        d_input_offset["tide_rain_interaction"] = d_input_offset["pasut_msl_max"] * daily_rain
        d_input_offset["is_high_tide"] = 1 if d_input_offset["pasut_msl_max"] > (2.5 - offset) else 0 # Logic adaptation
        
        result = model_utils.predict_flood(model_pack, d_input)
        result_offset = model_utils.predict_flood(model_pack, d_input_offset)

        print(f"{str(date_val):<15} | {daily_rain:<10.2f} | {daily_max_tide:<10.2f} | {result['probability']*100:<8.1f} | {result_offset['probability']*100:<8.1f} (Offset)")

    print("-" * 100)
    print("\nCHECK:")
    print("1. Is Tide prediction (~3.8m) reasonable? (Datum mismatch?)")
    print("2. Is Model sensitive to CumSum even with low daily rain?")
    print("3. Check lags used for CumSum calculation.")

if __name__ == "__main__":
    debug_7day_forecast()
