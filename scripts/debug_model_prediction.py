
import sys
import os
import pandas as pd
import numpy as np

# Add parent dir to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_utils
import data_ingestion
import config
from feature_extraction import SpatialFeatureExtractor

def debug_prediction():
    print("🔍 DIAGNOSTIC: Flood Prediction Model Evaluation")
    
    # 1. Load Resources
    print("\n[1/5] Loading Resources...")
    model_pack = model_utils.load_model()
    if not model_pack:
        print("❌ Model not found!")
        return

    # 2. Fetch Real Data
    print("\n[2/5] Fetching Real-time Data...")
    wf = data_ingestion.WeatherFetcher()
    tide_pred = data_ingestion.TidePredictor()
    
    # Fetch for current location
    lat, lon = config.LATITUDE, config.LONGITUDE
    weather_df = wf.fetch_weather_data(lat, lon)
    
    if weather_df.empty:
        print("❌ Failed to fetch weather data.")
        return

    # Get latest data point (now)
    now = pd.Timestamp.now(tz=config.TIMEZONE)
    # Find closest row
    current_row = weather_df.iloc[(weather_df['date'] - now).abs().argsort()[:1]]
    
    # Predict Tide
    current_tide = tide_pred.predict_hourly([now])[0]
    
    print(f"   Current Time: {now}")
    print(f"   Rain (24h Rolling): {current_row['rain_rolling_24h'].values[0]:.2f} mm")
    print(f"   Rain (Intensity): {current_row['precipitation'].values[0]:.2f} mm/h")
    print(f"   Tide Level: {current_tide:.2f} m")
    
    # 3. Spatial Features
    print("\n[3/5] Spatial Features...")
    try:
        extractor = SpatialFeatureExtractor()
        feats = extractor.get_features(lat, lon)
        print(f"   Flow Accumulation: {feats.get('flow_accumulation', 0)}")
    except Exception as e:
        print(f"   Spatial Error: {e}")
        feats = {}

    # 4. Construct Input
    print("\n[4/5] Constructing Input for Model...")
    # Mock Upstream for now or fetch
    upstream_rain = 0 
    
    input_data = {
        "rain_sum_imputed": current_row['rain_rolling_24h'].values[0],
        "rain_intensity_max": current_row['precipitation'].values[0],
        "soil_moisture_surface_mean": current_row.get('soil_moisture_surface', 0.5).values[0],
        "soil_moisture_root_mean": current_row.get('soil_moisture_root', 0.5).values[0],
        "pasut_msl_max": current_tide,
        "hujan_lag1": 5.0, # Mock low value
        "upstream_rain": upstream_rain,
        "flow_accumulation": feats.get('flow_accumulation', 0),
        "runoff_coefficient": 0.85
    }
    
    print("   Input Dict:", input_data)

    # 5. Prediction Breakout
    print("\n[5/5] Prediction Analysis...")
    
    # Access logic inside predict_flood manually to see components
    # (Since I can't easily modify the imported function to print without editing it,
    #  I will replicate the logic for display here)
    
    # A. Base Model
    # model = model_pack['model']
    # base_prob = model.predict_proba(df_input) ... (This is tricky to replicate exactly without DF construction)
    
    # Just call the function and inspect the result
    result = model_utils.predict_flood(model_pack, input_data)
    
    print("\n📊 FINAL RESULT:")
    print(f"   Depth: {result['depth_cm']:.2f} cm")
    print(f"   Level: {result['level']}")
    print(f"   Reasoning: {result['reasoning']}")
    
    # Heuristic Check
    # (Since we are now using Depth, not Probability, direct comparison is harder, 
    # but we can print the raw features for manual inspection)
    
    # Reverse Engineer the Heuristics based on logic in model_utils.py
    # Reverse Engineer the Heuristics (Simplified for Depth)
    # The V8 Model logic in model_utils.py doesn't apply explicit heuristic boosts to depth 
    # the same way it did for probability. The model itself (Random Forest) learns these patterns.
    # However, we can highlight risk factors present in the input.
    
    print("\n🔍 RISK FACTOR ANALYSIS:")
    if input_data.get('pasut_msl_max', 0) > 2.5:
        print("   🌊 High Tide Detected (>2.5m)")
        
    if input_data.get('rain_sum_imputed', 0) > 50:
        print("   🌧️ Heavy Rain Detected (>50mm)")
        
    if input_data.get('upstream_rain', 0) > 20:
        print("   ⛰️ Upstream Rain Detected")
        
    print(f"\n✅ Diagnostic Complete. Model V8 Output: {result['depth_cm']:.2f} cm")

if __name__ == "__main__":
    debug_prediction()
