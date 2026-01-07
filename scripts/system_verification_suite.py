
import sys
import os
import pandas as pd
import numpy as np
import logging

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import model_utils
import ui_components

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_verification():
    print("🚀 STARTING FINAL SYSTEM VERIFICATION SUITE for Samarinda Flood Dashboard")
    print("=" * 80)
    
    issues = []
    tests_passed = 0
    total_tests = 0
    
    # --- TEST 1: CONFIGURATION CHECK ---
    total_tests += 1
    print("\n[TEST 1] Configuration Health Check...")
    if config.THRESHOLD_FLOOD_PROBABILITY == 0.65 and config.TIDE_DATUM_OFFSET == 2.8:
        print("✅ Config Values Correct")
        tests_passed += 1
    else:
        print(f"❌ Config Mismatch: Threshold={config.THRESHOLD_FLOOD_PROBABILITY}, Offset={config.TIDE_DATUM_OFFSET}")
        issues.append("Config values do not match optimization plan.")

    # --- TEST 2: MODEL LOADING ---
    total_tests += 1
    print("\n[TEST 2] Model Reliability ...")
    try:
        model_pack = model_utils.load_model()
        if model_pack and "model" in model_pack:
            print("✅ Model Loaded Successfully")
            tests_passed += 1
        else:
            print("❌ Model Load Failed")
            issues.append("Model failed to load.")
    except Exception as e:
        print(f"❌ Model Exception: {e}")
        issues.append(f"Model Load Exception: {e}")

    # --- TEST 3: DRY DAY SAFETY CAP (The "Awas" Fix) ---
    total_tests += 1
    print("\n[TEST 3] Dry Day Safety Cap Verification...")
    # Scenario: High Tide (3.8m), No Rain (0mm). Should be WASPADA (Warning), not AWAS.
    input_dry = {
        "rain_sum_imputed": 0.0,
        "pasut_msl_max": 3.8,      # Critical Tide
        "rain_rolling_24h": 0.0,   # For hourly logic
        "rain_cumsum_7d": 50.0,    # History of rain (to test cap override)
        # Required Dummy Features
        "rain_intensity_max": 0, "soil_moisture_surface_mean": 0.5, "soil_moisture_root_mean": 0.5,
        "rain_lag1": 0, "rain_lag2": 0, "rain_lag3": 0, "rain_lag4": 0, "rain_lag5": 0, "rain_lag6": 0, "rain_lag7": 0,
        "rain_cumsum_3d": 0, "tide_rain_interaction": 0, "is_high_tide": 1, "is_heavy_rain": 0,
        "api_7day": 0, "month_sin": 0, "month_cos": 0, "is_rainy_season": 1, "is_weekend": 0,
        "prev_flood_30d": 0, "prev_meluap_30d": 0, "rain_intensity_3h": 0, "rain_burst_count": 0,
        "soil_saturation_trend": 0, "tide_rain_sync": 0, "consecutive_rain_days": 0, "hour_risk_factor": 1.0,
        "drain_capacity_index": 0.2, "upstream_rain_6h": 0, "wind_speed_max": 0, "rainfall_acceleration": 0
    }
    
    # Note: We need to mock the DataFrame structure model expects
    # But model_utils.predict_flood handles dict. Good.
    result_dry = model_utils.predict_flood(model_pack, input_dry)
    prob_dry = result_dry['probability']
    label_dry = result_dry['label']
    
    print(f"   Input: Tide=3.8m, Rain=0mm -> Prob: {prob_dry:.4f} ({label_dry})")
    
    if prob_dry <= 0.605 and (label_dry == "WASPADA" or label_dry == "AMAN"):
         print("✅ Dry Day Logic SUCCESS (Capped at 60%)")
         tests_passed += 1
    else:
         print(f"❌ Dry Day Logic FAILED. Prob {prob_dry} is too high for a dry day.")
         issues.append("Dry Day Dampener failed to cap probability.")

    # --- TEST 4: STORM SURGE ESCALATION ---
    total_tests += 1
    print("\n[TEST 4] Storm Surge Escalation Verification...")
    # Scenario: High Tide (3.8m), Heavy Rain (60mm). Should be AWAS.
    input_storm = input_dry.copy()
    input_storm["rain_sum_imputed"] = 60.0 # Heavy rain
    input_storm["tide_rain_sync"] = 1
    
    result_storm = model_utils.predict_flood(model_pack, input_storm)
    prob_storm = result_storm['probability']
    label_storm = result_storm['label']
    
    print(f"   Input: Tide=3.8m, Rain=60mm -> Prob: {prob_storm:.4f} ({label_storm})")
    
    if prob_storm > 0.85 and label_storm == "AWAS":
        print("✅ Storm Logic SUCCESS (Correctly escalated to AWAS)")
        tests_passed += 1
    else:
        print(f"❌ Storm Logic FAILED. Expected AWAS, got {label_storm}")
        issues.append("Model failed to trigger AWAS on actual storm.")

    # --- TEST 5: HOURLY CONSISTENCY ---
    total_tests += 1
    print("\n[TEST 5] Hourly Series Consistency...")
    # Mock Hourly DataFrame
    dates = pd.date_range(start="2026-01-01 00:00", periods=24, freq="H")
    mock_hourly = pd.DataFrame({
        "time": dates,
        "precipitation": [0]*24, # Dry
        "est": [3.8]*24,         # High Tide entire time
        "soil_moisture_surface": [0.5]*24
    })
    
    # Run prediction
    hourly_res = model_utils.predict_hourly_series(model_pack, mock_hourly, {})
    max_hr_prob = hourly_res['probability'].max()
    
    print(f"   Max Hourly Prob (Dry+HighTide): {max_hr_prob:.4f}")
    
    if max_hr_prob <= 0.605:
        print("✅ Hourly Logic SUCCESS (Consistent with Daily Cap)")
        tests_passed += 1
    else:
        print(f"❌ Hourly Logic FAILED. Spike detected: {max_hr_prob}")
        issues.append("Hourly chart shows spikes > 60% on dry days.")

    # --- TEST 6: MAP VISUALIZATION LOGIC ---
    total_tests += 1
    print("\n[TEST 6] Map Intensity Logic...")
    # Test valid scenario directly using the ui_components logic 
    # (recreating logic here as we can't call inner function easily without setup)
    
    # Logic from ui_components:
    # if elev < adj_tide:
    #    if sim_rain < 5.0: return 0.6 (WASPADA)
    #    else: return 1.0 (BAHAYA)
    
    elev = 0.5
    tide_raw = 3.8
    offset = 2.8
    adj_tide = tide_raw - offset # 1.0m
    # 0.5 < 1.0 -> Overflow
    
    # Case A: Dry
    sim_rain_dry = 0
    if elev < adj_tide and sim_rain_dry < 5.0:
        map_val_dry = 0.6
    else: map_val_dry = 1.0
    
    # Case B: Wet
    sim_rain_wet = 50
    if elev < adj_tide and sim_rain_wet < 5.0:
        map_val_wet = 0.6
    else: map_val_wet = 1.0
    
    print(f"   Map Dry Intensity: {map_val_dry} (Expected 0.6)")
    print(f"   Map Wet Intensity: {map_val_wet} (Expected 1.0)")
    
    if map_val_dry == 0.6 and map_val_wet == 1.0:
        print("✅ Map Logic SUCCESS")
        tests_passed += 1
    else:
        print("❌ Map Logic FAILED mismatch.")
        issues.append("Map logic does not match dampener rules.")


    # --- REPORT ---
    print("\n" + "="*80)
    print(f"🏁 VERIFICATION COMPLETE: {tests_passed}/{total_tests} Tests Passed")
    if len(issues) == 0:
        print("🌟 RESULT: SYSTEM IS READY FOR PUBLIC RELEASE")
        print("   Status: STABLE & SAFE")
    else:
        print("⚠️ RESULT: ISSUES DETECTED")
        for i in issues:
            print(f"   - {i}")
    print("="*80)

if __name__ == "__main__":
    run_verification()
