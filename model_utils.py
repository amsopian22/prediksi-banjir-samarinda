

import joblib
import os
import pandas as pd
import numpy as np
import streamlit as st
import config
import logging
from typing import Tuple, Dict, Any, List

# Setup Logging
logger = logging.getLogger(__name__)

@st.cache_resource(ttl="1h")
def load_model() -> Dict[str, Any]:
    """Memuat model ML yang sudah dilatih."""
    model_path = config.MODEL_PATH
    
    if not os.path.exists(model_path):
        error_msg = f"File model '{model_path}' tidak ditemukan. Harap jalankan notebook terlebih dahulu untuk generate model."
        logger.error(error_msg)
        st.error(error_msg)
        return None
        
    try:
        logger.info(f"Loading model from {model_path}...")
        return joblib.load(model_path)
    except Exception as e:
        error_msg = f"Error loading model: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        return None


# --- RISK LEVEL SYSTEM ---

class FloodRiskSystem:
    """
    Centralized logic for determining flood risk levels, reasoning, and recommendations.
    """
    LEVEL_NORMAL = "NORMAL"
    LEVEL_WASPADA = "WASPADA"
    LEVEL_SIAGA = "SIAGA"
    LEVEL_AWAS = "AWAS"
    
    VERSION = "2.1" # Force cache update
    
    @staticmethod
    def get_risk_assessment(probability: float, input_data: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Determines the risk level and associated metadata based on probability and context.
        """
        # 1. Determine Level
        pct = probability * 100
        if pct < 50:
             level = FloodRiskSystem.LEVEL_NORMAL
             label = "AMAN"
             color = "green"
        elif pct < 70:
             level = FloodRiskSystem.LEVEL_WASPADA
             label = "WASPADA"
             color = "yellow"
        elif pct < 85:
             level = FloodRiskSystem.LEVEL_SIAGA
             label = "SIAGA"
             color = "orange"
        else:
             level = FloodRiskSystem.LEVEL_AWAS
             label = "AWAS"
             color = "red"
             
        # 2. Reasoning (Explainability)
        reasons = []
        main_factor = "Tidak Ada"
        
        if input_data:
            # Check Rain
            rain = input_data.get('rain_sum_imputed', input_data.get('curah_hujan_mm', 0))
            if rain > 150:
                 reasons.append(f"Hujan Ekstrem (>150mm)")
                 main_factor = "Hujan Ekstrem"
            elif rain > 100:
                reasons.append("Hujan Sangat Deras (>100mm)")
                main_factor = "Hujan Deras"
            elif rain > 50:
                reasons.append("Hujan Deras (>50mm)")
                if main_factor == "Tidak Ada": main_factor = "Hujan Deras"
                
            # Check Tide
            tide = input_data.get('pasut_msl_max', 0)
            if tide > 2.5:
                reasons.append(f"Pasang Laut Tinggi ({tide:.1f}m)")
                if main_factor == "Tidak Ada": main_factor = "Pasang Rob"
                elif "Hujan" in main_factor: main_factor += " & Rob"
            elif tide > 1.5:
                reasons.append("Pasang Laut Sedang")
                
            # Check Soil
            soil = input_data.get('soil_moisture_surface_mean', 0)
            if soil > 0.8:
                reasons.append("Tanah Jenuh Air")
                
        reason_text = ", ".join(reasons) if reasons else "Kondisi Meteorologis Normal"
        
        # 3. Recommendations (SOP)
        recommendation = ""
        if level == FloodRiskSystem.LEVEL_NORMAL:
            recommendation = "Kondisi kondusif. Masyarakat dapat beraktivitas seperti biasa dan tetap memantau informasi cuaca."
        elif level == FloodRiskSystem.LEVEL_WASPADA:
            recommendation = "Masyarakat dihimbau waspada. Perhatikan kondisi parit dan drainase di sekitar lingkungan."
        elif level == FloodRiskSystem.LEVEL_SIAGA:
            recommendation = "Masyarakat dan Pemerintah perlu memantau CCTV Kota dan update informasi resmi dari BPBD."
        elif level == FloodRiskSystem.LEVEL_AWAS:
            recommendation = "Ikuti arahan petugas di lapangan. Persiapkan langkah mitigasi dan hindari area rawan genangan."
            
        return {
            "level": level,
            "label": label,
            "color": color,
            "probability": probability,
            "main_factor": main_factor,
            "reasoning": reason_text,
            "recommendation": recommendation
        }

def predict_flood(model_pack: Dict[str, Any], input_data: Dict[str, float]) -> Dict[str, Any]:
    """
    Simulasi Manual: Returns a rich dictionary with risk assessment.
    """
    if not model_pack:
        return {"level": "UNKNOWN", "probability": 0, "reasoning": "Model not loaded"}

    model = model_pack.get("model")
    
    if not model:
         return {"level": "ERROR", "probability": 0}

    # Feature Engineering (Unified Logic)
    features_needed = model_pack.get("features", [])
    df_input = None
    
    # Prepare input logic for V2/V3/V6
    if "rain_sum_imputed" in features_needed:
        # Extract all lag values
        h_lag1 = input_data.get("rain_lag1", input_data.get("hujan_lag1", 0))
        h_lag2 = input_data.get("rain_lag2", input_data.get("hujan_lag2", 0))
        h_lag3 = input_data.get("rain_lag3", input_data.get("hujan_lag3", 0))
        h_lag4 = input_data.get("rain_lag4", input_data.get("hujan_lag4", 0))
        h_lag5 = input_data.get("rain_lag5", input_data.get("hujan_lag5", 0))
        h_lag6 = input_data.get("rain_lag6", input_data.get("hujan_lag6", 0))
        h_lag7 = input_data.get("rain_lag7", input_data.get("hujan_lag7", 0))
        
        rain_today = input_data.get("rain_sum_imputed", input_data.get("rain_rolling_24h", 0))
        rain_intensity = input_data.get("rain_intensity_max", 0)
        pasut_max = input_data.get("pasut_msl_max", 0)
        soil_surface = input_data.get("soil_moisture_surface_mean", 0.45)
        soil_root = input_data.get("soil_moisture_root_mean", 0.45)
        
        # Calculate derived features
        k = 0.85  # Decay factor
        api_7day = (
            rain_today +
            k * h_lag1 +
            (k**2) * h_lag2 +
            (k**3) * h_lag3 +
            (k**4) * h_lag4 +
            (k**5) * h_lag5 +
            (k**6) * h_lag6 +
            (k**7) * h_lag7
        )
        
        # Build comprehensive feature dict
        feature_dict = {
            # Original V4 features (25)
            "rain_sum_imputed": rain_today,
            "rain_intensity_max": rain_intensity,
            "soil_moisture_surface_mean": soil_surface,
            "soil_moisture_root_mean": soil_root,
            "soil_saturation_index": (soil_surface + soil_root) / 2,
            "pasut_msl_max": pasut_max,
            "rain_lag1": h_lag1,
            "rain_lag2": h_lag2,
            "rain_lag3": h_lag3,
            "rain_lag4": h_lag4,
            "rain_lag5": h_lag5,
            "rain_lag6": h_lag6,
            "rain_lag7": h_lag7,
            "rain_cumsum_3d": input_data.get("rain_cumsum_3d", rain_today + h_lag1 + h_lag2),
            "rain_cumsum_7d": input_data.get("rain_cumsum_7d", sum([rain_today, h_lag1, h_lag2, h_lag3, h_lag4, h_lag5, h_lag6])),
            "tide_rain_interaction": pasut_max * rain_today,
            "is_high_tide": 1 if pasut_max > 2.5 else 0,
            "is_heavy_rain": 1 if rain_today > 50 else 0,
            "api_7day": api_7day,
            "month_sin": input_data.get("month_sin", 0),
            "month_cos": input_data.get("month_cos", 1),
            "is_rainy_season": input_data.get("is_rainy_season", 1),
            "is_weekend": input_data.get("is_weekend", 0),
            "prev_flood_30d": input_data.get("prev_flood_30d", 0),
            "prev_meluap_30d": input_data.get("prev_meluap_30d", 0),
            
            # New V6 features (10)
            "rain_intensity_3h": input_data.get("rain_intensity_3h", rain_intensity * 3),
            "rain_burst_count": input_data.get("rain_burst_count", 0),
            "soil_saturation_trend": input_data.get("soil_saturation_trend", 0),
            "tide_rain_sync": 1 if (pasut_max > 2.5 and rain_today > 50) else 0,
            "consecutive_rain_days": input_data.get("consecutive_rain_days", 0),
            "hour_risk_factor": input_data.get("hour_risk_factor", 1.0),
            "upstream_rain_6h": input_data.get("upstream_rain_6h", 0),
            "wind_speed_max": input_data.get("wind_speed_max", 0),
            "rainfall_acceleration": input_data.get("rainfall_acceleration", 0),
        }
        
        # Calculate drain_capacity_index after rain_cumsum_7d is available
        feature_dict["drain_capacity_index"] = input_data.get("drain_capacity_index", feature_dict["rain_cumsum_7d"] / 200.0)
        
        df_input = pd.DataFrame([feature_dict])
        
        # Only select features that the model expects
        available_cols = [f for f in features_needed if f in df_input.columns]
        df_input = df_input[available_cols]
    else:
        # V1 MAPPING (Fallback)
        # Helper for API calculation
        h_hari_ini = input_data.get("hujan_hari_ini", 0)
        durasi = input_data.get("durasi_hari_ini", 0)
        # ... (API Calc Logic if needed, omitted for brevity as V2 is priority)
        # Just creating a minimal valid DF for legacy support if needed
        # Assuming V2 is the active model as per files viewed
        
        # We need to construct a dataframe for V1 legacy
        # Re-using a simplified version of the deleted logic
        h_lag1 = input_data.get("hujan_lag1", 0)
        h_lag2 = input_data.get("hujan_lag2", 0)
        h_lag3 = input_data.get("hujan_lag3", 0)
        pasut_max = input_data.get("pasut_msl_max", 0)
        pasut_slope = input_data.get("pasut_slope", 0)
        
        # Simple API Approximation for now to avoid the loop in predict
        hujan_3days = h_hari_ini + h_lag1 + h_lag2 
        
        df_input = pd.DataFrame([{
            'curah_hujan_mm': h_hari_ini,
            'durasi_hujan_jam': durasi,
            'pasut_msl_max': pasut_max,
            'pasut_slope': pasut_slope,
            'hujan_lag1': h_lag1,
            'hujan_lag2': h_lag2,
            'hujan_lag3': h_lag3,
            'hujan_3days': hujan_3days
        }])
        

    try:
        # Predict
        probabilitas = 0.0
        if hasattr(model, "predict_proba"):
             # Finding Banjir Index logic
             banjir_idx = 1
             if hasattr(model, "classes_"):
                 classes_list = list(model.classes_)
                 if "Banjir" in classes_list: banjir_idx = classes_list.index("Banjir")
                 elif 1 in classes_list: banjir_idx = 1
             
             probs = model.predict_proba(df_input)[0]
             probabilitas = probs[banjir_idx] if len(probs) > banjir_idx else probs[-1]
        else:
             probabilitas = float(model.predict(df_input)[0])

        # --- SPATIAL RISK BIAS (Heuristic for "Banjir Kiriman") ---
        # Adjust probability based on Flow Accumulation if available.
        # This allows the model to be sensitive to topography even if trained on single-point data.
        flow_acc = input_data.get('flow_accumulation', 0)
        
        # Logic: If high accumulation (>5000 cells) AND significant rain (>20mm), boost risk.
        if flow_acc > 5000 and input_data.get('rain_sum_imputed', 0) > 20:
            # Boost factor: up to 15% increase for very high accumulation areas
            boost = min(0.15, (flow_acc / 50000) * 0.10) 
            probabilitas = min(0.99, probabilitas + boost)
            
        # --- UPSTREAM RAIN BIAS (Kiriman) ---
        # If upstream rain is high (>20mm in last 6h), increase risk due to "Kiriman"
        upstream_rain = input_data.get('upstream_rain', 0)
        if upstream_rain > 20:
             # Add 10% risk if Heavy Upstream Rain
             probabilitas = min(0.99, probabilitas + 0.10)
            
             # Add 10% risk if Heavy Upstream Rain
             probabilitas = min(0.99, probabilitas + 0.10)
            
        # --- LAND USE/RUNOFF FACTOR (Impervious Surface) ---
        # Adjust based on Runoff Coefficient (0.5 to 1.0)
        # Higher runoff (Urban) -> Higher probability
        runoff_coeff = input_data.get('runoff_coefficient', 0.85) # Default to high if unknown
        if runoff_coeff > 0.8: # Urban bias
             probabilitas += 0.05
        elif runoff_coeff < 0.6: # Green area reduction
             probabilitas -= 0.05
         
        probabilitas = max(0.01, min(0.99, probabilitas))

        # Get Rich Assessment
        assessment = FloodRiskSystem.get_risk_assessment(probabilitas, input_data)
        
        # Add XAI (Feature Contributions)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            contributions = dict(zip(features_needed, importances))
            # Sort by importance
            sorted_contribs = dict(sorted(contributions.items(), key=lambda item: item[1], reverse=True))
            assessment["contributions"] = sorted_contribs
            
        return assessment

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return {"level": "ERROR", "probability": 0, "reasoning": str(e)}

def predict_hourly_series(model_pack: Dict[str, Any], hourly_df: pd.DataFrame, daily_lags_lookup: Dict) -> pd.DataFrame:
    """
    Predict flood risk for a series of hourly timestamps.
    """
    if not model_pack:
        return pd.DataFrame()
        
    model = model_pack["model"]
    threshold = model_pack.get("threshold", config.THRESHOLD_FLOOD_PROBABILITY)
    
    # 1. Calculate Rolling 24h Rain
    hourly_df['rain_rolling_24h'] = hourly_df['precipitation'].rolling(window=24, min_periods=1).sum()
    
    # 2. Daily Lags
    hourly_df['date'] = hourly_df['time'].dt.date
    
    def get_lags(row):
        d = row['date']
        lags = daily_lags_lookup.get(d, {'hujan_lag1':0, 'hujan_lag2':0, 'hujan_lag3':0})
        return pd.Series(lags)
        
    lags_df = hourly_df.apply(get_lags, axis=1)
    hourly_df = pd.concat([hourly_df, lags_df], axis=1)
    
    # 3. Tide Features
    hourly_df['pasut_msl_max'] = hourly_df['est']
    hourly_df['pasut_slope'] = hourly_df['est'].diff().fillna(0)
    
    # 4. Other Features
    hourly_df['durasi_hujan_jam'] = hourly_df['precipitation'].rolling(window=24).apply(lambda x: (x>0).sum())
    # 4. Other Features
    hourly_df['durasi_hujan_jam'] = hourly_df['precipitation'].rolling(window=24).apply(lambda x: (x>0).sum())
    
    # --- UPGRADE: API Implementation for Time Series ---
    # Vectorized calculation for efficiency is hard with recursive dependence on row-1.
    # We use a loop for clarity and correctness of the formula: API_t = P_t + k * API_t-1
    
    k = config.API_DECAY_FACTOR
    api_values = []
    prev_api = 0
    
    # Pre-calculate daily rains for the series to match 'day_date' logic if needed, 
    # BUT 'hujan_3days' in training was likely based on DAILY aggregations.
    # To stay consistent with the 'predict_flood' logic above which uses daily lags:
    # We should compute API based on the *daily* lags attached to each row.
    
    def calculate_row_api(row):
        # Re-construct daily sequence from lags attached to the row
        p_lag3 = row['hujan_lag3']
        p_lag2 = row['hujan_lag2']
        p_lag1 = row['hujan_lag1']
        p_now  = row['rain_rolling_24h'] # Using rolling 24h as proxy for "Today's Rain" in hourly context
        
        val_lag3 = p_lag3
        val_lag2 = p_lag2 + (k * val_lag3)
        val_lag1 = p_lag1 + (k * val_lag2)
        val_now  = p_now  + (k * val_lag1)
        return val_now

    hourly_df['hujan_3days'] = hourly_df.apply(calculate_row_api, axis=1)
    
    # Prepare X
    # Dynamic Feature Construction based on Model Version
    features_needed = model_pack.get("features", [])
    logger.info(f"Predict Hourly: Features Needed = {features_needed}")
    
    # Check if V2/V3 features are needed
    if "rain_sum_imputed" in features_needed:
        # Construct V2/V3 DataFrame
        X = pd.DataFrame()
        X['rain_sum_imputed'] = hourly_df['rain_rolling_24h']
        # For hourly series, rain_intensity_max is just the hourly precip
        X['rain_intensity_max'] = hourly_df['precipitation']
        
        # NEW: Rolling 3h rain
        X['rain_rolling_3h'] = hourly_df['precipitation'].rolling(window=3, min_periods=1).sum()
        
        # Soil Moisture (Assumed available in hourly_df from V2 fetcher)
        X['soil_moisture_surface_mean'] = hourly_df.get('soil_moisture_surface', 0.4) 
        X['soil_moisture_root_mean'] = hourly_df.get('soil_moisture_root', 0.4)
        X['pasut_msl_max'] = hourly_df['est']
        
        # Lag Features (t-1 to t-7) - renamed to rain_lag for V6
        for lag_num in range(1, 8):
            col_name_old = f'hujan_lag{lag_num}'
            col_name_new = f'rain_lag{lag_num}'
            X[col_name_new] = hourly_df.get(col_name_old, hourly_df.get(col_name_new, 0))
        
        # Calculate derived features
        k = config.API_DECAY_FACTOR
        X['api_7day'] = (
            X['rain_sum_imputed'] +
            k * X['rain_lag1'] +
            (k**2) * X['rain_lag2'] +
            (k**3) * X['rain_lag3'] +
            (k**4) * X['rain_lag4'] +
            (k**5) * X['rain_lag5'] +
            (k**6) * X['rain_lag6'] +
            (k**7) * X['rain_lag7']
        )
        
        # Add V4 derived features
        X['soil_saturation_index'] = (X['soil_moisture_surface_mean'] + X['soil_moisture_root_mean']) / 2
        X['rain_cumsum_3d'] = X['rain_sum_imputed'].rolling(window=72, min_periods=1).sum()  # 3 days = 72 hours
        X['rain_cumsum_7d'] = X['rain_sum_imputed'].rolling(window=168, min_periods=1).sum()  # 7 days = 168 hours
        X['tide_rain_interaction'] = X['pasut_msl_max'] * X['rain_sum_imputed']
        X['is_high_tide'] = (X['pasut_msl_max'] > 2.5).astype(int)
        X['is_heavy_rain'] = (X['rain_sum_imputed'] > 50).astype(int)
        
        # Time features
        X['month_sin'] = np.sin(2 * np.pi * hourly_df['time'].dt.month / 12)
        X['month_cos'] = np.cos(2 * np.pi * hourly_df['time'].dt.month / 12)
        X['is_rainy_season'] = hourly_df['time'].dt.month.isin([11, 12, 1, 2, 3]).astype(int)
        X['is_weekend'] = (hourly_df['time'].dt.dayofweek >= 5).astype(int)
        
        # Flood history features (rolling counts)
        X['prev_flood_30d'] = 0  # Would need historical labels, default to 0
        X['prev_meluap_30d'] = 0  # Would need historical labels, default to 0
        
        # === NEW V6 FEATURES ===
        # 1. rain_intensity_3h
        X['rain_intensity_3h'] = X['rain_rolling_3h']
        
        # 2. rain_burst_count - count hours with >10mm in last 24h
        X['rain_burst_count'] = (hourly_df['precipitation'] > 10).rolling(window=24, min_periods=1).sum()
        
        # 3. soil_saturation_trend - diff over 72h (3 days)
        X['soil_saturation_trend'] = X['soil_saturation_index'].diff(periods=72).fillna(0)
        
        # 4. tide_rain_sync
        X['tide_rain_sync'] = ((X['pasut_msl_max'] > 2.5) & (X['rain_sum_imputed'] > 50)).astype(int)
        
        # 5. consecutive_rain_days - approximate from hourly
        has_rain = (hourly_df['precipitation'] > 0).astype(int)
        rain_streaks = has_rain.groupby((has_rain != has_rain.shift()).cumsum()).cumsum()
        X['consecutive_rain_days'] = (rain_streaks / 24.0).fillna(0)  # Convert hours to days
        
        # 6. hour_risk_factor - higher at night (22:00-06:00)
        hour = hourly_df['time'].dt.hour
        is_night = ((hour >= 22) | (hour <= 6)).astype(int)
        X['hour_risk_factor'] = 1.0 + (is_night * 0.2)
        
        # 7. drain_capacity_index
        X['drain_capacity_index'] = (X['rain_cumsum_7d'] / 200.0).clip(0, 2.0)
        
        # 8. upstream_rain_6h - would need upstream data, default to 0
        X['upstream_rain_6h'] = hourly_df.get('upstream_precipitation', 0)
        
        # 9. wind_speed_max - max in last 24h
        X['wind_speed_max'] = hourly_df.get('wind_speed', 0)
        if 'wind_speed' in hourly_df.columns:
            X['wind_speed_max'] = hourly_df['wind_speed'].rolling(window=24, min_periods=1).max()
        
        # 10. rainfall_acceleration
        X['rainfall_acceleration'] = X['rain_intensity_max'].diff().fillna(0)
        
        # Only select features that the model expects
        available_cols = [f for f in features_needed if f in X.columns]
        X = X[available_cols]
        
    else:
        # Fallback to V1 Legacy Features
        X = pd.DataFrame({
            'curah_hujan_mm': hourly_df['rain_rolling_24h'],
            'durasi_hujan_jam': hourly_df['durasi_hujan_jam'],
            'pasut_msl_max': hourly_df['pasut_msl_max'],
            'pasut_slope': hourly_df['pasut_slope'],
            'hujan_lag1': hourly_df['hujan_lag1'],
            'hujan_lag2': hourly_df['hujan_lag2'],
            'hujan_lag3': hourly_df['hujan_lag3'],
            'hujan_3days': hourly_df['hujan_3days']
        })
    
    X = X.fillna(0)
    

    # Predict
    if hasattr(model, "predict_proba"):
        banjir_idx = 1 # Default for V2 (Binary: 0=Aman, 1=Banjir)
        # Check classes
        if hasattr(model, "classes_"):
            classes_list = list(model.classes_)
            if "Banjir" in classes_list: # V1 Style
                banjir_idx = classes_list.index("Banjir")
            elif 1 in classes_list or 1.0 in classes_list: # V2 Style (Binary)
                banjir_idx = 1
                
        probs = model.predict_proba(X)[:, banjir_idx]
    else:
        probs = model.predict(X)
        
    hourly_df['probability'] = probs
    
    # Map Probability to Status Label (Vectorized)
    def get_status(p):
        if p < 0.5: return "AMAN"
        elif p < 0.7: return "WASPADA"
        elif p < 0.85: return "SIAGA"
        else: return "AWAS"
        
    hourly_df['status'] = hourly_df['probability'].apply(get_status)
    
    # Clean up temporary columns if needed, but returning select columns is fine
    cols_to_return = ['time', 'probability', 'status', 'rain_rolling_24h', 'est', 'hujan_3days', 'precipitation']
    # Add Soil Moisture if present
    for col in ['soil_moisture_surface', 'soil_moisture_root']:
        if col in hourly_df.columns:
            cols_to_return.append(col)
            
    return hourly_df[cols_to_return]
