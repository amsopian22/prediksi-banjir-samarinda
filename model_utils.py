
import joblib
import os
import pandas as pd
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

def predict_flood(model_pack: Dict[str, Any], input_data: Dict[str, float]) -> Tuple[str, float, float, float, Dict[str, float]]:
    """Melakukan prediksi menggunakan model untuk data tunggal (Simulasi Manual)."""
    if not model_pack:
        return "Unknown", 0.0, 0.5, 0.0, {}

    model = model_pack.get("model")
    # Use config threshold if not in model pack (backward compatibility)
    threshold = model_pack.get("threshold", config.THRESHOLD_FLOOD_PROBABILITY)
    
    if not model:
         return "Error", 0.0, 0.5, 0.0, {}

    # Feature Engineering logic (same as training)
    features_needed = model_pack.get("features", [])
    hujan_3days = 0 # Default initialization for V2 compatibility logic checking
    
    if "rain_sum_imputed" in features_needed:
        # V2 Logic
        # Mapping input_data (which comes from Simulation sliders or defaults) to V2 Features
        # Note: Simulation inputs in Dashboard need to provide these keys!
        # Dashboard v2 inputs: rain_sum_imputed, rain_intensity_max, soil_moisture..., pasut_msl_max, lags
        
        df_input = pd.DataFrame([{
            "rain_sum_imputed": input_data.get("rain_sum_imputed", 0),
            "rain_intensity_max": input_data.get("rain_intensity_max", 0),
            "soil_moisture_surface_mean": input_data.get("soil_moisture_surface_mean", 0.4),
            "soil_moisture_root_mean": input_data.get("soil_moisture_root_mean", 0.4),
            "pasut_msl_max": input_data.get("pasut_msl_max", 0),
            "hujan_lag1": input_data.get("hujan_lag1", 0),
            "hujan_lag2": input_data.get("hujan_lag2", 0)
        }])
        
        # Ensure column order matches training
        df_input = df_input[features_needed]
        
    else:
        # V1 Logic (Legacy)
        h_hari_ini = input_data.get("hujan_hari_ini", 0)
        durasi = input_data.get("durasi_hari_ini", 0)
        h_lag1 = input_data.get("hujan_lag1", 0)
        h_lag2 = input_data.get("hujan_lag2", 0)
        h_lag3 = input_data.get("hujan_lag3", 0)
        
        # ... API calculation ...
        k = config.API_DECAY_FACTOR
        api_lag3 = h_lag3
        api_lag2 = h_lag2 + (k * api_lag3)
        api_lag1 = h_lag1 + (k * api_lag2)
        api_now  = h_hari_ini + (k * api_lag1)
        hujan_3days = api_now
        
        pasut_max = input_data.get("pasut_msl_max", 0)
        pasut_slope = input_data.get("pasut_slope", 0)
    
        # DataFrame sesuai format training V1
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
        # Prediksi
        if hasattr(model, "predict_proba"):
            # Robust Class Index Finding
            # Assuming classes are ['Air Meluap', 'Aman', 'Banjir'] sorted
            # Verify if classes_ attribute exists
            banjir_idx = 2 # Default fallback
            if hasattr(model, "classes_"):
                classes_list = list(model.classes_)
                if "Banjir" in classes_list:
                    banjir_idx = classes_list.index("Banjir")
            
            probabilitas = model.predict_proba(df_input)[0][banjir_idx]
        else:
            probabilitas = float(model.predict(df_input)[0])

        status = "BAHAYA BANJIR" if probabilitas >= threshold else "AMAN"
        
        # XAI: Hitung Kontribusi Fitur
        sorted_contributions = {}
        feature_names = model_pack.get("features", df_input.columns.tolist())
        
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            contributions = {}
            for name, imp in zip(feature_names, importances):
                 if name in df_input.columns:
                    contributions[name] = imp
            sorted_contributions = dict(sorted(contributions.items(), key=lambda item: item[1], reverse=True))

        return status, probabilitas, threshold, hujan_3days, sorted_contributions

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return "Error", 0.0, threshold, hujan_3days, {}

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
    
    # Check if V2 features are needed
    if "rain_sum_imputed" in features_needed:
        # Construct V2 DataFrame
        X = pd.DataFrame()
        X['rain_sum_imputed'] = hourly_df['rain_rolling_24h']
        # For hourly series, rain_intensity_max is just the hourly precip
        X['rain_intensity_max'] = hourly_df['precipitation']
        # Soil Moisture (Assumed available in hourly_df from V2 fetcher)
        X['soil_moisture_surface_mean'] = hourly_df.get('soil_moisture_surface', 0.4) 
        X['soil_moisture_root_mean'] = hourly_df.get('soil_moisture_root', 0.4)
        X['pasut_msl_max'] = hourly_df['est']
        X['hujan_lag1'] = hourly_df['hujan_lag1'] # Placeholder/0
        X['hujan_lag2'] = hourly_df['hujan_lag2'] # Placeholder/0
        
        # Ensure order matches
        X = X[features_needed]
        
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
    hourly_df['status'] = hourly_df['probability'].apply(lambda x: "BAHAYA" if x >= threshold else "AMAN")
    
    # Clean up temporary columns if needed, but returning select columns is fine
    cols_to_return = ['time', 'probability', 'status', 'rain_rolling_24h', 'est', 'hujan_3days', 'precipitation']
    # Add Soil Moisture if present
    for col in ['soil_moisture_surface', 'soil_moisture_root']:
        if col in hourly_df.columns:
            cols_to_return.append(col)
            
    return hourly_df[cols_to_return]
