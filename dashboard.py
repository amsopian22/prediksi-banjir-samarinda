

import streamlit as st
import pandas as pd
import datetime
import os
import logging
import warnings

# Suppress SHAP warnings globally
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*TreeExplainer.*')
warnings.filterwarnings('ignore', message='.*feature_perturbation.*')

# New Modules
import data_ingestion
import model_utils
import tide_utils
from feature_extraction import SpatialFeatureExtractor
import config
import ui_components
import sentinel_utils
import monitoring_map  # NEW: 5 Lokasi Monitoring Map
import telegram_bot  # NEW: Telegram Alert Integration

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Reduce SHAP logging noise
logging.getLogger('shap').setLevel(logging.ERROR)

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sistem Peringatan Dini Banjir Samarinda",
    page_icon="🌊",
    layout="wide"
)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    model_pack = model_utils.load_model()
    tide_predictor = data_ingestion.TidePredictor()
    weather_fetcher = data_ingestion.WeatherFetcher()
    
    try:
        spatial_extractor = SpatialFeatureExtractor()
    except Exception as e:
        logger.error(f"Spatial Init Error: {e}")
        spatial_extractor = None
        
    return model_pack, tide_predictor, weather_fetcher, spatial_extractor

model_pack, tide_predictor, weather_fetcher, spatial_extractor = load_resources()

# --- UI UTAMA ---
# --- UI UTAMA ---
ui_components.load_custom_css() # Inject Modern CSS

# Header (Minimalist)
st.markdown('<div style="margin-top: -60px;"></div>', unsafe_allow_html=True) # Spacer hack
st.markdown('<h1 class="hero-title">Sistem Peringatan Dini Banjir</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.2rem; opacity: 0.8; margin-bottom: 30px;">Dashboard Eksekutif Pantau Kota Samarinda</p>', unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Konfigurasi")
mode = st.sidebar.radio("Mode Operasi:", ["Real-time Monitoring", "Simulasi Manual"])

# Location Selector
locations = config.LOCATIONS
selected_loc_name = st.sidebar.selectbox("Pilih Lokasi Pantau:", list(locations.keys()))
loc_data = locations[selected_loc_name]
lat = loc_data[0]
lon = loc_data[1]

# Spatial Risk Context
if spatial_extractor:
    with st.spinner("Menganalisis Topografi..."):
        loc_feats = spatial_extractor.get_features(lat, lon)
        
    elev = loc_feats.get('elevation', 0)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📍 Topografi Wilayah")
    st.sidebar.write(f"**Elevasi Tanah**: {elev:.1f} mdpl")
    st.sidebar.write(f"**Akumulasi Aliran**: {loc_feats.get('flow_accumulation', 0):.0f} sel (Risiko Kiriman)")
        
# --- LOGIKA OPERASI ---
    # 1. Fetch Weather (Cached Wrapper)
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_cached_weather_v2(lat, lon, label=None):
        return weather_fetcher.fetch_weather_data(lat=lat, lon=lon, location_label=label)

    # 2. Hybrid Validation (Cached Wrapper)
    @st.cache_data(ttl=900, show_spinner=False) # Cache for 15 mins
    def check_hybrid_validation(lat, lon):
        validator = sentinel_utils.FloodValidator()
        return validator.get_hybrid_status(lat, lon)

    with st.spinner("Mengambil data cuaca terkini..."):
        weather_df = get_cached_weather_v2(lat, lon)
        
        # 1b. Fetch Upstream Weather (Hulu)
        # Using the first upstream location defined in config
        upstream_name = list(config.UPSTREAM_LOCATIONS.keys())[0]
        upstream_df = get_cached_weather_v2(None, None, upstream_name)
    
    # --- UPSTREAM STATUS DISPLAY ---
    if not upstream_df.empty:
        # Calculate recent rain in upstream (last 6-12 hours)
        upstream_rain_recent = upstream_df['precipitation'].rolling(window=6).sum().iloc[-1]
        
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### ⛰️ Status Hulu ({upstream_name.split('(')[0].strip()})")
        
        if upstream_rain_recent > 20: 
            st.sidebar.error(f"⚠️ **Hujan Deras di Hulu!**\n{upstream_rain_recent:.1f} mm (6 jam terakhir)")
            st.sidebar.caption(f"Estimasi kiriman sampai: {config.UPSTREAM_LAG_HOURS} jam lagi.")
        elif upstream_rain_recent > 5:
            st.sidebar.warning(f"🌧️ **Hujan Ringan**\n{upstream_rain_recent:.1f} mm")
        else:
            st.sidebar.success(f"☁️ **Kering/Aman**\n{upstream_rain_recent:.1f} mm")

    if not weather_df.empty:
        # 2. Predict Tides
        weather_df['est'] = tide_predictor.predict_hourly(weather_df['date'])
        
        # Rename for consistency
        weather_df = weather_df.rename(columns={'date': 'time'})
        hourly_df = weather_df
        
        # 3. Create Lags Lookup
        hourly_df['day_date'] = hourly_df['time'].dt.date
        daily_sums = hourly_df.groupby('day_date')['precipitation'].sum() 
        
        lags_lookup = {}
        unique_dates = hourly_df['day_date'].unique()
        for d in unique_dates:
            # Create lag dates for t-1 to t-7
            lags_lookup[d] = {}
            for lag_num in range(1, 8):
                lag_date = d - datetime.timedelta(days=lag_num)
                lags_lookup[d][f'hujan_lag{lag_num}'] = daily_sums.get(lag_date, 0)
            
            
            
        # 4. Predict Risk Series
        if model_pack:
            hourly_risk_df = model_utils.predict_hourly_series(model_pack, hourly_df, lags_lookup)
            
            # --- Current Status ---
            now = pd.Timestamp.now(tz=config.TIMEZONE)
            current_row = hourly_risk_df.iloc[(hourly_risk_df['time'] - now).abs().argsort()[:1]]
            

            if not current_row.empty:
                curr_prob = current_row['depth_cm'].values[0] # Renamed variable but kept for logic flow
                curr_depth = current_row['depth_cm'].values[0]
                curr_status = current_row['status'].values[0]
                curr_rain_24h = current_row['rain_rolling_24h'].values[0]
                curr_tide = current_row['est'].values[0]
                
                # --- UI Simulation Controls (for overriding real-time values) ---
                with st.sidebar:
                    st.divider()
                    st.subheader("⚙️ Simulasi Data")
                    sim_rain = st.slider("Simulasi Hujan (mm/hari)", 0, 200, int(curr_rain_24h))
                    sim_tide = st.slider("Simulasi Pasang (meter)", 0.0, 4.0, float(curr_tide))
                    sim_soil = st.slider("Simulasi Kelembaban Tanah", 0.0, 1.0, 0.45)
                    
                    if st.button("Update Situasi"):
                        # Update current variables for simulation context
                        curr_rain_24h = sim_rain
                        curr_tide = sim_tide
                        
                        st.session_state['sim_active'] = True
                        st.session_state['sim_rain'] = sim_rain
                        st.session_state['sim_tide'] = sim_tide
                        st.session_state['sim_soil'] = sim_soil
                        st.rerun()

                # Apply Simulation State if Active
                if st.session_state.get('sim_active'):
                    curr_rain_24h = st.session_state['sim_rain']
                    curr_tide = st.session_state['sim_tide']
                    sim_soil = st.session_state['sim_soil']
                    st.toast(f"✅ Simulasi Aktif: Hujan {curr_rain_24h}mm, Pasang {curr_tide}m")

                # Prepare Data for Full Assessment (Real-time or Simulated)
                # Getting soil moisture default
                current_soil = current_row.get('soil_moisture_surface', 0.45)
                # Use simulated soil if active, else real
                final_soil = sim_soil if st.session_state.get('sim_active') else current_soil if isinstance(current_soil, float) else 0.45

                # Get Runoff Coeff from Config
                # config.LOCATIONS[selected_loc_name] is now (lat, lon, runoff)
                loc_data = config.LOCATIONS[selected_loc_name]
                runoff_val = loc_data[2] if len(loc_data) > 2 else 0.85

                input_data_pack = {
                    "rain_sum_imputed": curr_rain_24h,
                    "rain_intensity_max": current_row['precipitation'].values[0] if not current_row.empty else 0,
                    "soil_moisture_surface_mean": final_soil,
                    "soil_moisture_root_mean": final_soil,
                    "pasut_msl_max": curr_tide,
                    "hujan_lag1": 0, "hujan_lag2": 0, # Real-time simplification
                    "upstream_rain": upstream_rain_recent if 'upstream_rain_recent' in locals() else 0,
                    "flow_accumulation": loc_feats.get('flow_accumulation', 0) if 'loc_feats' in locals() else 0,
                    "runoff_coefficient": runoff_val
                }
                
                # Get Full Assessment
                assessment = model_utils.predict_flood(model_pack, input_data_pack)
                
                if assessment is None:
                    assessment = {
                        "level": "ERROR",
                        "label": "ERROR",
                        "depth_cm": 0,
                        "color": "gray",
                        "recommendation": "Gagal memperoleh hasil prediksi. Silakan coba lagi.",
                        "reasoning": "Internal Model Error",
                        "main_factor": "Data Error"
                    }

                # Defensive Check for Deployment Cache Issues
                if isinstance(assessment, tuple):
                    st.toast("⚠️ Warning: Legacy model Output detected. cache might be stale.", icon="⚠️")
                    # Manual patch for legacy tuple: (status, prob, thresh, akum, contrib)
                    assessment = {
                        "level": "WASPADA" if assessment[0] == "BAHAYA BANJIR" else "NORMAL",
                        "label": assessment[0],
                        "probability": assessment[1],
                        "color": "yellow" if assessment[0] == "BAHAYA BANJIR" else "green", 
                        "recommendation": "Sistem sedang pembaruan cache. Silakan refresh.",
                        "reasoning": "Legacy Output",
                        "main_factor": "Unknown"
                    }
                
                curr_prob = assessment.get('depth_cm', 0)
                curr_status = assessment.get('label', 'Unknown')

                # --- HYBRID VALIDATION (Satellite + Radar) ---
                val_result = None
                
                # Check Hybrid Validation if Risk is High
                if curr_status in ["WASPADA", "SIAGA", "AWAS"]:
                     hybrid_status = check_hybrid_validation(lat, lon)
                     if hybrid_status:
                        val_result = {
                            "status": "CONFIRMED",
                            "label": hybrid_status['label'],
                            "detail": hybrid_status['detail'],
                            "color": hybrid_status['color_hex']
                        }
                
                # --- VISUALISASI MENGGUNAKAN UI_COMPONENTS (COMMAND CENTER) ---
                
                # 1. Hero Section (Command Status)
                ui_components.render_command_center_hero(assessment, validation=val_result)
                ui_components.render_status_reference()
                
                # --- TELEGRAM ALERT (Auto-send on SIAGA/AWAS) ---
                alert_key = f"telegram_alert_{selected_loc_name}_{curr_status}"
                if curr_status in ["SIAGA", "AWAS"]:
                    # Check if alert already sent in this session (cooldown)
                    if alert_key not in st.session_state:
                        try:
                            sent = telegram_bot.send_flood_alert(
                                location=selected_loc_name,
                                status=curr_status,
                                depth_cm=curr_depth,
                                reasoning=assessment.get('reasoning', ''),
                                recommendation=assessment.get('recommendation', ''),
                                min_level="SIAGA"
                            )
                            if sent:
                                st.session_state[alert_key] = True
                                st.toast(f"📱 Alert Telegram terkirim: {curr_status}", icon="✅")
                        except Exception as e:
                            logger.warning(f"Telegram alert failed: {e}")
                
                # 2. Operational Fronts (The 3-Fronts Grid)
                # Prepare Dicts
                w_dict = {'rain_24h': curr_rain_24h}
                u_dict = {'rain_recent': upstream_rain_recent}
                o_dict = {'tide_max': curr_tide}
                s_dict = {'soil_moisture': final_soil}
                
                ui_components.render_operational_fronts(w_dict, u_dict, o_dict, s_dict)
                
                # 3. Decision Support (Tabs)
                st.markdown("---")
                
                # Filter Date (Global for Dashboard)
                col_filter, _ = st.columns([1, 4])
                with col_filter:
                    selected_date = st.date_input("📅 Filter Tanggal Dokumen Operasi:", value=pd.Timestamp.now().date())
                
                import json
                geojson_data = None
                if os.path.exists(config.RISK_MAP_PATH):
                    with open(config.RISK_MAP_PATH) as f:
                        geojson_data = json.load(f)

                # Filter Data based on Selected Date
                filtered_df = hourly_risk_df[hourly_risk_df['time'].dt.date == selected_date]
                
                
                ui_components.render_decision_support(geojson_data, filtered_df if not filtered_df.empty else hourly_risk_df, lat, lon, selected_date)
                
                if filtered_df.empty:
                     st.warning(f"Data spesifik untuk {config.format_id_date(selected_date)} tidak ditemukan. Menampilkan data umum.")

                # 4. Monitoring Locations Map (DISABLED - Causes continuous refresh)
                # st.divider()
                # st.subheader("🗺️ Peta 5 Lokasi Monitoring")
                # monitoring_map.render_monitoring_locations_map(model_pack, current_time=pd.Timestamp.now())

                # 5. 7-Day Forecast
                st.divider()
                st.subheader("📅 Prediksi 7 Hari Ke Depan")
                
                hourly_df['date_only'] = hourly_df['time'].dt.date
                today_date = pd.Timestamp.now(tz=config.TIMEZONE).date()
                
                # Filter for Today and Future
                future_groups = [
                    (d, g) for d, g in hourly_df.groupby('date_only') 
                    if d >= today_date
                ]
                
                # Prepare Layout: One row of 7
                row1 = st.columns(7)
                
                idx = 0
                for date_val, group in future_groups:
                    if idx >= 7: break
                    
                    # Select container
                    current_col = row1[idx]
                    
                    daily_rain = group['precipitation'].sum()
                    daily_max_tide = group['est'].max()
                    
                    # Correct Lags from Lookup
                    lags = lags_lookup.get(date_val, {})
                    lag1 = lags.get('hujan_lag1', 0)
                    lag2 = lags.get('hujan_lag2', 0)
                    lag3 = lags.get('hujan_lag3', 0)
                    lag4 = lags.get('hujan_lag4', 0)
                    lag5 = lags.get('hujan_lag5', 0)
                    lag6 = lags.get('hujan_lag6', 0)
                    lag7 = lags.get('hujan_lag7', 0)
                    
                    # Actual Soil Moisture
                    soil_mean = group['soil_moisture_surface'].mean() if 'soil_moisture_surface' in group else 0.5

                    # Calculate temporal features
                    import datetime as dt
                    date_obj = pd.to_datetime(date_val)
                    month = date_obj.month
                    is_weekend = 1 if date_obj.dayofweek >= 5 else 0  # Saturday=5, Sunday=6
                    is_rainy_season = 1 if month in [11, 12, 1, 2, 3, 4] else 0
                    
                    # Month cyclical encoding
                    import math
                    month_sin = math.sin(2 * math.pi * month / 12)
                    month_cos = math.cos(2 * math.pi * month / 12)

                    d_input = {
                        "rain_sum_imputed": daily_rain,
                        "rain_intensity_max": group['precipitation'].max(),
                        "soil_moisture_surface_mean": soil_mean,
                        "soil_moisture_root_mean": soil_mean,
                        "pasut_msl_max": daily_max_tide,
                        "hujan_lag1": lag1, 
                        "hujan_lag2": lag2,
                        "hujan_lag3": lag3,
                        "hujan_lag4": lag4,
                        "hujan_lag5": lag5,
                        "hujan_lag6": lag6,
                        "hujan_lag7": lag7,
                        # Temporal features (CRITICAL!)
                        "is_weekend": is_weekend,
                        "is_rainy_season": is_rainy_season,
                        "month_sin": month_sin,
                        "month_cos": month_cos,
                        # Additional derived features
                        "rain_cumsum_3d": daily_rain + lag1 + lag2,
                        "rain_cumsum_7d": daily_rain + lag1 + lag2 + lag3 + lag4 + lag5 + lag6,
                        "upstream_rain_6h": upstream_rain_recent if 'upstream_rain_recent' in locals() else 0,
                        "prev_flood_30d": 0,
                        "prev_meluap_30d": 0,
                        
                        # V6 Complete Features to avoid default bias
                        "drain_capacity_index": (daily_rain + lag1 + lag2 + lag3 + lag4 + lag5 + lag6) / 200.0,
                        "tide_rain_sync": 1 if (daily_max_tide > 2.5 and daily_rain > 50) else 0,
                        "rain_intensity_3h": daily_rain / 4.0 if daily_rain > 10 else 0, # Rough estimate
                        "rainfall_acceleration": 0, # Assume stable for daily forecast
                        "rain_burst_count": 0,
                        "hour_risk_factor": 1.0,
                    }
                    
                    # Predict using new function
                    d_assess = model_utils.predict_flood(model_pack, d_input)
                    if d_assess is None:
                        d_assess = {"label": "ERROR", "depth_cm": 0, "color": "gray"}

                    d_status = d_assess.get('label', 'Unknown')
                    d_depth = d_assess.get('depth_cm', 0)
                    d_color = d_assess.get('color', 'gray')
                    
                    with current_col:
                        date_str = config.format_id_date(date_val)
                        st.markdown(f"**{date_str}**")
                        
                        # Map color to UI style
                        status_class = "success"
                        if d_color == "yellow": status_class = "warning"
                        elif d_color == "orange": status_class = "error" # Streamlit has no orange alert
                        elif d_color == "red": status_class = "error"
                        
                        if status_class == "success":
                            st.success(f"{d_status}\n{d_depth:.0f} cm")
                        elif status_class == "warning":
                            st.warning(f"{d_status}\n{d_depth:.0f} cm")
                        else:
                            st.error(f"{d_status}\n{d_depth:.0f} cm")
                            
                        st.caption(f"🌧️ {daily_rain:.1f}mm\n🌊 {daily_max_tide:.1f}m")
                        
                        # DEBUG: Show key features
                        rain_cumsum_7d = daily_rain + lag1 + lag2 + lag3 + lag4 + lag5 + lag6
                        st.caption(f"🔍 Week: {'WEEKEND' if is_weekend else 'WEEKDAY'} | Cumsum7d: {rain_cumsum_7d:.1f}mm")
                    idx += 1
                
                # Explanation for constant low probability
                if hourly_df['precipitation'].sum() == 0:
                     st.info("💡 **Catatan:** Probabilitas risiko rendah dan konstan dikarenakan prakiraan cuaca menunjukkan **tidak ada hujan** (0 mm) untuk periode ini. Risiko didominasi oleh faktor pasang surut rutin.")

                # --- DEBUG SECTION (To diagnose strange 7-day predictions) ---
                with st.expander("🕵️‍♂️ Debug Data Prediksi 7 Hari (Teknis)", expanded=False):
                    st.caption("Tabel ini menampilkan nilai fitur yang digunakan untuk prediksi 7 hari ke depan.")
                    
                    # Re-generate debug data list
                    debug_data_list = []
                    idx_debug = 0
                    for date_val, group in future_groups:
                        if idx_debug >= 7: break
                        
                        daily_rain = group['precipitation'].sum()
                        daily_max_tide = group['est'].max()
                        lags = lags_lookup.get(date_val, {})
                        
                        import datetime as dt
                        date_obj = pd.to_datetime(date_val)
                        is_weekend = 1 if date_obj.dayofweek >= 5 else 0
                        
                        lag1 = lags.get('hujan_lag1', 0)
                        lag2 = lags.get('hujan_lag2', 0)
                        rain_cumsum_7d = daily_rain + lag1 + lag2 + lags.get('hujan_lag3',0) + lags.get('hujan_lag4',0) + lags.get('hujan_lag5',0) + lags.get('hujan_lag6',0)
                        
                        # Re-construct input for explanation
                        d_input_debug = {
                            "rain_sum_imputed": daily_rain,
                            "rain_intensity_max": group['precipitation'].max(),
                            "soil_moisture_surface_mean": 0.5, # Default
                            "soil_moisture_root_mean": 0.5,
                            "pasut_msl_max": daily_max_tide,
                            "hujan_lag1": lag1, 
                            "hujan_lag2": lag2,
                            "hujan_lag3": lags.get('hujan_lag3', 0),
                            "hujan_lag4": lags.get('hujan_lag4', 0),
                            "hujan_lag5": lags.get('hujan_lag5', 0),
                            "hujan_lag6": lags.get('hujan_lag6', 0),
                            "hujan_lag7": lags.get('hujan_lag7', 0),
                            "is_weekend": is_weekend,
                            "is_rainy_season": 1,
                            # Fix NameError: use math instead of np
                            "month_sin": math.sin(2 * math.pi * date_obj.month / 12),
                            "month_cos": math.cos(2 * math.pi * date_obj.month / 12),
                            # V6 Features
                            "rain_cumsum_3d": daily_rain + lag1 + lag2,
                            "rain_cumsum_7d": rain_cumsum_7d,
                            "drain_capacity_index": rain_cumsum_7d / 200.0,
                            "tide_rain_sync": 1 if (daily_max_tide > 2.5 and daily_rain > 50) else 0,
                            "rain_intensity_3h": daily_rain / 4.0 if daily_rain > 10 else 0,
                            "rainfall_acceleration": 0,
                            "rain_burst_count": 0,
                            "hour_risk_factor": 1.0,
                            "upstream_rain_6h": 0,
                            "wind_speed_max": 0,
                            "prev_flood_30d": 0,
                            "prev_meluap_30d": 0,
                            "tide_rain_interaction": daily_max_tide * daily_rain,
                            "is_high_tide": 1 if daily_max_tide > 2.5 else 0,
                            "is_heavy_rain": 1 if daily_rain > 50 else 0,
                            "api_7day": 0 # Simplified
                        }
                        
                        # Get Prediction & Explanation
                        d_assess_debug = model_utils.predict_flood(model_pack, d_input_debug)
                        if d_assess_debug is None:
                             d_assess_debug = {"label": "ERROR", "depth_cm": 0, "contributions": {}}

                        contributors = ""
                        if "contributions" in d_assess_debug:
                             # Top 3
                             top3 = list(d_assess_debug["contributions"].items())[:3]
                             contributors = ", ".join([f"{k}={v:.3f}" for k,v in top3])

                        row = {
                            "Tanggal": config.format_id_date(date_val),
                            "Rain (mm)": float(f"{daily_rain:.2f}"),
                            "Tide (m)": float(f"{daily_max_tide:.2f}"),
                            "Cumsum7d": float(f"{rain_cumsum_7d:.2f}"),
                            "Prediksi": f"{d_assess_debug['label']} ({d_assess_debug['depth_cm']:.1f} cm)",
                            "Penyebab Utama (XAI)": contributors
                        }
                        debug_data_list.append(row)
                        idx_debug += 1
                        
                    debug_df = pd.DataFrame(debug_data_list)
                    st.dataframe(debug_df, use_container_width=True)
                # -------------------------------------------------------------

                # 6. Map Simulation (Moved to top)
                # Code removed to prevent duplicate widget error
    
    else:
        st.error("Gagal mengambil data cuaca. Cek koneksi internet.")

else: # Mode Simulasi
    st.warning("🛠️ Mode Simulasi Manual")
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        hujan = st.number_input("Curah Hujan (mm)", 0.0, 500.0, 50.0)
        durasi = st.number_input("Durasi (jam)", 0.0, 24.0, 5.0)
        pasut = st.number_input("Tinggi Pasut Max (m)", -2.0, 5.0, 1.5)
        
    with col_sim2:
        lag1 = st.number_input("Hujan Lag 1 (mm)", 0.0, 200.0, 20.0)
        lag2 = st.number_input("Hujan Lag 2 (mm)", 0.0, 200.0, 10.0)
        lag3 = st.number_input("Hujan Lag 3 (mm)", 0.0, 200.0, 0.0)
        
    # Predict Button
    if st.button("Hitung Risiko"):
        sim_input = {
            "hujan_hari_ini": hujan,
            "durasi_hari_ini": durasi,
            "pasut_msl_max": pasut,
            "pasut_slope": 0.1, # Mock
            "hujan_lag1": lag1,
            "hujan_lag2": lag2,
            "hujan_lag3": lag3,
            # Params for V2 compatibility
            "rain_sum_imputed": hujan,
            "rain_intensity_max": hujan/durasi if durasi > 0 else 0,
            "soil_moisture_surface_mean": 0.5,
            "soil_moisture_root_mean": 0.5
        }
        
        assessment = model_utils.predict_flood(model_pack, sim_input)
        
        if assessment is None:
            assessment = {"level": "ERROR", "label": "ERROR", "depth_cm": 0, "contributions": {}, "reasoning": "Error"}

        # Display Result using new components
        ui_components.render_executive_summary(assessment)
        ui_components.render_risk_context(assessment)
        
        st.divider()
        st.caption("Kontribusi Fitur:")
        st.json(assessment.get("contributions", {}))

st.markdown("---")
st.caption("Supported by: Utide (Tidal Analysis), Open-Meteo (Weather), & Scikit-Learn (ML).")
