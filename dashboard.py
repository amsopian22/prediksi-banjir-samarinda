
import streamlit as st
import pandas as pd
import datetime
import os
import logging

# New Modules
import data_ingestion
import model_utils
import tide_utils
from feature_extraction import SpatialFeatureExtractor
import config
import ui_components

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
lat, lon = locations[selected_loc_name]

# Spatial Risk Context
if spatial_extractor:
    with st.spinner("Menganalisis Topografi..."):
        loc_feats = spatial_extractor.get_features(lat, lon)
        
    elev = loc_feats.get('elevation', 0)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📍 Topografi Wilayah")
    st.sidebar.write(f"**Elevasi Tanah**: {elev:.1f} mdpl")
        
# --- LOGIKA OPERASI ---

if mode == "Real-time Monitoring":
    # 1. Fetch Weather
    weather_df = weather_fetcher.fetch_weather_data(lat=lat, lon=lon)
    
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
            lag1_date = d - datetime.timedelta(days=1)
            lag2_date = d - datetime.timedelta(days=2)
            lag3_date = d - datetime.timedelta(days=3)
            
            lags_lookup[d] = {
                'hujan_lag1': daily_sums.get(lag1_date, 0),
                'hujan_lag2': daily_sums.get(lag2_date, 0),
                'hujan_lag3': daily_sums.get(lag3_date, 0)
            }
            
        # 4. Predict Risk Series
        if model_pack:
            hourly_risk_df = model_utils.predict_hourly_series(model_pack, hourly_df, lags_lookup)
            
            # --- Current Status ---
            now = pd.Timestamp.now(tz=config.TIMEZONE)
            current_row = hourly_risk_df.iloc[(hourly_risk_df['time'] - now).abs().argsort()[:1]]
            
            if not current_row.empty:
                curr_prob = current_row['probability'].values[0]
                curr_status = current_row['status'].values[0]
                curr_rain_24h = current_row['rain_rolling_24h'].values[0]
                curr_tide = current_row['est'].values[0]
                
                # Initialize tma_benanga for real-time mode (REMOVED as per user request)
                # tma_benanga = 6.80 
                
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
                        # tma_benanga = sim_benanga
                        
                        # Prepare Input for Model V2
                        sim_input_for_model = {
                            "rain_sum_imputed": curr_rain_24h, 
                            "rain_intensity_max": current_row['precipitation'].values[0] if not current_row.empty else 0,
                            "soil_moisture_surface_mean": sim_soil,
                            "soil_moisture_root_mean": sim_soil, # Simplified assumption
                            "pasut_msl_max": curr_tide,
                            "hujan_lag1": 0,
                            "hujan_lag2": 0
                        }
                        
                        # Re-calculate Prediction
                        v2_cols = [
                            'rain_sum_imputed', 'rain_intensity_max', 
                            'soil_moisture_surface_mean', 'soil_moisture_root_mean', 
                            'pasut_msl_max', 'hujan_lag1', 'hujan_lag2'
                        ]
                        input_df = pd.DataFrame([sim_input_for_model])[v2_cols]
                        
                        try:
                            model = model_pack[0] 
                            prob_val = model.predict_proba(input_df)[0][1]
                            curr_prob = prob_val
                            curr_status = "Siaga" if curr_prob > config.THRESHOLD_FLOOD_PROBABILITY else "Aman"
                        except Exception as e:
                            logging.error(f"Prediction Error V2: {e}")
                            curr_prob = 0.0
                            curr_status = "Error"
                        
                        st.toast(f"Simulasi Aktif: Hujan {sim_rain}mm, Pasang {sim_tide}m, Soil {sim_soil}")
                
                # Logic for Soil Moisture Value
                if 'sim_input_for_model' in locals():
                    sm_val = sim_input_for_model['soil_moisture_surface_mean']
                else:
                    sm_val = current_row['soil_moisture_surface'].values[0] if 'soil_moisture_surface' in current_row else 0.45

                # --- VISUALISASI MENGGUNAKAN UI_COMPONENTS ---
                
                # 1. Executive Summary
                ui_components.render_executive_summary(curr_prob, curr_tide, hourly_risk_df)
                
                # 2. Metrics (Original + Gauge + Soil REPLACED BENANGA)
                tide_status = "Normal" if curr_tide < config.THRESHOLD_TIDE_PHYSICAL_DANGER else "Bahaya"
                ui_components.render_metrics(curr_status, curr_rain_24h, curr_tide, tide_status, sm_val)
                
                # Metric Soil Moisture (Standalone removed, integrated above)
                
                st.markdown("---")
                
                # Filter Date (Global for Dashboard)
                col_filter, _ = st.columns([1, 4])
                with col_filter:
                    selected_date = st.date_input("📅 Filter Tanggal Simulasi & Detail:", value=pd.Timestamp.now().date())
                
                # 3. Split Layout: Map & Charts
                col_main_1, col_main_2 = st.columns([1.5, 1])
                
                with col_main_1:
                    # Map Simulation
                    import json
                    if os.path.exists(config.RISK_MAP_PATH):
                        with open(config.RISK_MAP_PATH) as f:
                            geojson_data = json.load(f)
                        ui_components.render_map_simulation(geojson_data, hourly_risk_df, lat, lon, selected_date=selected_date)
                        
                with col_main_2:
                    # Filter Data based on Selected Date
                    filtered_df = hourly_risk_df[hourly_risk_df['time'].dt.date == selected_date]
                    
                    if filtered_df.empty:
                        st.info(f"Tidak ada data untuk tanggal {config.format_id_date(selected_date)}.")
                    else:
                        # Hourly Chart
                        ui_components.render_hourly_chart(filtered_df)
                        
                        # Detail Table in Expander
                        with st.expander("📄 Data Detail Per Jam"):
                             st.dataframe(filtered_df[['time', 'status', 'probability', 'rain_rolling_24h', 'est']], use_container_width=True)

                # 5. 7-Day Forecast (Simplified here, logic usually same)
                # 5. 7-Day Forecast (Simplified here, logic usually same)
                st.divider()
                st.subheader("📅 Prediksi 7 Hari Ke Depan")
                # Date filter moved to top
                
                hourly_df['date_only'] = hourly_df['time'].dt.date
                daily_groups = hourly_df.groupby('date_only')
                cols = st.columns(7)
                idx = 0
                for date_val, group in daily_groups:
                    if idx >= 7: break
                    daily_rain = group['precipitation'].sum()
                    daily_max_tide = group['est'].max()
                    
                    lag1 = lags_lookup.get(date_val, {}).get('hujan_lag1', 0)
                    lag2 = lags_lookup.get(date_val, {}).get('hujan_lag2', 0)
                    lag3 = lags_lookup.get(date_val, {}).get('hujan_lag3', 0)
                    
                    d_input = {
                        "hujan_hari_ini": daily_rain,
                        "durasi_hari_ini": (group['precipitation']>0).sum(),
                        "pasut_msl_max": daily_max_tide,
                        "pasut_slope": 0,
                        "hujan_lag1": lag1, "hujan_lag2": lag2, "hujan_lag3": lag3
                    }
                    d_status, d_prob, _, _, _ = model_utils.predict_flood(model_pack, d_input)
                    
                    with cols[idx]:
                        date_str = config.format_id_date(date_val)
                        st.markdown(f"**{date_str}**")
                        if d_status == "BAHAYA BANJIR":
                            st.error(f"{d_status}\n{d_prob*100:.0f}%")
                        else:
                            st.success(f"AMAN\n{d_prob*100:.0f}%")
                        st.caption(f"🌧️ {daily_rain:.1f}mm\n🌊 {daily_max_tide:.1f}m")
                    idx += 1

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
            "hujan_lag3": lag3
        }
        
        status, prob, thresh, akum, contrib = model_utils.predict_flood(model_pack, sim_input)
        
        st.metric("Status Prediksi", status)
        st.metric("Probabilitas", f"{prob*100:.2f}%")
        
        st.caption("Kontribusi Fitur:")
        st.json(contrib)

st.markdown("---")
st.caption("Supported by: Utide (Tidal Analysis), Open-Meteo (Weather), & Scikit-Learn (ML).")
