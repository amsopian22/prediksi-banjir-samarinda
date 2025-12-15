
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
    # 1. Fetch Weather (Cached Wrapper)
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_cached_weather(lat, lon):
        return weather_fetcher.fetch_weather_data(lat=lat, lon=lon)

    with st.spinner("Mengambil data cuaca terkini..."):
        weather_df = get_cached_weather(lat, lon)
    
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
                    st.sidebar.toast(f"✅ Simulasi Aktif: Hujan {curr_rain_24h}mm, Pasang {curr_tide}m")

                # Prepare Data for Full Assessment (Real-time or Simulated)
                # Getting soil moisture default
                current_soil = current_row.get('soil_moisture_surface', 0.45)
                # Use simulated soil if active, else real
                final_soil = sim_soil if st.session_state.get('sim_active') else current_soil if isinstance(current_soil, float) else 0.45

                input_data_pack = {
                    "rain_sum_imputed": curr_rain_24h,
                    "rain_intensity_max": current_row['precipitation'].values[0] if not current_row.empty else 0,
                    "soil_moisture_surface_mean": final_soil,
                    "soil_moisture_root_mean": final_soil,
                    "pasut_msl_max": curr_tide,
                    "hujan_lag1": 0, "hujan_lag2": 0 # Real-time simplification
                }
                
                # Get Full Assessment
                assessment = model_utils.predict_flood(model_pack, input_data_pack)
                
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
                
                curr_prob = assessment.get('probability', 0)
                curr_status = assessment.get('label', 'Unknown')

                # --- VISUALISASI MENGGUNAKAN UI_COMPONENTS ---
                
                # 1. Executive Summary (New Signature)
                ui_components.render_executive_summary(assessment)
                
                # 2. Key Metrics
                tide_status = "Bahaya" if curr_tide >= config.THRESHOLD_TIDE_PHYSICAL_DANGER else "Normal"
                ui_components.render_metrics(curr_status, curr_rain_24h, curr_tide, tide_status, final_soil)
                
                # 3. Risk Context (Why & How) -- NEW
                ui_components.render_risk_context(assessment)
                
                st.markdown("---")
                
                # Filter Date (Global for Dashboard)
                col_filter, _ = st.columns([1, 4])
                with col_filter:
                    selected_date = st.date_input("📅 Filter Tanggal Simulasi & Detail:", value=pd.Timestamp.now().date())
                
                # 4. Split Layout: Map & Charts
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
                
                cols = st.columns(7)
                idx = 0
                for date_val, group in future_groups:
                    if idx >= 7: break
                    daily_rain = group['precipitation'].sum()
                    daily_max_tide = group['est'].max()
                    
                    # Correct Lags from Lookup
                    lags = lags_lookup.get(date_val, {})
                    lag1 = lags.get('hujan_lag1', 0)
                    lag2 = lags.get('hujan_lag2', 0)
                    
                    # Actual Soil Moisture
                    soil_mean = group['soil_moisture_surface'].mean() if 'soil_moisture_surface' in group else 0.5

                    d_input = {
                        "rain_sum_imputed": daily_rain,
                        "rain_intensity_max": group['precipitation'].max(),
                        "soil_moisture_surface_mean": soil_mean,
                        "soil_moisture_root_mean": soil_mean, # Assumption for root
                        "pasut_msl_max": daily_max_tide,
                        "hujan_lag1": lag1, 
                        "hujan_lag2": lag2
                    }
                    
                    # Predict using new function
                    d_assess = model_utils.predict_flood(model_pack, d_input)
                    d_status = d_assess['label']
                    d_prob = d_assess['probability']
                    d_color = d_assess['color']
                    
                    # DEBUG: Inspect Input Data
                    with st.expander(f"Debug {date_val}"):
                         st.write(d_input)
                         st.write(f"Prob: {d_prob}")
                    
                    with cols[idx]:
                        date_str = config.format_id_date(date_val)
                        st.markdown(f"**{date_str}**")
                        
                        # Map color to UI style
                        status_class = "success"
                        if d_color == "yellow": status_class = "warning"
                        elif d_color == "orange": status_class = "error" # Streamlit has no orange alert
                        elif d_color == "red": status_class = "error"
                        
                        if status_class == "success":
                            st.success(f"{d_status}\n{d_prob*100:.0f}%")
                        elif status_class == "warning":
                            st.warning(f"{d_status}\n{d_prob*100:.0f}%")
                        else:
                            st.error(f"{d_status}\n{d_prob*100:.0f}%")
                            
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
            "hujan_lag3": lag3,
            # Params for V2 compatibility
            "rain_sum_imputed": hujan,
            "rain_intensity_max": hujan/durasi if durasi > 0 else 0,
            "soil_moisture_surface_mean": 0.5,
            "soil_moisture_root_mean": 0.5
        }
        
        assessment = model_utils.predict_flood(model_pack, sim_input)
        
        # Display Result using new components
        ui_components.render_executive_summary(assessment)
        ui_components.render_risk_context(assessment)
        
        st.divider()
        st.caption("Kontribusi Fitur:")
        st.json(assessment.get("contributions", {}))

st.markdown("---")
st.caption("Supported by: Utide (Tidal Analysis), Open-Meteo (Weather), & Scikit-Learn (ML).")
