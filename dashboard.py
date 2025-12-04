import streamlit as st
import pandas as pd
import joblib
import requests
import datetime
import plotly.graph_objects as go
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sistem Peringatan Dini Banjir Samarinda",
    page_icon="🌊",
    layout="wide"
)

# --- KONSTANTA ---
MODEL_PATH = "banjir_model_v1.pkl"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
LAT = -0.5022
LON = 117.1536

# --- FUNGSI UTILITAS ---

@st.cache_resource
def load_model():
    """Memuat model ML yang sudah dilatih."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model '{MODEL_PATH}' tidak ditemukan. Harap jalankan notebook terlebih dahulu untuk generate model.")
        return None
    return joblib.load(MODEL_PATH)

def fetch_weather_data():
    """Mengambil data cuaca real-time dari Open-Meteo."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": ["precipitation_sum", "precipitation_hours"],
        "timezone": "Asia/Singapore",
        "past_days": 2,
        "forecast_days": 1
    }
    try:
        response = requests.get(OPEN_METEO_URL, params=params)
        data = response.json()
        
        daily = data["daily"]
        
        # Mapping data
        hujan_lusa = daily["precipitation_sum"][0]
        hujan_kemarin = daily["precipitation_sum"][1]
        hujan_hari_ini = daily["precipitation_sum"][2]
        durasi_hari_ini = daily["precipitation_hours"][2]
        
        return {
            "hujan_hari_ini": hujan_hari_ini,
            "durasi_hari_ini": durasi_hari_ini,
            "hujan_kemarin": hujan_kemarin,
            "hujan_lusa": hujan_lusa,
            "tanggal": daily["time"][2]
        }
    except Exception as e:
        st.error(f"Gagal mengambil data cuaca: {e}")
        return None

def predict_flood(model_pack, input_data):
    """Melakukan prediksi menggunakan model."""
    model = model_pack["model"]
    threshold = model_pack["threshold"]
    
    # Feature Engineering
    akumulasi_3hari = input_data["hujan_hari_ini"] + input_data["hujan_kemarin"] + input_data["hujan_lusa"]
    
    # DataFrame sesuai format training
    df_input = pd.DataFrame([{
        'curah_hujan_mm': input_data["hujan_hari_ini"],
        'durasi_hujan_jam': input_data["durasi_hari_ini"],
        'hujan_kemarin': input_data["hujan_kemarin"],
        'hujan_akumulasi_3hari': akumulasi_3hari
    }])
    
    # Prediksi
    probabilitas = model.predict_proba(df_input)[0][1]
    status = "BAHAYA BANJIR" if probabilitas >= threshold else "AMAN"
    
    return status, probabilitas, threshold, akumulasi_3hari

# --- UI UTAMA ---

st.title("🌊 Sistem Peringatan Dini Banjir Samarinda")
st.markdown("Dashboard monitoring potensi banjir berbasis Machine Learning & Data Cuaca Real-time.")

# Load Model
model_pack = load_model()

if model_pack:
    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Pengaturan")
    mode = st.sidebar.radio("Mode Input Data:", ["Otomatis (Real-time)", "Manual (Simulasi)"])
    
    input_data = {}
    
    if mode == "Otomatis (Real-time)":
        st.sidebar.info("Mengambil data langsung dari Open-Meteo API.")
        weather_data = fetch_weather_data()
        
        if weather_data:
            input_data = weather_data
            st.sidebar.success(f"Data terupdate: {weather_data['tanggal']}")
            
    else: # Mode Manual
        st.sidebar.warning("Mode Simulasi Aktif.")
        input_data["hujan_hari_ini"] = st.sidebar.number_input("Curah Hujan Hari Ini (mm)", 0.0, 200.0, 50.0)
        input_data["durasi_hari_ini"] = st.sidebar.number_input("Durasi Hujan (jam)", 0.0, 24.0, 5.0)
        input_data["hujan_kemarin"] = st.sidebar.number_input("Curah Hujan Kemarin (mm)", 0.0, 200.0, 20.0)
        input_data["hujan_lusa"] = st.sidebar.number_input("Curah Hujan Lusa Kemarin (mm)", 0.0, 200.0, 10.0)

    # --- MAIN CONTENT ---
    if input_data:
        # Lakukan Prediksi
        status, prob, threshold, akumulasi = predict_flood(model_pack, input_data)
        
        # 1. Status Banner
        if status == "BAHAYA BANJIR":
            st.error(f"### 🚨 STATUS: {status}")
            st.markdown(f"**Probabilitas Banjir: {prob*100:.1f}%** (Threshold: {threshold*100:.1f}%)")
        else:
            st.success(f"### ✅ STATUS: {status}")
            st.markdown(f"**Probabilitas Banjir: {prob*100:.1f}%** (Threshold: {threshold*100:.1f}%)")
            
        st.divider()
        
        # 2. Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Hujan Hari Ini", f"{input_data['hujan_hari_ini']} mm")
        col2.metric("Durasi", f"{input_data['durasi_hari_ini']} jam")
        col3.metric("Hujan Kemarin", f"{input_data['hujan_kemarin']} mm")
        col4.metric("Akumulasi 3 Hari", f"{akumulasi:.1f} mm")
        
        # 3. Visualizations
        st.subheader("📊 Analisis Visual")
        
        col_chart, col_map = st.columns([2, 1])
        
        with col_chart:
            # Gauge Chart untuk Probabilitas
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilitas Banjir (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "red" if status == "BAHAYA BANJIR" else "green"},
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            st.plotly_chart(fig)
            
        with col_map:
            st.markdown("**Lokasi Pantauan: Samarinda**")
            
            # Size dynamic based on rainfall (in meters)
            # Base size 500m + (Rainfall * 20)
            # Example: 0mm -> 500m, 50mm -> 1500m
            rainfall_val = input_data['hujan_hari_ini']
            marker_size = 500 + (rainfall_val * 20)
            
            map_data = pd.DataFrame({
                'lat': [LAT], 
                'lon': [LON],
                'size': [marker_size],
                'color': [[255, 0, 0, 160]] # Red with transparency
            })
            
            # st.map automatically uses 'size' and 'color' columns if present
            st.map(map_data, zoom=11, size='size', color='color')

        # 4. Detail Table
        st.subheader("📋 Detail Data Input")
        detail_df = pd.DataFrame({
            "Parameter": ["Curah Hujan Hari Ini", "Durasi Hujan Hari Ini", "Curah Hujan Kemarin", "Curah Hujan Lusa Kemarin", "Total Akumulasi 3 Hari"],
            "Nilai": [
                f"{input_data['hujan_hari_ini']} mm",
                f"{input_data['durasi_hari_ini']} jam",
                f"{input_data['hujan_kemarin']} mm",
                f"{input_data['hujan_lusa']} mm",
                f"{akumulasi:.1f} mm"
            ],
            "Keterangan": [
                "Data H-0 (Prediksi/Aktual)",
                "Lama hujan terjadi",
                "Data H-1 (Lag 1)",
                "Data H-2 (Lag 2)",
                "Faktor penting penyebab banjir"
            ]
        })
        st.table(detail_df)

        # 5. Save & Show History
        st.subheader("📜 Riwayat Prediksi")
        
        # Save current prediction
        history_file = "prediction_history.csv"
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_record = pd.DataFrame([{
            "Waktu": current_time,
            "Status": status,
            "Probabilitas": f"{prob*100:.1f}%",
            "Hujan Hari Ini": input_data['hujan_hari_ini'],
            "Akumulasi 3 Hari": akumulasi
        }])
        
        if os.path.exists(history_file):
            history_df = pd.read_csv(history_file)
            # Check if the last record is identical to avoid duplicate logging on refresh
            if not history_df.empty:
                last_record = history_df.iloc[-1]
                # Simple check: if timestamp is very close (e.g. within same minute) and data same, skip
                # For simplicity in Streamlit (which reruns on interaction), we might just append.
                # But to avoid spamming on every interaction, we can check if the input data is exactly the same as last time?
                # Let's just append for now, user asked to "save information".
                # Better: Check if we just saved this data in this session state?
                pass
            
            # Append new record
            # Note: In a real app, we might want to trigger saving only on a button press or specific event.
            # Here, since it runs on every refresh, we need to be careful.
            # Let's use session state to track if we already saved this specific prediction.
            
            # Create a unique signature for the current input
            current_signature = f"{input_data['hujan_hari_ini']}_{input_data['durasi_hari_ini']}_{input_data['hujan_kemarin']}_{input_data['hujan_lusa']}"
            
            if 'last_saved_signature' not in st.session_state or st.session_state['last_saved_signature'] != current_signature:
                new_record.to_csv(history_file, mode='a', header=False, index=False)
                st.session_state['last_saved_signature'] = current_signature
                st.toast("Data prediksi tersimpan ke riwayat!", icon="💾")
                
        else:
            new_record.to_csv(history_file, index=False)
            st.session_state['last_saved_signature'] = f"{input_data['hujan_hari_ini']}_{input_data['durasi_hari_ini']}_{input_data['hujan_kemarin']}_{input_data['hujan_lusa']}"
            st.toast("File riwayat baru dibuat!", icon="🆕")

        # Show History Table
        if os.path.exists(history_file):
            history_df = pd.read_csv(history_file)
            # Sort by time desc
            history_df = history_df.sort_values(by="Waktu", ascending=False)
            st.dataframe(history_df, use_container_width=True)

else:
    st.warning("Silakan train model terlebih dahulu.")
