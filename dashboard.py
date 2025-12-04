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
    """Mengambil data cuaca real-time dari Open-Meteo untuk 7 hari kedepan."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": ["precipitation_sum", "precipitation_hours"],
        "timezone": "Asia/Singapore",
        "past_days": 2,
        "forecast_days": 7
    }
    try:
        response = requests.get(OPEN_METEO_URL, params=params)
        data = response.json()
        
        daily = data["daily"]
        forecast_list = []
        
        # Loop untuk 7 hari kedepan (Index 2 sampai 8)
        # Index 0: H-2, Index 1: H-1, Index 2: H-0 (Hari ini), ..., Index 8: H+6
        for i in range(7):
            idx_today = i + 2
            
            hujan_lusa = daily["precipitation_sum"][idx_today - 2]
            hujan_kemarin = daily["precipitation_sum"][idx_today - 1]
            hujan_hari_ini = daily["precipitation_sum"][idx_today]
            durasi_hari_ini = daily["precipitation_hours"][idx_today]
            tanggal = daily["time"][idx_today]
            
            forecast_list.append({
                "hujan_hari_ini": hujan_hari_ini,
                "durasi_hari_ini": durasi_hari_ini,
                "hujan_kemarin": hujan_kemarin,
                "hujan_lusa": hujan_lusa,
                "tanggal": tanggal
            })
            
        return forecast_list
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
    
    # XAI: Hitung Kontribusi Fitur
    # Contribution = Value * Coefficient
    feature_names = model_pack["features"]
    coefficients = model.coef_[0]
    
    contributions = {}
    for name, coef in zip(feature_names, coefficients):
        val = df_input[name].iloc[0]
        contributions[name] = val * coef
        
    # Sort contributions descending
    sorted_contributions = dict(sorted(contributions.items(), key=lambda item: item[1], reverse=True))
    
    return status, probabilitas, threshold, akumulasi_3hari, sorted_contributions

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
        weather_data_list = fetch_weather_data()
        
        if weather_data_list:
            # Ambil data hari ini (elemen pertama) untuk input utama
            input_data = weather_data_list[0]
            st.sidebar.success(f"Data terupdate: {input_data['tanggal']}")
            
    else: # Mode Manual
        st.sidebar.warning("Mode Simulasi Aktif.")
        input_data["hujan_hari_ini"] = st.sidebar.number_input("Curah Hujan Hari Ini (mm)", 0.0, 200.0, 50.0)
        input_data["durasi_hari_ini"] = st.sidebar.number_input("Durasi Hujan (jam)", 0.0, 24.0, 5.0)
        input_data["hujan_kemarin"] = st.sidebar.number_input("Curah Hujan Kemarin (mm)", 0.0, 200.0, 20.0)
        input_data["hujan_lusa"] = st.sidebar.number_input("Curah Hujan Lusa Kemarin (mm)", 0.0, 200.0, 10.0)

    # --- HUMAN-IN-THE-LOOP FEEDBACK ---
    st.sidebar.divider()
    st.sidebar.header("🙋‍♂️ Human-in-the-Loop")
    st.sidebar.caption("Bantu kami meningkatkan akurasi model. Apakah kondisi lapangan sesuai prediksi?")
    
    col_fb1, col_fb2 = st.sidebar.columns(2)
    
    report_status = None
    if col_fb1.button("🚨 Lapor Banjir"):
        report_status = "Banjir"
    
    if col_fb2.button("✅ Lapor Aman"):
        report_status = "Aman"
        
    if report_status:
        if input_data:
            # Hitung prediksi saat ini untuk log
            pred_status, pred_prob, _, _, _ = predict_flood(model_pack, input_data)
            
            feedback_data = {
                "Waktu": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Prediksi_Model": pred_status,
                "Probabilitas": f"{pred_prob*100:.1f}%",
                "Kondisi_Aktual": report_status,
                "Curah_Hujan_Hari_Ini": input_data.get('hujan_hari_ini', 0),
                "Keterangan": "Quick Report (Human-in-the-Loop)"
            }
            
            feedback_file = "feedback_log.csv"
            feedback_df = pd.DataFrame([feedback_data])
            
            if not os.path.exists(feedback_file):
                feedback_df.to_csv(feedback_file, index=False)
            else:
                feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
                
            st.sidebar.success(f"Laporan '{report_status}' tersimpan! Terima kasih.")
        else:
            st.sidebar.error("Data input belum tersedia.")
            
            feedback_file = "feedback_log.csv"
            feedback_df = pd.DataFrame([feedback_data])
            
            if not os.path.exists(feedback_file):
                feedback_df.to_csv(feedback_file, index=False)
            else:
                feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
                
            st.sidebar.success("Laporan terkirim! Terima kasih.")
        else:
            st.sidebar.error("Data input belum tersedia.")

    # --- MAIN CONTENT ---
    if input_data:
        # Lakukan Prediksi
        status, prob, threshold, akumulasi, contributions = predict_flood(model_pack, input_data)
        
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
            
            # --- XAI SECTION ---
            st.subheader("🔍 Analisis Faktor Penyebab")
            
            # Ambil top contributor
            top_feature = list(contributions.keys())[0]
            top_val = input_data.get(top_feature, 0)
            
            # Mapping nama fitur ke label user-friendly
            feature_labels = {
                'curah_hujan_mm': 'Curah Hujan Hari Ini',
                'durasi_hujan_jam': 'Durasi Hujan',
                'hujan_kemarin': 'Hujan Kemarin',
                'hujan_akumulasi_3hari': 'Tanah Jenuh (Akumulasi 3 Hari)'
            }
            
            st.info(f"**Kontributor Utama:** {feature_labels.get(top_feature, top_feature)}")
            
            # Tampilkan breakdown
            for feature, contrib in contributions.items():
                fname = feature_labels.get(feature, feature)
                fval = 0
                
                # Ambil nilai asli (perlu handling khusus karena input_data mungkin beda key dengan feature name di model)
                if feature == 'curah_hujan_mm': fval = input_data['hujan_hari_ini']
                elif feature == 'durasi_hujan_jam': fval = input_data['durasi_hari_ini']
                elif feature == 'hujan_kemarin': fval = input_data['hujan_kemarin']
                elif feature == 'hujan_akumulasi_3hari': fval = akumulasi
                
                # Label Kualitatif Sederhana
                qual_label = ""
                if "hujan" in feature or "akumulasi" in feature:
                    if fval > 100: qual_label = "(Sangat Tinggi)"
                    elif fval > 50: qual_label = "(Tinggi)"
                    elif fval > 20: qual_label = "(Sedang)"
                    else: qual_label = "(Rendah)"
                
                # Bar progress untuk kontribusi (normalisasi visual saja)
                # Kita pakai st.progress atau st.caption
                st.write(f"- **{fname}**: {fval:.1f} {qual_label}")
                # st.caption(f"Kontribusi Model: {contrib:.2f}") # Optional: tampilkan skor mentah
            
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

        # 4. 7-Day Forecast Section
        if mode == "Otomatis (Real-time)" and 'weather_data_list' in locals() and weather_data_list:
            st.divider()
            st.subheader("📅 Prediksi 7 Hari Kedepan")
            
            # Create columns for 7 days
            cols = st.columns(7)
            
            for i, day_data in enumerate(weather_data_list):
                with cols[i]:
                    # Predict for this day
                    d_status, d_prob, _, _, _ = predict_flood(model_pack, day_data)
                    
                    # Formatting date
                    date_obj = datetime.datetime.strptime(day_data['tanggal'], "%Y-%m-%d")
                    date_str = date_obj.strftime("%d %b")
                    day_name = date_obj.strftime("%A")
                    
                    # Color coding
                    status_color = "red" if d_status == "BAHAYA BANJIR" else "green"
                    
                    st.markdown(f"**{day_name}**")
                    st.markdown(f"_{date_str}_")
                    st.markdown(f":{status_color}[**{d_status}**]")
                    st.markdown(f"Prob: {d_status == 'BAHAYA BANJIR' and '**' or ''}{d_prob*100:.0f}%{d_status == 'BAHAYA BANJIR' and '**' or ''}")
                    st.caption(f"Hujan: {day_data['hujan_hari_ini']}mm")

        # 5. Detail Table
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
