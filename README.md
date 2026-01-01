# 🌊 Sistem Peringatan Dini Banjir Samarinda

**Dashboard Eksekutif & Analisis Prediktif Berbasis Machine Learning**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://prediksi-banjir-smr.streamlit.app/)

Sistem **Early Warning System (EWS)** cerdas untuk memprediksi risiko banjir di Kota Samarinda. Mengintegrasikan data cuaca *real-time*, prediksi pasang surut astronomis, dan model *Machine Learning*.

---

## 📋 Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| 🕐 **Monitoring Real-time** | Memantau risiko banjir setiap jam dengan status: AMAN, WASPADA, SIAGA, AWAS |
| 📅 **Prediksi 7 Hari** | Forecasting risiko untuk perencanaan operasional |
| 🎛️ **Simulasi What-If** | Mode manual untuk skenario ekstrem |
| 🗺️ **Heatmap Interaktif** | Visualisasi risiko berbasis peta dengan Layer Control |
| 🛰️ **Validasi Satelit** | Verifikasi genangan via Sentinel-1 (SAR) |
| 📡 **Radar Cuaca** | Konfirmasi curah hujan real-time via RainViewer |

---

## 📂 Struktur Proyek

```
prediksi_banjir/
├── 📁 data/                    # Dataset & logs
│   ├── dataset_banjir_*.csv    # Data historis banjir
│   └── feedback_log.csv        # User feedback
├── 📁 data-baru/               # Data cuaca terbaru
├── 📁 data-demhas/             # Data DEM (Elevasi)
├── 📁 data-refactored/         # GeoJSON & data olahan
├── 📁 models/                  # Model ML artifacts
│   ├── model_banjir_v2_advanced.pkl
│   ├── tide_model_urs.pkl
│   └── label_encoder.pkl
├── 📁 scripts/                 # Utility scripts
│   ├── train_model*.py         # Training scripts
│   ├── evaluate_model_v2.py    # Evaluation
│   └── zonal_stats.py          # GIS analysis
├── 📁 docs/                    # Dokumentasi
│   ├── Laporan_Validasi.md
│   └── SETUP_REALTIME_DATA.md
├── 📄 dashboard.py             # 🚀 Aplikasi Utama
├── 📄 config.py                # Konfigurasi Global
├── 📄 model_utils.py           # Inference Engine
├── 📄 ui_components.py         # Komponen UI
├── 📄 data_ingestion.py        # API Data Fetcher
├── 📄 feature_extraction.py    # Feature Engineering
├── 📄 sentinel_utils.py        # Sentinel-1 Integration
├── 📄 tide_utils.py            # Prediksi Pasang Surut
├── 📄 requirements.txt         # Dependencies
├── 📄 Dockerfile               # Container config
└── 📄 README.md                # Dokumentasi ini
```

---

## 🚀 Quick Start

### Persyaratan
- Python 3.8+
- Koneksi Internet

### Instalasi

```bash
# Clone repository
git clone https://github.com/amsopian22/prediksi-banjir-samarinda.git
cd prediksi-banjir-samarinda

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run dashboard.py
```

Akses di browser: `http://localhost:8501`

---

## 🧠 Metodologi

### Machine Learning Model (V2 Advanced)
**Algoritma:** Random Forest Classifier

**Input Features:**
| Feature | Deskripsi |
|---------|-----------|
| `rain_sum_imputed` | Akumulasi curah hujan harian (mm) |
| `rain_intensity_max` | Intensitas hujan terderas per jam (mm/h) |
| `pasut_msl_max` | Tinggi pasang tertinggi (meter) |
| `soil_moisture` | Kejenuhan tanah (0-1) |
| `hujan_lag1/2` | Memori hujan 1-2 hari sebelumnya |

### Prediksi Per-Jam (Rolling Window)
Setiap jam, sistem menghitung:
- Akumulasi hujan **24 jam terakhir**
- Tinggi pasang **saat ini**

---

## ⚙️ Konfigurasi

Edit `config.py` untuk menyesuaikan:

```python
THRESHOLD_FLOOD_PROBABILITY = 0.40  # Sensitivitas trigger
LOCATIONS = {...}                    # Titik pantau
API_DECAY_FACTOR = 0.85             # Faktor hidrologi
```

---

## 🛠️ Tech Stack

| Kategori | Teknologi |
|----------|-----------|
| Frontend | Streamlit, Plotly |
| ML | Scikit-Learn (Random Forest) |
| Data | Pandas, NumPy |
| Scientific | Utide (Harmonik Pasut), Rasterio |
| External API | Open-Meteo, RainViewer, Google Earth Engine |

---

## 📄 License

MIT License - Dikembangkan oleh **Tim Diskominfo Kota Samarinda**

*Untuk mendukung pengambilan keputusan berbasis data (Data-Driven Decision Making).*
