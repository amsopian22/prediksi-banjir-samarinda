# 🌊 Sistem Peringatan Dini Banjir Samarinda (Flood Early Warning System)

**Dashboard Eksekutif & Analisis Prediktif Berbasis Hybrid AI (V8)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://prediksi-banjir-smr.streamlit.app/) [![Run Tests](https://github.com/amsopian22/prediksi-banjir-samarinda/actions/workflows/ci.yml/badge.svg)](https://github.com/amsopian22/prediksi-banjir-samarinda/actions/workflows/ci.yml)

Sistem **Early Warning System (EWS)** cerdas untuk memprediksi risiko banjir di Kota Samarinda. Mengintegrasikan data cuaca *real-time* (10 tahun historis), prediksi pasang surut astronomis, dan model *Machine Learning* terbaru (V8 Hybrid) dengan antarmuka **Command Center**.

---

## 📋 Fitur Utama (V8 Next-Gen)

| Fitur | Deskripsi |
|-------|-----------|
| 🖥️ **Command Center UI** | Dashboard modern "Dark Glassmorphism" dengan indikator status real-time terpusat |
| 🤖 **Model Hybrid V8** | Dilatih dengan 10 tahun data (2015-2025) menggunakan algoritma XGBoost/Random Forest + SMOTE |
| 📡 **Telegram Alerts** | Notifikasi otomatis ke smartphone pejabat saat status mencapai **SIAGA** atau **AWAS** |
| 🛰️ **Validasi Satelit** | Verifikasi genangan banjir menggunakan citra satelit **Sentinel-1 (SAR)** |
| 🛡️ **Operational Fronts** | Analisa 4 Sisi: Curah Hujan (Rain), Hulu (Upstream), Pasang (Tide), Tanah (Soil) |
| 🗺️ **Peta Risiko Interaktif** | Visualisasi zona rawan banjir dinamis menggunakan GeoJSON |

---

## 📂 Struktur Proyek

```
prediksi_banjir/
├── 📁 data/                    # Dataset (DuckDB & CSV)
├── 📁 models/                  # Model ML artifacts
│   ├── model_banjir_v8_10years.pkl  # 🧠 MAIN MODEL (V8)
│   └── tide_model_urs.pkl           # Model Pasang Surut
├── 📁 scripts/                 # Utility scripts & Verification
│   ├── system_verification_suite.py # ✅ System Health Check
│   └── train_model_v8.py            # Training Pipeline
├── 📁 .github/workflows/       # CI/CD Automation
│   └── ci.yml                  # Automated Testing
├── 📄 dashboard.py             # 🚀 Aplikasi Utama (Streamlit)
├── 📄 config.py                # Konfigurasi Global
├── 📄 model_utils.py           # Inference Engine v8
├── 📄 ui_components.py         # Modern UI System
├── 📄 telegram_bot.py          # Bot Notifikasi
├── 📄 sentinel_utils.py        # Analisis Satelit
├── 📄 Dockerfile               # Production Container
└── 📄 requirements.txt         # Python Dependencies
```

---

## 🚀 Quick Start (Local)

### Persyaratan
- Python 3.10+
- Akun Telegram (untuk fitur alert - opsional)

### Instalasi

```bash
# Clone repository
git clone https://github.com/amsopian22/prediksi-banjir-samarinda.git
cd prediksi-banjir-samarinda

# (Opsional) Buat Virtual Environment
python -m venv venv
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run dashboard.py
```

Akses di browser: `http://localhost:8501`

---

## 🐳 Menjalankan dengan Docker

```bash
# Build Image
docker build -t banjir-samarinda:v8 .

# Run Container
docker run -p 8501:8501 banjir-samarinda:v8
```

---

## 🧠 Metodologi V8 (Hybrid)

### Machine Learning Core
**Model:** Random Forest Classifier & Regressor (Ensemble)
**Training Data:** 2015 - 2025 (10 Tahun Data Historis)

**Key Input Features:**
1.  **Meteorologi:** Curah hujan harian, intensitas per jam, akumulasi 3 & 7 hari (Antecedent Precipitation Index).
2.  **Hidrologi:** Tinggi muka air pasang (Tide), kelembaban tanah (Soil Moisture), aliran hulu (Upstream Flow).
3.  **Temporal:** Musiman (Rainy Season Flag), Siklus Bulan (Lunar Cycle for Tides).

### 3-Level Validation System
Setiap prediksi diverifikasi melalui 3 lapisan:
1.  **Statistical Check:** Ambang batas probabilitas (0.0 - 1.0).
2.  **Physical Check:** Validasi "Dry Day Safety Cap" (memastikan tidak ada alert banjir saat hari kering).
3.  **Remote Sensing:** Cross-check dengan data satelit Sentinel-1 jika tersedia.

---

## 📄 Pengembangan & Kontribusi

Proyek ini menggunakan **GitHub Actions** untuk Continuous Integration (CI).
Setiap *Push* atau *Pull Request* akan otomatis menjalankan:
1.  `pytest` suite (Unit Tests)
2.  `system_verification_suite.py` (Integration Tests)

---

## 📄 License

MIT License - Dikembangkan oleh **Tim Diskominfo Kota Samarinda** v/ **Bidang 4 (E-Gov)**.

*Data-Driven Decision Making for Flood Resilience.*
