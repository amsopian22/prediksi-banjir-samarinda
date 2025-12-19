# 🌊 Sistem Peringatan Dini Banjir Samarinda (Flood Early Warning System)

**Dashboard Eksekutif & Analisis Prediktif Berbasis Machine Learning**

Proyek ini adalah sistem **Early Warning System (EWS)** cerdas yang dirancang untuk memprediksi risiko banjir di Kota Samarinda. Sistem ini mengintegrasikan data cuaca *real-time*, prediksi pasang surut astronomis, dan model *Machine Learning* untuk memberikan status kewaspadaan yang akurat kepada pengambil keputusan.

---

## 📋 Fitur Utama

1.  **Monitoring Real-time (Per Jam):**
    *   Memantau risiko banjir setiap jam berdasarkan kondisi hujan dan pasang laut terkini.
    *   Menampilkan status: **AMAN, WASPADA, SIAGA, AWAS**.
2.  **Prediksi 7 Hari Kedepan:**
    *   Forecasting risiko banjir untuk satu minggu ke depan untuk perencanaan operasional.
    *   Integrasi data ramalan cuaca global (Open-Meteo) yang dikalibrasi lokal.
3.  **Simulasi "What-If" Analysis:**
    *   Mode manual untuk mensimulasikan skenario ekstrem (Contoh: "Apa yang terjadi jika hujan 100mm + Pasang 2.5 meter?").
    *   Digunakan untuk validasi kebijakan dan analisis dampak.
4.  **Analisis Multi-Faktor:**
    *   Mempertimbangkan **Curah Hujan** (Intensitas & Durasi).
    *   Mempertimbangkan **Pasang Surut Air Laut** (Backwater Effect).
    *   Mempertimbangkan **Kelembaban Tanah** (Soil Moisture).
    *   Faktor koreksi lokal (Drainase & Topografi).
5.  **Validasi Hibrida (Satellite & Radar):**
    *   **Sentinel-1 (SAR):** Verifikasi genangan banjir aktual menggunakan citra satelit radar (tembus awan) via Google Earth Engine.
    *   **RainViewer (Radar Cuaca):** Konfirmasi curah hujan *real-time* menggunakan data radar meteorologi untuk memvalidasi prediksi model.

---

## 🧠 Metodologi Ilmiah

Sistem ini menggunakan pendekatan **Hybrid-Heuristic** yang menggabungkan statistik data historis dengan prinsip hidrologi.

### 1. Model Machine Learning (V2 Advanced)
Model inti menggunakan algoritma **Random Forest Classifier** yang dilatih menggunakan data kejadian banjir historis (2020-2025).

*   **Input Features:**
    *   `rain_sum_imputed`: Akumulasi curah hujan harian.
    *   `rain_intensity_max`: Intensitas hujan terderas per jam.
    *   `pasut_msl_max`: Tinggi muka air laut (Pasang Tertinggi).
    *   `soil_moisture`: Tingkat kejenuhan tanah (0-1).
    *   `hujan_lag1` & `hujan_lag2`: Memori hujan 1-2 hari sebelumnya.

### 2. Transformasi Harian ke Per-Jam (Rolling Window)
Meskipun model dilatih dengan data kejadian harian, dashboard dapat memprediksi risiko **setiap jam** menggunakan teknik *Sliding Window*:

> **Logika:** Setiap jam, sistem menghitung akumulasi hujan dalam **24 jam terakhir** dan menggabungkannya dengan tinggi pasang **saat ini**. Ini memungkinkan model mendeteksi banjir akibat hujan durasi panjang maupun hujan intensitas tinggi sesaat.

---

## 📂 Struktur Proyek

```bash
prediksi_banjir/
├── dashboard.py             # APLIKASI UTAMA (Frontend Streamlit)
├── model_utils.py           # Logika Inti & Inference Engine
├── data_ingestion.py        # Modul penarik data API (Weather & Tide)
├── config.py                # Konfigurasi Global (Thresholds, Lokasi)
├── train_model_v2.py        # Script pelatihan model ML
├── dataset_banjir_v2_advanced.csv  # Dataset latih (Historical Data)
├── requirements.txt         # Daftar library Python
├── model_banjir_v2_advanced.pkl    # File Model (Artifact)
└── README.md                # Dokumentasi ini
```

---

## 🚀 Cara Menjalankan

### Persyaratan
*   Python 3.8+
*   Koneksi Internet (untuk mengambil data cuaca real-time)

### Instalasi

1.  **Clone Repository / Masuk ke Direktori:**
    ```bash
    cd /path/to/prediksi_banjir
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan Aplikasi:**
    ```bash
    streamlit run dashboard.py
    ```

Aplikasi akan terbuka otomatis di browser pada alamat `http://localhost:8501`.

---

## ⚙️ Konfigurasi & Kalibrasi

Anda dapat mengubah parameter sensitivitas sistem melalui file `config.py`:

*   **`THRESHOLD_FLOOD_PROBABILITY`**: Batas probabilitas untuk trigger peringatan (Default: 0.40).
*   **`LOCATIONS`**: Daftar koordinat titik pantau (Lat/Long).
*   **`API_DECAY_FACTOR`**: Faktor peluruhan air tanah (Hidrologi).

---

## 🛠️ Tech Stack

*   **Frontend:** Streamlit
*   **Machine Learning:** Scikit-Learn (Random Forest)
*   **Data Processing:** Pandas, NumPy
*   **Scientific:** Utide (Analisis Harmonik Pasut)
*   **Data Source:** 
    *   **Open-Meteo API** (Weather Forecast)
    *   **Google Earth Engine** (Sentinel-1 SAR Imagery)
    *   **RainViewer API** (Real-time Weather Radar)

---

**Dikembangkan oleh Tim Diskominfo Kota Samarinda.**
*Untuk mendukung pengambilan keputusan berbasis data (Data-Driven Decision Making).*
