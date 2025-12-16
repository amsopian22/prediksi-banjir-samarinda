
# 🛠️ Panduan Konfigurasi Data Real-time

Panduan ini menjelaskan langkah teknis untuk menghubungkan dashboard dengan sumber data satelit dan radar secara **Real-time** dan **Legal**.

## 1. Google Earth Engine (Sentinel-1)

Untuk mengganti simulasi Sentinel-1 dengan data asli, Anda memerlukan akun Google Cloud.

**Langkah Konfigurasi:**
1.  **Daftar Akun**: Kunjungi [earthengine.google.com/signup](https://earthengine.google.com/signup/) (Gratis untuk riset/pemerintah).
2.  **Buat Project**: Di Google Cloud Console, buat project baru (misal: `banjir-samarinda-radar`).
3.  **Service Account**:
    -   Masuk ke **IAM & Admin > Service Accounts**.
    -   Buat Service Account baru.
    -   Download kunci JSON (`private-key.json`).
4.  **Install Library**:
    ```bash
    pip install earthengine-api google-auth
    ```
5.  **Update Dashboard**:
    -   Simpan file JSON di folder project.
    -   Ganti logika di `sentinel_utils.py` untuk memanggil:
        ```python
        import ee
        ee.Initialize(credentials=...)
        collection = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(region)...
        ```

## 2. Himawari-8/9 (Satelit Cuaca)

Untuk data awan real-time tiap 10 menit.

**Sumber Data**:
-   **JMA (Japan Meteorological Agency)**: Gratis via [WIS Portal](https://www.wis-jma.go.jp/).
-   **Open Access (AWS Public Dataset)**: Cara termudah.

**Langkah Integrasi:**
1.  Gunakan library `s3fs` untuk akses bucket AWS: `noaa-himawari8`.
2.  Ambil band `AHI-B13` (Infrared) untuk suhu puncak awan.
3.  Konversi nilai pixel menjadi suhu (°C). Jika < -70°C, berarti awan badai.

## 3. RainViewer (Radar Hujan) - SUDAH TERPASANG

Dashboard ini sekarang sudah menggunakan API Publik RainViewer (`https://api.rainviewer.com/public/weather-maps.json`).

**Cara Kerja:**
-   Sistem mengecek data radar global dalam 30 menit terakhir.
-   Jika tersedia, sistem akan menampilkan Badge **"TERKONFIRMASI RADAR"**.
-   **Upgrade (Opsional)**: Jika ingin resolusi tinggi (1km), Anda bisa menghubungi RainViewer untuk akses API Premium, namun versi Publik sudah cukup untuk validasi keberadaan hujan.

---
**Catatan**: Fitur "Hybrid Verification" yang baru dipasang sudah otomatis menggunakan RainViewer (Publik) sebagai cadangan jika Sentinel-1 tidak lewat.
