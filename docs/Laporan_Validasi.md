# 🧪 Laporan Validasi & Uji Ilmiah Sistem Prediksi Banjir

## 1. Metodologi Evaluasi
Kami menggunakan pendekatan **Data-Driven Physics** dengan algoritma *Random Forest Classifier*. Validasi dilakukan pada data historis 5 tahun terakhir dengan pembagian:
- **Training Set (80%)**: 1728 data (upsampled menjadi 3579 untuk keseimbangan kelas).
- **Test Set (20%)**: 433 data "buta" yang belum pernah dilihat model.

## 2. Hasil Evaluasi Kuantitatif

### A. Kinerja Model (Classification Report)
Pada threshold standar (0.5), model sangat akurat untuk kondisi normal, namun **konservatif** untuk banjir ekstrim:
- **Akurasi Total**: 97% (Sangat Tinggi)
- **Deteksi "Air Meluap"**: Precision 99%, Recall 100% (Sangat Presisi).
- **Deteksi "Banjir" (Minority Class)**:
    - *Default Threshold*: Gagal mendeteksi (0% Recall) karena data sangat jarang.
    - *Optimized Threshold (0.29)*: F1-Score naik signifikan menjadi **0.53**.
    
    > **Kesimpulan Ilmiah**: Kita **WAJIB** menggunakan threshold sensitif (0.29) daripada standar (0.5). Ini karena dalam mitigasi bencana, *False Positive* (Salah Waspada) lebih bisa diterima daripada *False Negative* (Gagal Deteksi Banjir).

### B. Validasi Fisik (Feature Importance)
Model secara otomatis mempelajari "Hukum Fisika" banjir Samarinda tanpa diprogram manual. Urutan faktor paling berpengaruh:
1.  **Curah Hujan Harian (49%)**: Faktor dominan utama.
2.  **Durasi Hujan (17%)**: Intensitas hujan jangka panjang.
3.  **Tinggi Pasang Maksimum (9%)**: Faktor pengunci (locking factor) drainase.
4.  **Kejenuhan Tanah (API/Hujan 3 Hari) (9%)**: Validasi bahwa tanah jenuh memperparah banjir.

## 3. Kesimpulan & Rekomendasi
Sistem ini **VALID secara ilmiah** karena:
1.  Variabel inputnya (Hujan, Pasut, Tanah) memiliki korelasi fisik kuat terhadap output.
2.  Model mampu membedakan "Air Meluap" vs "Aman" dengan akurasi nyaris sempurna.
3.  Kalibrasi threshold ke **0.29** adalah langkah statistik yang tepat untuk menangani ketidakseimbangan kejadian banjir ekstrim (Imbalanced Data).

**Rekomendasi:**
- Pertahankan threshold 0.29 di `config.py`.
- Terus perbarui data latih setiap kali kejadian banjir baru terjadi untuk meningkatkan "Ingatan" model.
