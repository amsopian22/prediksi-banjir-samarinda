import requests
import json
from datetime import datetime

# URL API
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
MODEL_API_URL = "http://127.0.0.1:8000/predict" # Alamat FastAPI lokal Anda

def run_automation():
    print(f"[{datetime.now()}] 🤖 Memulai proses otomatis...")

    # 1. FETCH: Ambil Data dari Open-Meteo
    # Trik: parameter 'past_days=2' akan mengambil data hari ini + 2 hari ke belakang
    params = {
        "latitude": -0.5022,  # Koordinat Samarinda
        "longitude": 117.1536,
        "daily": ["precipitation_sum", "precipitation_hours"],
        "timezone": "Asia/Singapore", # WITA
        "past_days": 2,     # Ambil H-1 dan H-2
        "forecast_days": 1  # Ambil Hari Ini (H-0)
    }

    try:
        response = requests.get(OPEN_METEO_URL, params=params)
        data_meteo = response.json()
        
        # Cek jika data harian ada
        if "daily" not in data_meteo:
            print("❌ Gagal mendapatkan data daily dari Open-Meteo")
            return

        # 2. TRANSFORM: Mapping Data ke Format Model
        # Array 'daily' akan berisi 3 data: [H-2, H-1, Hari_Ini]
        daily = data_meteo["daily"]
        
        # Indeks Array:
        # 0 = Lusa Kemarin (H-2)
        # 1 = Kemarin (H-1)
        # 2 = Hari Ini (H-0)
        
        hujan_lusa = daily["precipitation_sum"][0]
        hujan_kemarin = daily["precipitation_sum"][1]
        hujan_hari_ini = daily["precipitation_sum"][2]
        durasi_hari_ini = daily["precipitation_hours"][2]

        print(f"📊 Data Cuaca Terkini:")
        print(f"   - Hujan Lusa (H-2)   : {hujan_lusa} mm")
        print(f"   - Hujan Kemarin (H-1): {hujan_kemarin} mm")
        print(f"   - Hujan Hari Ini     : {hujan_hari_ini} mm")

        # Susun Payload JSON sesuai yang diminta main.py
        payload = {
            "curah_hujan_mm": hujan_hari_ini,
            "durasi_hujan_jam": durasi_hari_ini,
            "hujan_kemarin_mm": hujan_kemarin,
            "hujan_lusa_mm": hujan_lusa
        }

        # 3. PREDICT: Kirim ke Model FastAPI
        res_model = requests.post(MODEL_API_URL, json=payload)
        
        if res_model.status_code == 200:
            result = res_model.json()
            print("\n✅ PREDIKSI SUKSES!")
            print(f"   Status: {result['prediction']}")
            print(f"   Probabilitas: {result['probability']*100:.1f}%")
            print(f"   Pesan: {result['detail']['pesan']}")
            
            # (Opsional) Disini bisa tambah kode kirim ke Telegram
            if result['prediction'] == "BAHAYA BANJIR":
                print("   🔔 MENGIRIM NOTIFIKASI TELEGRAM...")
                # send_telegram_alert(result['detail']['pesan'])
                
        else:
            print(f"❌ Model Error: {res_model.text}")

    except Exception as e:
        print(f"❌ Terjadi Kesalahan Sistem: {e}")

if __name__ == "__main__":
    run_automation()