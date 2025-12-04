from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# 1. Inisialisasi Aplikasi
app = FastAPI(
    title="Samarinda Flood Warning System API",
    description="API untuk memprediksi potensi banjir berdasarkan data curah hujan.",
    version="1.0"
)

# 2. Load Model saat Server Nyala
print("⏳ Loading model...")
try:
    model_pack = joblib.load("banjir_model_v1.pkl")
    model = model_pack["model"]
    THRESHOLD = model_pack["threshold"]
    FEATURE_NAMES = model_pack["features"]
    print(f"✅ Model loaded! Threshold set to: {THRESHOLD}")
except Exception as e:
    print(f"❌ Gagal load model: {e}")
    # Gunakan dummy untuk testing jika file pkl belum ada
    model = None 

# 3. Definisi Format Data Input (Data Validation)
class WeatherData(BaseModel):
    curah_hujan_mm: float
    durasi_hujan_jam: float
    hujan_kemarin_mm: float
    hujan_lusa_mm: float  # Dibutuhkan untuk menghitung rolling sum

# 4. Endpoint Prediksi
@app.post("/predict")
def predict_flood(data: WeatherData):
    if not model:
        raise HTTPException(status_code=500, detail="Model belum siap/tidak ditemukan.")

    # A. Feature Engineering on-the-fly
    # Ingat: Model butuh 'hujan_akumulasi_3hari', bukan 'hujan_lusa'
    # Rumus: Hujan Hari Ini + Kemarin + Lusa
    akumulasi_3hari = data.curah_hujan_mm + data.hujan_kemarin_mm + data.hujan_lusa_mm
    
    # Susun DataFrame sesuai urutan waktu training
    input_df = pd.DataFrame([{
        'curah_hujan_mm': data.curah_hujan_mm,
        'durasi_hujan_jam': data.durasi_hujan_jam,
        'hujan_kemarin': data.hujan_kemarin_mm,
        'hujan_akumulasi_3hari': akumulasi_3hari
    }])
    
    # B. Prediksi Probabilitas
    # predict_proba returns [[prob_0, prob_1]]
    prob_banjir = model.predict_proba(input_df)[0][1]
    
    # C. Keputusan berdasarkan Threshold Custom
    status = "BAHAYA BANJIR" if prob_banjir >= THRESHOLD else "AMAN"
    
    return {
        "prediction": status,
        "probability": round(prob_banjir, 4),
        "threshold_used": THRESHOLD,
        "detail": {
            "pesan": "Segera amankan barang berharga." if status == "BAHAYA BANJIR" else "Kondisi terkendali.",
            "curah_hujan_input": data.curah_hujan_mm
        }
    }

# 5. Endpoint Cek Kesehatan (Health Check)
@app.get("/")
def home():
    return {"status": "Server Berjalan", "service": "Flood Warning AI"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)