
# ⚙️ Configuration File
# Centralizes all hardcoded values, paths, and thresholds.

import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data-baru")
REF_DIR = os.path.join(BASE_DIR, "data-refactored")
DEM_DIR = os.path.join(BASE_DIR, "data", "dem")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "model_banjir_v8_10years.pkl")
TIDE_MODEL_PATH = os.path.join(MODELS_DIR, "tide_model_urs.pkl")
DATA_CSV_PATH = os.path.join(BASE_DIR, "data", "dataset_banjir_v8_10years.csv")
DATA_PARQUET_PATH = os.path.join(BASE_DIR, "data", "dataset_banjir_v8_10years.parquet")
TIDE_MODEL_PATH = os.path.join(MODELS_DIR, "tide_model_urs.pkl")
DEM_PATH = os.path.join(DEM_DIR, "DEMNAS_1915-13_v1.0.tif")
RISK_MAP_PATH = os.path.join(REF_DIR, "samarinda_risk_map_calculated.geojson")

# --- THRESHOLDS ---
# --- THRESHOLDS (DEPTH in CM) ---
THRESHOLD_DEPTH_WASPADA = 20.0 # cm
THRESHOLD_DEPTH_SIAGA = 50.0 # cm
THRESHOLD_DEPTH_AWAS = 100.0 # cm (~Pinggang Dewasa)

# Legacy Thresholds (will be deprecated)
THRESHOLD_FLOOD_PROBABILITY = 0.65 
THRESHOLD_TIDE_LOW_RISK = 3.0 # meters (Approx 20cm above ground)
THRESHOLD_TIDE_PHYSICAL_DANGER = 3.2 # meters (Deep > 40cm)
THRESHOLD_ELEVATION_LOW = 5.0 # meters
THRESHOLD_ELEVATION_SAFE = 10.0 # meters

# SOIL SATURATION (API)
# Faktor peluruhan air tanah. 0.85 - 0.90 untuk tanah lempung/gambut Samarinda.
API_DECAY_FACTOR = 0.85

def format_id_date(date_obj):
    """Format datetime object to Indonesian string (e.g., 'Senin, 08 Des')."""
    days = {
        'Mon': 'Senin', 'Tue': 'Selasa', 'Wed': 'Rabu', 'Thu': 'Kamis',
        'Fri': 'Jumat', 'Sat': 'Sabtu', 'Sun': 'Minggu'
    }
    months = {
        'Jan': 'Jan', 'Feb': 'Feb', 'Mar': 'Mar', 'Apr': 'Apr', 'May': 'Mei', 'Jun': 'Jun',
        'Jul': 'Jul', 'Aug': 'Agust', 'Sep': 'Sep', 'Oct': 'Okt', 'Nov': 'Nov', 'Dec': 'Des'
    }
    
    eng_day = date_obj.strftime('%a')
    eng_month = date_obj.strftime('%b')
    day = date_obj.strftime('%d')
    
    return f"{days.get(eng_day, eng_day)}, {day} {months.get(eng_month, eng_month)}"

# TIDE CORRECTION (Datum Separation)
# Nilai ini dikurangkan dari prediksi pasang agar match dengan elevasi DEM
# Est 3.0m (Gauge) - 2.8m = 0.2m (Relatif terhadap Tanah). Sangat aman.
TIDE_DATUM_OFFSET = 2.8 # meters

# TOLERANSI GENANGAN (Floor Tolerance)
# Air dianggap "Bahaya" (Merah) hanya jika kedalaman > nilai ini di atas tanah (P50).
# Asumsi: Lantai rumah rata-rata ditinggikan 0.5m dari tanah asli.
THRESHOLD_FLOOD_DEPTH_TOLERANCE = 0.5 # meters

# BENANGA DAM THRESHOLDS
THRESHOLD_BENANGA_SIAGA = 7.50 # meters (Level Waspada / Kuning)
THRESHOLD_BENANGA_BAHAYA = 8.00 # meters (Level Awas / Merah / Spillway Overflow)

# --- VISUAL STYLING ---
COLOR_PALETTE = {
    "bg_gradient": "linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)", # Deep Blue Sea
    "card_bg": "rgba(255, 255, 255, 0.05)",
    "text_primary": "#FFFFFF",
    "text_secondary": "#B0C4DE",
    "status_safe": "#00C853", # Brighter Green
    "status_warning": "#FFD600", # Vivid Yellow
    "status_danger": "#D50000", # Deep Red
    "status_critical": "#aa00ff" # Purple for Benanga Limit
}

# --- API ---
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Singapore" # WITA
LATITUDE = -0.5022
LONGITUDE = 117.1536

# Titik Pantau Banjir (Nama Location -> Lat, Lon, Runoff Coefficient)
# Runoff Coeff: 0.9 (Urban/Concrete), 0.7 (Residential), 0.5 (Green/Soil)
LOCATIONS = {
    "Simpang Lembuswana": (-0.472740, 117.143783, 0.90),
    "Simpang Sempaja": (-0.457889, 117.155432, 0.85),
    "Jalan Antasari": (-0.493922, 117.136894, 0.95), # Sangat Padat
    "Lempake (Hulu)": (-0.428987, 117.168341, 0.60), # Lebih Hijau
    "Kebon Agung": (-0.439812, 117.172938, 0.70)
}

# Lokasi Hulu (Catchment Area Hujan Kiriman) - Badak Baru / Kukar
UPSTREAM_LOCATIONS = {
    "Hulu Karang Mumus (Badak Baru)": (-0.352493, 117.228945) # Example coord for upstream
}

# Rata-rata waktu tempuh air dari hulu ke kota (Jam)
UPSTREAM_LAG_HOURS = 6

# --- TELEGRAM ALERTS ---
# Set via environment variables for security
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_IDS = os.getenv("TELEGRAM_CHAT_IDS", "").split(",")
TELEGRAM_ALERT_THRESHOLD = "SIAGA"  # Minimum level to trigger alert (WASPADA, SIAGA, AWAS)

# --- BMKG API ---
BMKG_API_URL = "https://api.bmkg.go.id/publik/prakiraan-cuaca"

# Complete list of 59 Kelurahan in Samarinda with ADM4 codes
# Format: Province.City.District.Village (64=Kaltim, 72=Samarinda)
SAMARINDA_KELURAHAN = {
    # Kecamatan Palaran (64.72.01)
    "Handil Bakti": {"code": "64.72.01.1001", "kecamatan": "Palaran"},
    "Rawa Makmur": {"code": "64.72.01.1002", "kecamatan": "Palaran"},
    "Bukuan": {"code": "64.72.01.1003", "kecamatan": "Palaran"},
    "Simpang Pasir": {"code": "64.72.01.1004", "kecamatan": "Palaran"},
    "Bantuas": {"code": "64.72.01.1005", "kecamatan": "Palaran"},
    
    # Kecamatan Samarinda Seberang (64.72.02)
    "Sungai Keledang": {"code": "64.72.02.1001", "kecamatan": "Samarinda Seberang"},
    "Baqa": {"code": "64.72.02.1002", "kecamatan": "Samarinda Seberang"},
    "Mesjid": {"code": "64.72.02.1003", "kecamatan": "Samarinda Seberang"},
    "Mangkupalas": {"code": "64.72.02.1009", "kecamatan": "Samarinda Seberang"},
    "Tenun": {"code": "64.72.02.1010", "kecamatan": "Samarinda Seberang"},
    "Gunung Panjang": {"code": "64.72.02.1011", "kecamatan": "Samarinda Seberang"},
    
    # Kecamatan Samarinda Ulu (64.72.03)
    "Teluk Lerong Ilir": {"code": "64.72.03.1001", "kecamatan": "Samarinda Ulu"},
    "Jawa": {"code": "64.72.03.1002", "kecamatan": "Samarinda Ulu"},
    "Air Putih": {"code": "64.72.03.1004", "kecamatan": "Samarinda Ulu"},
    "Sidodadi": {"code": "64.72.03.1005", "kecamatan": "Samarinda Ulu"},
    "Air Hitam": {"code": "64.72.03.1006", "kecamatan": "Samarinda Ulu"},
    "Dadi Mulya": {"code": "64.72.03.1007", "kecamatan": "Samarinda Ulu"},
    "Gunung Kelua": {"code": "64.72.03.1008", "kecamatan": "Samarinda Ulu"},
    "Bukit Pinang": {"code": "64.72.03.1009", "kecamatan": "Samarinda Ulu"},
    
    # Kecamatan Samarinda Ilir (64.72.04)
    "Pelita": {"code": "64.72.04.1001", "kecamatan": "Samarinda Ilir"},
    "Selili": {"code": "64.72.04.1002", "kecamatan": "Samarinda Ilir"},
    "Sidodamai": {"code": "64.72.04.1003", "kecamatan": "Samarinda Ilir"},
    "Sidomulyo": {"code": "64.72.04.1004", "kecamatan": "Samarinda Ilir"},
    "Sungai Dama": {"code": "64.72.04.1005", "kecamatan": "Samarinda Ilir"},
    
    # Kecamatan Samarinda Utara (64.72.05)
    "Lempake": {"code": "64.72.05.1001", "kecamatan": "Samarinda Utara"},
    "Tanah Merah": {"code": "64.72.05.1002", "kecamatan": "Samarinda Utara"},
    "Sungai Siring": {"code": "64.72.05.1003", "kecamatan": "Samarinda Utara"},
    "Budaya Pampang": {"code": "64.72.05.1004", "kecamatan": "Samarinda Utara"},
    "Sempaja Selatan": {"code": "64.72.05.1005", "kecamatan": "Samarinda Utara"},
    "Sempaja Utara": {"code": "64.72.05.1006", "kecamatan": "Samarinda Utara"},
    "Sempaja Barat": {"code": "64.72.05.1007", "kecamatan": "Samarinda Utara"},
    "Sempaja Timur": {"code": "64.72.05.1008", "kecamatan": "Samarinda Utara"},
    
    # Kecamatan Sungai Kunjang (64.72.06)
    "Teluk Lerong Ulu": {"code": "64.72.06.1001", "kecamatan": "Sungai Kunjang"},
    "Loa Bahu": {"code": "64.72.06.1002", "kecamatan": "Sungai Kunjang"},
    "Loa Bakung": {"code": "64.72.06.1003", "kecamatan": "Sungai Kunjang"},
    "Loa Buah": {"code": "64.72.06.1004", "kecamatan": "Sungai Kunjang"},
    "Karang Asam Ulu": {"code": "64.72.06.1005", "kecamatan": "Sungai Kunjang"},
    "Karang Asam Ilir": {"code": "64.72.06.1006", "kecamatan": "Sungai Kunjang"},
    "Karang Anyar": {"code": "64.72.06.1007", "kecamatan": "Sungai Kunjang"},
    
    # Kecamatan Sambutan (64.72.07)
    "Sambutan": {"code": "64.72.07.1001", "kecamatan": "Sambutan"},
    "Makroman": {"code": "64.72.07.1002", "kecamatan": "Sambutan"},
    "Sungai Kapih": {"code": "64.72.07.1003", "kecamatan": "Sambutan"},
    "Pulau Atas": {"code": "64.72.07.1004", "kecamatan": "Sambutan"},
    "Sindang Sari": {"code": "64.72.07.1005", "kecamatan": "Sambutan"},
    
    # Kecamatan Sungai Pinang (64.72.08)
    "Sungai Pinang Dalam": {"code": "64.72.08.1001", "kecamatan": "Sungai Pinang"},
    "Temindung Permai": {"code": "64.72.08.1002", "kecamatan": "Sungai Pinang"},
    "Mugirejo": {"code": "64.72.08.1003", "kecamatan": "Sungai Pinang"},
    "Bandara": {"code": "64.72.08.1004", "kecamatan": "Sungai Pinang"},
    "Gunung Lingai": {"code": "64.72.08.1005", "kecamatan": "Sungai Pinang"},
    
    # Kecamatan Samarinda Kota (64.72.09)
    "Karang Mumus": {"code": "64.72.09.1001", "kecamatan": "Samarinda Kota"},
    "Pelabuhan": {"code": "64.72.09.1002", "kecamatan": "Samarinda Kota"},
    "Pasar Pagi": {"code": "64.72.09.1003", "kecamatan": "Samarinda Kota"},
    "Bugis": {"code": "64.72.09.1004", "kecamatan": "Samarinda Kota"},
    "Sungai Pinang Luar": {"code": "64.72.09.1005", "kecamatan": "Samarinda Kota"},
    
    # Kecamatan Loa Janan Ilir (64.72.10)
    "Tani Aman": {"code": "64.72.10.1001", "kecamatan": "Loa Janan Ilir"},
    "Sengkotek": {"code": "64.72.10.1002", "kecamatan": "Loa Janan Ilir"},
    "Simpang Tiga": {"code": "64.72.10.1003", "kecamatan": "Loa Janan Ilir"},
    "Rapak Dalam": {"code": "64.72.10.1004", "kecamatan": "Loa Janan Ilir"},
    "Harapan Baru": {"code": "64.72.10.1005", "kecamatan": "Loa Janan Ilir"},
}

# Legacy mapping for backward compatibility with LOCATIONS config
BMKG_LOCATIONS = {
    "Simpang Lembuswana": "64.72.09.1001",  # Karang Mumus area
    "Simpang Sempaja": "64.72.05.1006",      # Sempaja Utara
    "Jalan Antasari": "64.72.03.1005",       # Sidodadi
    "Lempake (Hulu)": "64.72.05.1001",       # Lempake
    "Kebon Agung": "64.72.07.1001"           # Sambutan
}
# Default location for citywide forecast
BMKG_DEFAULT_ADM4 = "64.72.09.1001"

