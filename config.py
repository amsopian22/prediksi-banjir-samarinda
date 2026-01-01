
# ⚙️ Configuration File
# Centralizes all hardcoded values, paths, and thresholds.

import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data-baru")
REF_DIR = os.path.join(BASE_DIR, "data-refactored")
DEM_DIR = os.path.join(BASE_DIR, "data-demhas")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "model_banjir_v2_advanced.pkl")
TIDE_MODEL_PATH = os.path.join(MODELS_DIR, "tide_model_urs.pkl")
DEM_PATH = os.path.join(DEM_DIR, "DEMNAS_1915-13_v1.0.tif")
RISK_MAP_PATH = os.path.join(REF_DIR, "samarinda_risk_map_calculated.geojson")

# --- THRESHOLDS ---
THRESHOLD_FLOOD_PROBABILITY = 0.40 # Optimized manually (Prev: 0.80). Lowered to capture heavy rain events.
THRESHOLD_TIDE_LOW_RISK = 2.5 # meters (Mulai Meluap / Genangan Ringan)
THRESHOLD_TIDE_PHYSICAL_DANGER = 2.9 # meters (Bahaya / Genangan Dalam > 40cm)
THRESHOLD_ELEVATION_LOW = 5.0 # meters
THRESHOLD_ELEVATION_SAFE = 10.0 # meters

# SOIL SATURATION (API)
# Faktor peluruhan air tanah. 0.85 - 0.90 untuk tanah lempung/gambut Samarinda.
API_DECAY_FACTOR = 0.85

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
