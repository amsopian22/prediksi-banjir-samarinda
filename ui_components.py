
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import config
from datetime import datetime
import warnings

# Suppress SHAP warnings about TreeExplainer compatibility
warnings.filterwarnings('ignore', category=FutureWarning, module='shap')
warnings.filterwarnings('ignore', message='.*TreeExplainer.*')
warnings.filterwarnings('ignore', message='.*feature_perturbation.*')

def load_custom_css():
    """
    Injects custom CSS for ultra-modern premium dashboard design.
    Features: Advanced glassmorphism, vibrant gradients, smooth animations, depth layers.
    """
    st.markdown("""
        <style>
        /* ============================================
           🎨 ULTRA-MODERN FLOOD DASHBOARD v2.0
           ============================================ */
        
        /* Import Premium Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        
        /* ---- ROOT VARIABLES ---- */
        :root {
            --bg-primary: #0a1929;
            --bg-secondary: #0d2137;
            --bg-card: rgba(15, 35, 60, 0.7);
            --accent-cyan: #00d4ff;
            --accent-blue: #0099ff;
            --accent-green: #00ff88;
            --accent-yellow: #ffd700;
            --accent-red: #ff5252;
            --text-primary: #ffffff;
            --text-secondary: #b8c5d6;
            --text-muted: #6b7a8a;
        }
        
        /* ---- BASE RESET ---- */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 400;
        }
        
        /* ---- MAIN BACKGROUND (SIMPLE & VISIBLE) ---- */
        .stApp {
            background: linear-gradient(180deg, #0f1f35 0%, #1a2744 50%, #0f1f35 100%) !important;
            background-attachment: fixed !important;
        }
        
        /* Animated wave pattern overlay */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            z-index: 0;
        }
        
        /* ---- TYPOGRAPHY ---- */
        h1, h2, h3, h4 {
            color: #ffffff !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }
        
        p, span, div, label {
            color: #b8c5d6;
        }
        
        /* ---- HERO STATUS BANNER (SIMPLE & CLEAR) ---- */
        .hero-status-banner {
            background: linear-gradient(135deg, rgba(20, 40, 70, 0.95) 0%, rgba(15, 30, 55, 0.95) 100%);
            border-radius: 24px;
            padding: 48px 56px;
            margin-bottom: 30px;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }
        
        .status-glow-safe {
            box-shadow: 
                0 0 60px rgba(0, 255, 136, 0.25),
                0 0 100px rgba(0, 255, 136, 0.15),
                0 8px 32px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255,255,255,0.15);
            border: 2px solid rgba(0, 255, 136, 0.6);
            animation: glow-safe 3s ease-in-out infinite;
        }
        
        @keyframes glow-safe {
            0%, 100% { box-shadow: 0 0 60px rgba(0, 255, 136, 0.25), 0 0 100px rgba(0, 255, 136, 0.15); }
            50% { box-shadow: 0 0 80px rgba(0, 255, 136, 0.35), 0 0 120px rgba(0, 255, 136, 0.25); }
        }
        
        .status-glow-warning {
            box-shadow: 
                0 0 60px rgba(255, 193, 7, 0.2),
                0 0 100px rgba(255, 193, 7, 0.1),
                inset 0 1px 0 rgba(255,255,255,0.1);
            border: 2px solid rgba(255, 193, 7, 0.6);
            animation: pulse-warning 2s ease-in-out infinite;
        }
        
        .status-glow-danger {
            box-shadow: 
                0 0 80px rgba(255, 82, 82, 0.3),
                0 0 120px rgba(255, 82, 82, 0.15),
                inset 0 1px 0 rgba(255,255,255,0.1);
            border: 2px solid rgba(255, 82, 82, 0.7);
            animation: pulse-danger 1.5s ease-in-out infinite;
        }
        
        @keyframes pulse-warning {
            0%, 100% { box-shadow: 0 0 60px rgba(255, 193, 7, 0.2), 0 0 100px rgba(255, 193, 7, 0.1); }
            50% { box-shadow: 0 0 80px rgba(255, 193, 7, 0.35), 0 0 120px rgba(255, 193, 7, 0.2); }
        }
        
        @keyframes pulse-danger {
            0%, 100% { box-shadow: 0 0 80px rgba(255, 82, 82, 0.3), 0 0 120px rgba(255, 82, 82, 0.15); }
            50% { box-shadow: 0 0 100px rgba(255, 82, 82, 0.5), 0 0 150px rgba(255, 82, 82, 0.25); }
        }
        
        .status-label {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 8px;
            opacity: 0.7;
        }
        
        .status-text {
            font-size: 4rem;
            font-weight: 900;
            letter-spacing: 6px;
            text-transform: uppercase;
            margin: 0;
            line-height: 1;
            text-shadow: 0 2px 20px rgba(0, 0, 0, 0.5);
        }
        
        .status-text-safe { color: #00ff88; text-shadow: 0 0 30px rgba(0, 255, 136, 0.5); }
        .status-text-warning { color: #ffc107; text-shadow: 0 0 30px rgba(255, 193, 7, 0.5); }
        .status-text-danger { color: #ff5252; text-shadow: 0 0 30px rgba(255, 82, 82, 0.5); }
        
        .status-subtitle {
            font-size: 1.1rem;
            color: #8899aa;
            margin-top: 15px;
            font-weight: 400;
        }
        
        /* ---- GLASSMORPHISM CARDS ---- */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 24px;
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
        }
        
        /* ---- METRIC CARDS (CLEAR & VISIBLE) ---- */
        .metric-card-modern {
            background: linear-gradient(135deg, rgba(25, 45, 75, 0.92) 0%, rgba(20, 35, 60, 0.95) 100%);
            border-radius: 20px;
            padding: 32px 28px;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.15);
            transition: all 0.3s ease;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }
        
        .metric-card-modern:hover {
            transform: translateY(-6px);
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5);
            border-color: rgba(255, 255, 255, 0.25);
        }
        
        .metric-card-modern::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            border-radius: 24px 24px 0 0;
            opacity: 0.9;
        }
        
        .metric-card-rain::before { 
            background: linear-gradient(90deg, #00d4ff 0%, #0099ff 50%, #00b8e6 100%);
            box-shadow: 0 2px 12px rgba(0, 212, 255, 0.4);
        }
        .metric-card-tide::before { 
            background: linear-gradient(90deg, #ffd700 0%, #ff9500 50%, #ffb800 100%);
            box-shadow: 0 2px 12px rgba(255, 215, 0, 0.4);
        }
        .metric-card-soil::before { 
            background: linear-gradient(90deg, #00ff88 0%, #00cc6a 50%, #00e67a 100%);
            box-shadow: 0 2px 12px rgba(0, 255, 136, 0.4);
        }
        
        .metric-icon {
            font-size: 3rem;
            margin-bottom: 16px;
            filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3));
            transition: transform 0.3s ease;
        }
        
        .metric-card-modern:hover .metric-icon {
            transform: scale(1.1) rotate(5deg);
        }
        
        .metric-label {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: #8899aa;
            margin-bottom: 8px;
        }
        
        .metric-value-large {
            font-size: 3.2rem;
            font-weight: 800;
            color: #ffffff;
            line-height: 1;
            text-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
            letter-spacing: -0.02em;
        }
        
        .metric-unit {
            font-size: 1rem;
            font-weight: 400;
            color: #6b7a8a;
            margin-left: 4px;
        }
        
        .metric-status {
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 12px;
            padding: 6px 16px;
            border-radius: 24px;
            display: inline-block;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        
        .metric-status-good { background: rgba(0, 255, 136, 0.15); color: #00ff88; }
        .metric-status-warning { background: rgba(255, 193, 7, 0.15); color: #ffc107; }
        .metric-status-danger { background: rgba(255, 82, 82, 0.15); color: #ff5252; }
        
        /* Sparkline container */
        .metric-sparkline {
            position: absolute;
            bottom: 15px;
            right: 15px;
            opacity: 0.3;
            width: 80px;
            height: 40px;
        }
        
        /* ---- LOCATION SIDEBAR CARD ---- */
        .location-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 20px;
        }
        
        .location-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .location-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }
        
        .weather-mini-stat {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .weather-mini-stat:last-child {
            border-bottom: none;
        }
        
        /* ---- BENTO GRID LAYOUT ---- */
        .bento-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 25px;
        }
        
        @media (max-width: 768px) {
            .bento-grid {
                grid-template-columns: 1fr;
            }
        }
        
        /* ---- MAP CONTAINER ---- */
        .map-container {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 20px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            position: relative;
            overflow: hidden;
        }
        
        .map-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .map-title {
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: #8899aa;
        }
        
        .map-legend {
            display: flex;
            gap: 15px;
            font-size: 0.75rem;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .legend-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        
        /* ---- CHART CONTAINER ---- */
        .chart-container {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 20px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .chart-title {
            font-size: 1rem;
            font-weight: 600;
            color: #ffffff;
        }
        
        .chart-period-selector {
            display: flex;
            gap: 8px;
        }
        
        .period-btn {
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 500;
            background: rgba(255, 255, 255, 0.05);
            color: #8899aa;
            border: none;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .period-btn.active {
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
        }
        
        /* ---- STREAMLIT OVERRIDES ---- */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: white !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #8899aa !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1f30 0%, #0a1929 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #b8c5d6;
        }
        
        /* Input fields */
        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
        }

        /* ---- SLIDER REFINEMENT ---- */
        .stSlider [data-baseweb="slider"] {
            margin-top: 15px;
            margin-bottom: 25px;
        }
        
        /* Garis Slider (Track) */
        .stSlider [data-baseweb="slider"] > div:first-child > div:first-child {
            background-color: rgba(255, 255, 255, 0.1) !important;
            height: 6px !important;
            border-radius: 3px !important;
        }
        
        /* Garis Aktif (Progress) */
        .stSlider [data-baseweb="slider"] [data-testid="stSliderTickBar"] + div > div {
            background: linear-gradient(90deg, #00d4ff, #0099ff) !important;
            height: 6px !important;
        }

        /* Angka di atas thumb (Current Value) */
        div[data-testid="stThumbValue"] {
            background-color: transparent !important;
            color: #00d4ff !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
        }

        /* Label Min/Max (Tepi Slider) */
        [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {
            background-color: transparent !important;
            color: #6b7a8a !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.75rem !important;
            top: 10px !important;
        }
        
        /* Menghilangkan border/background liar pada angka slider */
        .stSlider span {
            background: transparent !important;
            border: none !important;
        }
        
        /* ---- MODERN TABS (PREMIUM PILL DESIGN) ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: rgba(15, 30, 60, 0.45) !important;
            padding: 10px !important;
            border-radius: 18px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            margin-bottom: 30px !important;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 14px !important;
            padding: 12px 28px !important;
            background-color: transparent !important;
            border: 1px solid transparent !important;
            color: #8899aa !important;
            font-weight: 600 !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            font-size: 0.75rem !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #ffffff !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(0, 212, 255, 0.2) !important;
            color: #00d4ff !important;
            font-weight: 800 !important;
            box-shadow: 
                0 4px 15px rgba(0, 0, 0, 0.3),
                0 0 20px rgba(0, 212, 255, 0.15) !important;
            border: 1px solid rgba(0, 212, 255, 0.4) !important;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }

        /* Sidebar & Input Fixes */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 1.5rem;
        }

        .stSelectbox label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #8899aa !important;
            margin-bottom: 8px;
        }
        
        /* Divider */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            margin: 30px 0;
        }
        
        /* Legacy compatibility classes */
        .hero-banner {
            padding: 40px;
            border-radius: 24px;
            text-align: center;
            margin-bottom: 30px;
            background: linear-gradient(135deg, rgba(10, 25, 41, 0.95) 0%, rgba(13, 33, 55, 0.95) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .hero-status-text {
            font-size: 3.5rem;
            font-weight: 800;
            letter-spacing: 4px;
            text-transform: uppercase;
            margin: 0;
            line-height: 1.1;
        }
        
        .hero-subtext {
            font-size: 1.1rem;
            color: #8899aa;
            margin-top: 15px;
        }
        
        .grid-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .grid-item {
            background: linear-gradient(135deg, rgba(20, 30, 48, 0.9) 0%, rgba(15, 25, 40, 0.95) 100%);
            border-radius: 20px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .grid-header {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: #8899aa;
            margin-bottom: 12px;
        }
        
        .grid-value {
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
        }
        
        .grid-sub {
            font-size: 0.85rem;
            color: #6b7a8a;
            margin-top: 8px;
        }
        
        .pulse-red {
            animation: pulse-danger 1.5s ease-in-out infinite;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        /* ---- SECTION HEADERS ---- */
        .section-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 40px 0 25px 0;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .section-icon {
            font-size: 1.8rem;
        }
        
        .section-title {
            font-size: 1.5rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #ffffff;
            margin: 0;
        }

        /* ---- BENTO GRID LAYOUT ---- */
        .bento-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        </style>
    """, unsafe_allow_html=True)



def render_status_reference():
    """
    Displays the reference legend for operational statuses with dynamic colors.
    """
    st.markdown("""
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 25px; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
<div style="text-align: center;">
<div style="color: #2ecc71; font-weight: 800; font-size: 0.9rem; margin-bottom: 4px;">✅ KONDUSIF</div>
<div style="color: #a0a0a0; font-size: 0.75rem;">Aman Terkendali</div>
</div>
<div style="text-align: center;">
<div style="color: #f1c40f; font-weight: 800; font-size: 0.9rem; margin-bottom: 4px;">⚠️ SIAGA III</div>
<div style="color: #a0a0a0; font-size: 0.75rem;">Waspada (Persiapan)</div>
</div>
<div style="text-align: center;">
<div style="color: #e67e22; font-weight: 800; font-size: 0.9rem; margin-bottom: 4px;">📢 SIAGA II</div>
<div style="color: #a0a0a0; font-size: 0.75rem;">Darurat (Evakuasi)</div>
</div>
<div style="text-align: center;">
<div style="color: #e74c3c; font-weight: 800; font-size: 0.9rem; margin-bottom: 4px;">🚨 SIAGA I</div>
<div style="color: #a0a0a0; font-size: 0.75rem;">Mobilisasi</div>
</div>
</div>
""", unsafe_allow_html=True)


def render_command_center_hero(assessment: dict, validation: dict = None):
    """
    Modern Hero Section with Neon Glow Effects.
    Premium dashboard design with dynamic status indicators.
    """
    import pandas as pd
    import config
    
    level = assessment.get("level", "UNKNOWN")
    label = assessment.get("label", "NORMAL")
    depth_cm = assessment.get("depth_cm", 0)
    
    # Map to display values
    status_text = "STATUS: AMAN"
    glow_class = "status-glow-safe"
    text_class = "status-text-safe"
    status_icon = "✓"
    
    if level == "WASPADA":
        status_text = "STATUS: WASPADA"
        glow_class = "status-glow-warning"
        text_class = "status-text-warning"
        status_icon = "⚠"
    elif level == "SIAGA":
        status_text = "STATUS: SIAGA"
        glow_class = "status-glow-danger"
        text_class = "status-text-danger"
        status_icon = "!"
    elif level == "AWAS":
        status_text = "STATUS: AWAS"
        glow_class = "status-glow-danger"
        text_class = "status-text-danger"
        status_icon = "🚨"

    # Subtitle with reasoning
    reasoning = assessment.get("reasoning", "Kondisi normal, tidak ada ancaman banjir terdeteksi.")
    
    # Get current time in WITA for initial display
    now_wita = pd.Timestamp.now(tz=config.TIMEZONE)
    update_time_initial = now_wita.strftime("%d %b %Y, %H:%M:%S WITA")
    
    st.markdown(f"""
<div class="hero-status-banner {glow_class}">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 30px;">
        <div style="flex: 2; min-width: 300px;">
            <div class="status-label">SISTEM PERINGATAN DINI BANJIR</div>
            <h1 class="status-text {text_class}">{status_text}</h1>
            <div class="status-subtitle">
                {status_icon} {reasoning}
            </div>
        </div>
        <div style="flex: 1; min-width: 200px; text-align: right;">
            <div style="font-size: 0.75rem; letter-spacing: 2px; color: #6b7a8a; text-transform: uppercase; margin-bottom: 8px;">Estimasi Genangan</div>
            <div style="font-size: 3rem; font-weight: 800; color: #ffffff; line-height: 1;">{depth_cm:.0f}<span style="font-size: 1.2rem; font-weight: 400; color: #6b7a8a;"> cm</span></div>
        </div>
    </div>
    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); text-align: right;">
        <span style="font-size: 0.8rem; color: #8899aa; letter-spacing: 0.5px;">
            ⏰ <span id="live-clock-hero" style="color: #00d4ff; font-weight: 600;">{update_time_initial}</span>
        </span>
    </div>
</div>

<script>
function updateLiveClock() {{
    const now = new Date();
    
    // WITA is UTC+8
    const witaOffset = 8 * 60; // minutes
    const localOffset = now.getTimezoneOffset(); // minutes from UTC
    const witaTime = new Date(now.getTime() + (witaOffset + localOffset) * 60000);
    
    // Format: DD MMM YYYY, HH:MM:SS WITA
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'];
    const day = String(witaTime.getDate()).padStart(2, '0');
    const month = months[witaTime.getMonth()];
    const year = witaTime.getFullYear();
    const hours = String(witaTime.getHours()).padStart(2, '0');
    const minutes = String(witaTime.getMinutes()).padStart(2, '0');
    const seconds = String(witaTime.getSeconds()).padStart(2, '0');
    
    const timeString = `${{day}} ${{month}} ${{year}}, ${{hours}}:${{minutes}}:${{seconds}} WITA`;
    
    const clockElement = document.getElementById('live-clock-hero');
    if (clockElement) {{
        clockElement.textContent = timeString;
    }}
}}

// This script is now empty as the live clock is handled by st.components.html
</script>
""", unsafe_allow_html=True)
    
    # Inject live clock JavaScript using st.components.html for better reliability
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function() {
        function updateClocks() {
            const now = new Date();
            const witaOffset = 8 * 60;
            const localOffset = now.getTimezoneOffset();
            const witaTime = new Date(now.getTime() + (witaOffset + localOffset) * 60000);
            
            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'];
            const day = String(witaTime.getDate()).padStart(2, '0');
            const month = months[witaTime.getMonth()];
            const year = witaTime.getFullYear();
            const hours = String(witaTime.getHours()).padStart(2, '0');
            const minutes = String(witaTime.getMinutes()).padStart(2, '0');
            const seconds = String(witaTime.getSeconds()).padStart(2, '0');
            
            const timeString = `${day} ${month} ${year}, ${hours}:${minutes}:${seconds} WITA`;
            
            // Update both clocks in the parent document
            const heroEl = window.parent.document.getElementById('live-clock-hero');
            const metricsEl = window.parent.document.getElementById('live-clock-metrics');
            
            if (heroEl) heroEl.textContent = timeString;
            if (metricsEl) metricsEl.textContent = timeString;
        }
        
        updateClocks();
        setInterval(updateClocks, 1000);
    })();
    </script>
    """, height=0)


def render_operational_fronts(weather: dict, upstream: dict, ocean: dict, spatial: dict):
    """
    Modern 3-Metric Cards: Curah Hujan, Pasang Surut, Kelembaban Tanah.
    Features gradient top borders and clean visual hierarchy.
    """
    # 1. Rain Data
    rain_val = weather.get('rain_24h', 0)
    upstream_val = upstream.get('rain_recent', 0)
    rain_status = "Ringan"
    rain_status_class = "metric-status-good"
    if rain_val > 20:
        rain_status = "Sedang"
        rain_status_class = "metric-status-warning"
    if rain_val > 50:
        rain_status = "Lebat"
        rain_status_class = "metric-status-danger"
         
    # 2. Tide Data
    tide_val = ocean.get('tide_max', 0)
    tide_status = "Stabil"
    tide_status_class = "metric-status-good"
    if tide_val > 2.5:
        tide_status = "Pasang Tinggi"
        tide_status_class = "metric-status-warning"
    if tide_val > 3.0:
        tide_status = "Overflow"
        tide_status_class = "metric-status-danger"

    # 3. Soil Data
    soil_val = spatial.get('soil_moisture', 0)
    # Convert to percentage for display
    soil_pct = int(soil_val * 100) if soil_val <= 1 else int(soil_val)
    soil_status = "Normal"
    soil_status_class = "metric-status-good"
    if soil_val > 0.5:
        soil_status = "Lembab"
        soil_status_class = "metric-status-warning"
    if soil_val > 0.7:
        soil_status = "Jenuh"
        soil_status_class = "metric-status-danger"
        
    # Use st.columns to render the three metrics separately to avoid HTML rendering issues
    st.markdown('<div class="bento-grid">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
    <div class="metric-card-modern metric-card-rain">
        <div class="metric-icon">💧</div>
        <div class="metric-label">CURAH HUJAN</div>
        <div class="metric-value-large">{rain_val:.1f}<span class="metric-unit">mm</span></div>
        <div class="metric-status {rain_status_class}">{rain_status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
    <div class="metric-card-modern metric-card-tide">
        <div class="metric-icon">🌊</div>
        <div class="metric-label">PASANG SURUT</div>
        <div class="metric-value-large">{tide_val:.1f}<span class="metric-unit">m</span></div>
        <div class="metric-status {tide_status_class}">{tide_status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
    <div class="metric-card-modern metric-card-soil">
        <div class="metric-icon">🌱</div>
        <div class="metric-label">KELEMBABAN TANAH</div>
        <div class="metric-value-large">{soil_pct}<span class="metric-unit">%</span></div>
        <div class="metric-status {soil_status_class}">{soil_status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data update timestamp info bar with live clock
    import pandas as pd
    import config
    now_wita = pd.Timestamp.now(tz=config.TIMEZONE)
    update_time_initial = now_wita.strftime("%d %b %Y, %H:%M:%S WITA")
    
    st.markdown(f"""
    <div style="margin-top: 15px; padding: 12px 20px; background: rgba(0, 212, 255, 0.05); border-left: 3px solid #00d4ff; border-radius: 8px;">
        <span style="font-size: 0.85rem; color: #b8c5d6;">
            📡 <strong>Waktu Real-Time</strong> • <span id="live-clock-metrics" style="color: #00d4ff; font-weight: 600;">{update_time_initial}</span>
        </span>
    </div>
    
    <script>
    function updateMetricsClock() {{
        const now = new Date();
        
        // WITA is UTC+8
        const witaOffset = 8 * 60;
        const localOffset = now.getTimezoneOffset();
        const witaTime = new Date(now.getTime() + (witaOffset + localOffset) * 60000);
        
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'];
        const day = String(witaTime.getDate()).padStart(2, '0');
        const month = months[witaTime.getMonth()];
        const year = witaTime.getFullYear();
        const hours = String(witaTime.getHours()).padStart(2, '0');
        const minutes = String(witaTime.getMinutes()).padStart(2, '0');
        const seconds = String(witaTime.getSeconds()).padStart(2, '0');
        
        const timeString = `${{day}} ${{month}} ${{year}}, ${{hours}}:${{minutes}}:${{seconds}} WITA`;
        
        const clockElement = document.getElementById('live-clock-metrics');
        if (clockElement) {{
            clockElement.textContent = timeString;
        }}
    }}
    
    updateMetricsClock();
    setInterval(updateMetricsClock, 1000);
    </script>
    """, unsafe_allow_html=True)


def render_bmkg_kelurahan_data():
    """
    Renders BMKG weather data per kelurahan from DuckDB.
    Shows forecast for selected kelurahan with weather description cards.
    """
    import config
    from database_manager import get_db
    from data_ingestion import BMKGFetcher
    
    st.subheader("🌦️ Prakiraan Cuaca BMKG per Kelurahan")
    st.caption("Data dari API BMKG (api.bmkg.go.id) - Diperbarui setiap 3 jam")
    
    # Kelurahan selector grouped by kecamatan
    kelurahan_list = list(config.SAMARINDA_KELURAHAN.keys())
    
    # Group by kecamatan for better UX
    kecamatan_groups = {}
    for kel, info in config.SAMARINDA_KELURAHAN.items():
        kec = info["kecamatan"]
        if kec not in kecamatan_groups:
            kecamatan_groups[kec] = []
        kecamatan_groups[kec].append(kel)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Kecamatan filter
        kecamatan_options = ["Semua Kecamatan"] + sorted(kecamatan_groups.keys())
        selected_kec = st.selectbox("📍 Filter Kecamatan:", kecamatan_options, key="bmkg_kec_select")
        
    with col2:
        # Kelurahan selector based on kecamatan
        if selected_kec == "Semua Kecamatan":
            kel_options = sorted(kelurahan_list)
        else:
            kel_options = sorted(kecamatan_groups.get(selected_kec, []))
        
        selected_kel = st.selectbox("🏘️ Pilih Kelurahan:", kel_options, key="bmkg_kel_select")
    
    if selected_kel:
        kel_info = config.SAMARINDA_KELURAHAN.get(selected_kel)
        adm4_code = kel_info["code"]
        kecamatan = kel_info["kecamatan"]
        
        st.markdown(f"**Kode ADM4:** `{adm4_code}` | **Kecamatan:** {kecamatan}")
        
        # Try to get cached data from DB first
        db = get_db()
        cached_df = db.get_bmkg_weather(kelurahan_code=adm4_code, hours=24)
        
        if cached_df.empty:
            # Fetch live data
            st.info("⏳ Mengambil data BMKG...")
            fetcher = BMKGFetcher()
            try:
                df = fetcher.fetch_weather_data(adm4_code=adm4_code)
                if not df.empty:
                    db.log_bmkg_weather(df, adm4_code, selected_kel, kecamatan)
                    cached_df = df
            except Exception as e:
                st.warning(f"Gagal mengambil data: {e}")
        
        if not cached_df.empty:
            st.markdown("---")
            
            # Normalize column names: database uses 'timestamp', live fetch uses 'date'
            if 'timestamp' in cached_df.columns and 'date' not in cached_df.columns:
                cached_df = cached_df.rename(columns={'timestamp': 'date'})
            
            # Current/nearest forecast
            now_df = cached_df.head(4)
            
            cols = st.columns(4)
            for i, (_, row) in enumerate(now_df.iterrows()):
                with cols[i]:
                    time_col = row.get('date') if 'date' in row.index else row.get('timestamp')
                    # Safely format time string manually to avoid NotImplementedError on some platforms
                    if hasattr(time_col, 'hour') and hasattr(time_col, 'minute'):
                        time_str = f"{time_col.hour:02d}:{time_col.minute:02d}"
                    else:
                        time_str = str(time_col)[11:16] if len(str(time_col)) >= 16 else str(time_col)[:5]
                    weather = row.get('weather_desc', 'N/A')
                    precip = row.get('precipitation', 0)
                    temp = row.get('temperature', 'N/A')
                    humidity = row.get('humidity', 'N/A')
                    
                    icon = "☀️"
                    if "Hujan" in str(weather):
                        icon = "🌧️" if "Lebat" in str(weather) else "🌦️"
                    elif "Berawan" in str(weather):
                        icon = "⛅"
                    elif "Mendung" in str(weather):
                        icon = "☁️"
                    
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 15px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
                        <div style="font-size: 2rem;">{icon}</div>
                        <div style="font-weight: 600; color: #fff;">{time_str}</div>
                        <div style="font-size: 0.8rem; color: #a0a0a0;">{weather}</div>
                        <div style="margin-top: 8px;"><span style="color: #5DADEC;">💧 {precip:.1f}mm</span></div>
                        <div style="font-size: 0.75rem; color: #808495;">🌡️ {temp}°C | 💨 {humidity}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with st.expander("📋 Lihat Prakiraan Lengkap", expanded=False):
                display_df = cached_df[['date', 'weather_desc', 'precipitation', 'temperature', 'humidity', 'wind_speed']].copy()
                display_df.columns = ['Waktu', 'Cuaca', 'Curah Hujan (mm)', 'Suhu (°C)', 'Kelembaban (%)', 'Angin (km/h)']
                st.dataframe(display_df, use_container_width=True)
            
            st.caption(f"🕐 Data terakhir: {cached_df['date'].max()}")
        else:
            st.warning("⚠️ Belum ada data BMKG untuk kelurahan ini.")
            if st.button("🔄 Fetch Data BMKG", key="bmkg_fetch_btn"):
                st.rerun()


def render_decision_support(geojson: dict, risk_df: pd.DataFrame, lat: float, lon: float, date_val=None):
    """
    Tabbed Interface for Decision Support: Map (Target), Chart (Timing), Forecast (Future).
    """
    # Modern Section Header
    st.markdown("""
        <div class="section-header">
            <span class="section-icon">🎯</span>
            <h2 class="section-title">Pendukung Keputusan Operasional</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Custom Tabs with cleaner labels
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Peta Operasi", 
        "Tren Waktu", 
        "Data BMKG",
        "Monitor Hulu",
        "Explainability"
    ])
    
    with tab1:
        # Primary map with RainViewer integration
        render_map_simulation(geojson, risk_df, lat, lon, date_val)
        
    with tab2:
        render_hourly_chart(risk_df)
        
    with tab3:
        render_bmkg_kelurahan_data()
    
    with tab4:
        st.info("Fitur Monitor Grafik Hulu Khusus (Placeholder untuk Integrasi AWS Bedrock/Camera)")
        # Simple stats for now
        st.write("Data Curah Hujan Hulu (6 Jam Terakhir):")
        # Logic to be connected in dashboard.py if needed, for now placeholders
    
    with tab5:
        render_shap_explanation(risk_df)
        
# ---------------- LEGACY FUNCTIONS (KEPT FOR COMPATIBILITY UNTIL SWAP) ----------------

def render_executive_summary(assessment: dict, **kwargs):
    # ... (code preserved below)

    """
    Renders the Executive Summary section as a Hero Banner (Ultra-Modern Style).
    """
    # Extract Data
    import pandas as pd
    level = assessment.get("level", "UNKNOWN")
    label = assessment.get("label", "Unknown")
    depth_cm = assessment.get("depth_cm", 0)
    reasoning = assessment.get("reasoning", "Tidak ada data.")
    
    # Map to display values matching render_command_center_hero
    status_text = f"STATUS: {label}"
    glow_class = "status-glow-safe"
    text_class = "status-text-safe"
    status_icon = "✓"
    
    if level == "WASPADA":
        glow_class = "status-glow-warning"
        text_class = "status-text-warning"
        status_icon = "⚠"
    elif level == "SIAGA":
        glow_class = "status-glow-danger"
        text_class = "status-text-danger"
        status_icon = "📢"
    elif level == "AWAS":
        glow_class = "status-glow-danger"
        text_class = "status-text-danger"
        status_icon = "🚨"

    # Get current time
    now_wita = pd.Timestamp.now(tz=config.TIMEZONE)
    update_time = now_wita.strftime("%d %b %Y, %H:%M:%S WITA")

    st.markdown(f"""
    <div class="hero-status-banner {glow_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 30px;">
            <div style="flex: 2; min-width: 300px;">
                <div class="status-label">HASIL SIMULASI BANJIR</div>
                <h1 class="status-text {text_class}">{status_text}</h1>
                <div class="status-subtitle">
                    {status_icon} {reasoning}
                </div>
            </div>
            <div style="flex: 1; min-width: 200px; text-align: right;">
                <div style="font-size: 0.75rem; letter-spacing: 2px; color: #6b7a8a; text-transform: uppercase; margin-bottom: 8px;">Estimasi Genangan</div>
                <div style="font-size: 3rem; font-weight: 800; color: #ffffff; line-height: 1;">{depth_cm:.0f}<span style="font-size: 1.2rem; font-weight: 400; color: #6b7a8a;"> cm</span></div>
            </div>
        </div>
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); text-align: right;">
            <span style="font-size: 0.8rem; color: #8899aa; letter-spacing: 0.5px;">
                ⏰ Generated: <span style="color: #00d4ff; font-weight: 600;">{update_time}</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Render Validation Badge if exists
    validation = assessment.get('validation_data') or kwargs.get('validation')
    if validation and validation.get('status') == 'CONFIRMED':
        val_label = validation.get('label', 'TERKONFIRMASI')
        val_detail = validation.get('detail', '')
        val_color = validation.get('color', '#2ecc71')
        
        st.markdown(f"""
            <div style="margin-top: 20px; text-align: center;">
                <span style="background: {val_color}33; border: 1px solid {val_color}; color: #e0e0e0; padding: 8px 16px; border-radius: 20px; font-size: 0.9rem;">
                    🛰️ <b>{val_label}</b> • {val_detail}
                </span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_risk_context(assessment: dict):
    """
    Displays the 'Why' (Reasoning) and 'Action' (Recommendation) in a structured way.
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="glass-card" style="margin-top: 40px;">
            <h3 style="margin-top:0; margin-bottom: 20px;">🔍 Analisis Penyebab</h3>
            <p style="font-size: 1.1rem; font-weight: 500; color: #ffcc80; margin-bottom: 12px;">{assessment.get('main_factor', '-')}</p>
            <p style="font-size: 0.9rem; color: #b0bec5; line-height: 1.5;">{assessment.get('reasoning', '-')}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Process newlines in recommendation for better formatting
        rec_text = assessment.get('recommendation', '-').replace('\n', '<br><br>')
        st.markdown(f"""
        <div class="glass-card" style="margin-top: 40px;">
            <h3 style="margin-top:0; margin-bottom: 20px;">📋 Rekomendasi Tindakan</h3>
            <p style="font-size: 1rem; color: #e0e0e0; line-height: 1.6;">{rec_text}</p>
        </div>
        """, unsafe_allow_html=True)


# ... (render_hourly_chart ... fetch_radar_timestamp ... render_map_simulation)

# We need to jump to render_map_simulation to fix the slider

def render_metrics(curr_status: str, total_rain_24h: float, curr_tide: float, tide_status: str, sm_val: float):
    """
    Renders key metrics using custom HTML cards.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    def metric_card(title, value, subtext, border_color="rgba(255,255,255,0.1)"):
        return f"""
        <div class="metric-card" style="background: linear-gradient(135deg, {border_color} 0%, rgba(255,255,255,0.01) 100%);">
            <div class="metric-inner">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{subtext}</div>
            </div>
        </div>
        """
        
    # Logic for Dynamic Colors
    rain_color = config.COLOR_PALETTE["status_warning"] if total_rain_24h > 50 else "rgba(255,255,255,0.1)"
    tide_color = config.COLOR_PALETTE["status_danger"] if tide_status == "Bahaya" else "rgba(255,255,255,0.1)"
    
    # Soil logic
    soil_status = "Jenuh Air" if sm_val > 0.5 else "Normal"
    soil_color = config.COLOR_PALETTE["status_warning"] if sm_val > 0.5 else "rgba(255,255,255,0.1)"
        
    with col1:
        st.markdown(metric_card("Status Teknis", curr_status, "AI Prediction"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Curah Hujan (24h)", f"{total_rain_24h:.1f} <span style='font-size:1rem'>mm</span>", "Akumulasi Harian", rain_color), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("Tinggi Pasang", f"{curr_tide:.2f} <span style='font-size:1rem'>m</span>", tide_status, tide_color), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("Kelembaban Tanah", f"{sm_val:.2f} <span style='font-size:1rem'>m³/m³</span>", soil_status, soil_color), unsafe_allow_html=True)

def render_hourly_chart(hourly_risk_df: pd.DataFrame):
    """
    Renders the Plotly chart for hourly risk.
    """
    from plotly.subplots import make_subplots

    st.divider()
    st.subheader("📉 Grafik Tren Terpadu (48 Jam)")
    
    # Create Subplots: Row 1 = Rain & Tide, Row 2 = Flood Risk
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.1,
                        subplot_titles=("Curah Hujan & Pasang Surut", "Prediksi Tinggi Genangan (cm)"),
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
    
    # --- ROW 1: Rain (Bar) & Tide (Line) ---
    # 1. Rain (Bar)
    fig.add_trace(go.Bar(
        x=hourly_risk_df['time'],
        y=hourly_risk_df['precipitation'],
        name='Curah Hujan (mm)',
        marker_color='#5DADEC',
        opacity=0.6
    ), row=1, col=1, secondary_y=False)
    
    # 2. Tide (Line)
    fig.add_trace(go.Scatter(
        x=hourly_risk_df['time'],
        y=hourly_risk_df['est'],
        name='Tinggi Pasang (m)',
        line=dict(color='#FFD700', width=3)
    ), row=1, col=1, secondary_y=True)
    
    # Critical Tide Threshold (Dashed Line)
    fig.add_hline(y=config.THRESHOLD_TIDE_PHYSICAL_DANGER, line_dash="dash", line_color="red", 
                  annotation_text=f"Batas Bahaya ({config.THRESHOLD_TIDE_PHYSICAL_DANGER}m)", 
                  annotation_position="top right", row=1, col=1, secondary_y=True)

    # --- ROW 2: Flood Depth (Area) ---
    fig.add_trace(go.Scatter(
        x=hourly_risk_df['time'],
        y=hourly_risk_df['depth_cm'],
        name='Tinggi Genangan (cm)',
        fill='tozeroy',
        mode='lines',
        line=dict(color='#ff5252')
    ), row=2, col=1)
    
    # Logic Threshold
    fig.add_hline(y=config.THRESHOLD_DEPTH_WASPADA, line_dash="dot", line_color="yellow", annotation_text="Waspada (20cm)", row=2, col=1)
    fig.add_hline(y=config.THRESHOLD_DEPTH_SIAGA, line_dash="dash", line_color="orange", annotation_text="Siaga (50cm)", row=2, col=1)

    # --- LAYOUT ---
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", y=1.1, x=0),
        margin=dict(t=50, b=50, l=50, r=50),
        hovermode="x unified"
    )
    
    # Y-Axis Labels
    fig.update_yaxes(title_text="Hujan (mm)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Pasang (m)", row=1, col=1, secondary_y=True, range=[0, 4.2])
    fig.update_yaxes(title_text="Genangan (cm)", row=2, col=1, range=[0, 150])
    
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_radar_timestamp():
    """Fetch the latest available radar timestamp and host from RainViewer API."""
    import requests
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        url = "https://api.rainviewer.com/public/weather-maps.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            host = data.get("host", "https://tilecache.rainviewer.com")
            # Get the very latest past timestamp
            if "radar" in data and "past" in data["radar"] and len(data["radar"]["past"]) > 0:
                latest = data["radar"]["past"][-1]
                logger.info(f"RainViewer radar loaded: timestamp={latest['time']}")
                return host, latest["time"]
        else:
            logger.warning(f"RainViewer API returned status {response.status_code}")
    except Exception as e:
        logger.warning(f"RainViewer API error: {e}")
    return None

def render_map_simulation(geojson_data: dict, hourly_risk_df: pd.DataFrame, lat: float, lon: float, selected_date=None):
    """
    Renders the Dynamic Inundation Map with Time Slider using Heatmap (Density).
    """
    import os
    if not geojson_data:
        st.warning("Data GeoJSON peta tidak tersedia.")
        return

    # Helper to calculate centroid from GeoJSON geometry
    def get_centroid(geometry):
        try:
            coords = []
            if geometry['type'] == 'Polygon':
                coords = geometry['coordinates'][0] # Outer ring
            elif geometry['type'] == 'MultiPolygon':
                coords = geometry['coordinates'][0][0] # First polygon outer ring
            
            if coords:
                # Simple arithmetic mean of vertices
                lons = [p[0] for p in coords]
                lats = [p[1] for p in coords]
                return sum(lats)/len(lats), sum(lons)/len(lons)
        except Exception:
            return None, None
        return None, None

    # Prepare Data features with Centroids
    feats = []
    for f in geojson_data['features']:
        props = f['properties']
        c_lat, c_lon = get_centroid(f['geometry'])
        if c_lat and c_lon:
            props['lat_center'] = c_lat
            props['lon_center'] = c_lon
            feats.append(props)

    df_map = pd.DataFrame(feats)
    
    st.divider()
    st.subheader("🗺️ Peta Simulasi & Heatmap Risiko")
    st.info("💡 **Petunjuk**: Peta ini menggunakan **Heatmap**. Area berwarna **Gelap** menunjukkan lokasi dengan akumulasi risiko banjir tertinggi (Cekungan Air). Gunakan slider untuk melihat perubahan seiring pasang surut.")

    # --- Dynamic Slider for Tide Simulation ---
    if selected_date:
        # Filter for specific date
        target_start = pd.to_datetime(selected_date).tz_localize(hourly_risk_df['time'].dt.tz)
        target_end = target_start + pd.Timedelta(days=1)
        future_tide_df = hourly_risk_df[(hourly_risk_df['time'] >= target_start) & (hourly_risk_df['time'] < target_end)]
    else:
        # Fallback to next 48h
        now = datetime.now(tz=hourly_risk_df['time'].dt.tz) if not hourly_risk_df.empty else datetime.now()
        now_floor = now.replace(minute=0, second=0, microsecond=0)
        future_tide_df = hourly_risk_df[hourly_risk_df['time'] >= now_floor].head(48) 
    
    if not future_tide_df.empty:
        # Create timestamp map for slider
        time_options = future_tide_df['time'].dt.strftime('%d %b %H:%M').tolist()
        
        # Determine Default Value (Current Hour)
        default_idx = 0
        if not selected_date:
            now_dt = datetime.now(tz=hourly_risk_df['time'].dt.tz) if not hourly_risk_df.empty else datetime.now()
            current_hour = now_dt.replace(minute=0, second=0, microsecond=0)
            current_hour_str = current_hour.strftime('%d %b %H:%M')
            
            if current_hour_str in time_options:
                default_idx = time_options.index(current_hour_str)
        
        # Layout for controls
        col_ctrl1, col_ctrl2 = st.columns([3, 1])
        with col_ctrl1:
            # Use regular slider with index to avoid 'values' property conflict
            slider_idx = st.slider(
                "⏳ **Pilih Waktu Simulasi**:", 
                min_value=0,
                max_value=len(time_options) - 1,
                value=default_idx,
                format=f"",  # Hide default number display
                key='map_simulation_slider'
            )
            selected_time_str = time_options[slider_idx]
            
            # Display selected time below slider
            st.caption(f"**Waktu Terpilih:** {selected_time_str}")
        
        # Get tide level for selected time
        selected_idx = future_tide_df.index[future_tide_df['time'].dt.strftime('%d %b %H:%M') == selected_time_str][0]
        selected_row = future_tide_df.loc[selected_idx]
        sim_tide_level = selected_row['est']
        
        # Calculate Trend
        tide_trend = 0
        if selected_idx > future_tide_df.index[0]:
             prev_level = future_tide_df.loc[selected_idx - 1, 'est'] if (selected_idx - 1) in future_tide_df.index else sim_tide_level
             tide_trend = sim_tide_level - prev_level
        elif selected_idx < future_tide_df.index[-1]:
             next_level = future_tide_df.loc[selected_idx + 1, 'est']
             tide_trend = sim_tide_level - next_level

        # Determine Arrow, Text, and Color
        if abs(tide_trend) < 0.05:
            arrow = "—"  # Em dash for stable
            color = "#808495"  # Grey
            text = "Stabil"
        elif tide_trend > 0:
            arrow = "↑"  # UP arrow
            color = "#ff4b4b"  # Red
            text = "Pasang Naik"
        else:
            arrow = "↓"  # DOWN arrow
            color = "#00c853"  # Green
            text = "Surut"

        # Display Current Tide Context
        with col_ctrl2:
            st.markdown(f"""
            <div style="text-align: center;">
                <p style="color: #808495; font-size: 0.75rem; margin: 0 0 0.1rem 0; font-weight: 400;">Tinggi Pasut</p>
                <p style="font-size: 1.75rem; font-weight: 700; color: white; margin: 0; line-height: 1.2;">{sim_tide_level:.2f} m</p>
                <p style="color: {color}; font-size: 0.85rem; margin: 0.1rem 0 0 0; font-weight: 500;">{arrow} {text}</p>
            </div>
            """, unsafe_allow_html=True)

        # --- Calculate Vulnerability Intensity (0 - 1) ---
        # 1.0 = BAHAYA (HITAM) - Tenggelam
        # 0.7 = SIAGA (MERAH) - Risiko Tinggi
        # 0.3 = WASPADA (KUNING) - Risiko Rendah
        # 0.0 = AMAN (TRANSPARAN)
        
        # Get Rain for Context
        sim_rain = selected_row.get('rain_rolling_24h', 0)
        
        def get_intensity(row):
            elev = row['mean_elev']
            adj_tide = sim_tide_level - config.TIDE_DATUM_OFFSET
            depth = adj_tide - elev
            
            # Base Logic
            if elev < adj_tide:
                # OVERFLOW CONDITION
                if sim_rain < 10.0:
                    # Dry Day Dampener: High Tide without Heavy Rain usually just fills channels
                    return 0.4, "WASPADA (PASANG)", f"Genangan {depth*100:.0f} cm"
                else:
                    # Wet Day: True Flood Risk
                    return 1.0, "BAHAYA (TENGGELAM)", f"Banjir {depth*100:.0f} cm"
            elif elev < (adj_tide + 0.3):
                return 0.7 if sim_rain > 30 else 0.4, "SIAGA (RISIKO TINGGI)", "Hampir Meluap"
            elif elev < (adj_tide + 0.6):
                return 0.3, "WASPADA (RISIKO)", "Belum Tergenang"
            else:
                return 0.0, "AMAN", "Kering"
                
        # Apply logic
        df_map[['heatmap_intensity', 'status_text', 'depth_est']] = df_map.apply(
            lambda x: pd.Series(get_intensity(x)), axis=1
        )
        
        # User Custom Colorscale: Transparent -> Yellow -> Red -> Black
        custom_heatmap_colors = [
            [0.0, 'rgba(0,0,0,0)'],           # Aman (Transparan)
            [0.3, 'rgba(255, 235, 59, 0.5)'], # Waspada (Kuning Pudar)
            [0.7, 'rgba(255, 0, 0, 0.8)'],    # Bahaya (Merah)
            [1.0, 'rgba(0, 0, 0, 0.9)']       # Ekstrem (Hitam)
        ]

        # Prepare risk/safe dataframes for layers (Common for all tabs)
        df_risk = df_map[df_map['heatmap_intensity'] > 0].copy()
        df_safe = df_map[df_map['heatmap_intensity'] == 0].copy()

        # --- TAB CONTROLS ---
        tab1, tab2, tab3 = st.tabs(["🗺️ Peta Risiko", "🛣️ Cek Jalan (Baru)", "📊 Statistik Wilayah"])
        
        with tab1:
            # Map Layout Controls
            c_layer1, c_layer2 = st.columns([1, 1])
            with c_layer1:
                 map_engine = st.radio("Mode Peta:", ["Plotly (Ringan)", "Folium (Interaktif)"], horizontal=True)
                
            with c_layer2:
                base_map_style = "Citra Satelit" # Default
                show_roads = st.checkbox("🛣️ Tampilkan Jalan Utama", value=False)
                
                if map_engine == "Plotly (Ringan)":
                    base_map_style = st.radio("Tampilan Dasar:", ["Peta Jalan", "Citra Satelit"], horizontal=True)
    
            if map_engine == "Folium (Interaktif)":
                # Call Folium Renderer
                # We pass None for model_pack as strict prediction isn't needed for visualization of pre-calc data
                render_folium_heatmap(None, hourly_risk_df, geojson_data)
               
            else:
                # --- PLOTLY RENDERING (Existing Code) ---
                # Initialize Map Layers
                layers = []
                mapbox_style = "carto-positron"
                if base_map_style == "Citra Satelit":
                    mapbox_style = "white-bg"
                    layers.append({
                        "below": 'traces',
                        "sourcetype": "raster",
                        "sourceattribution": "Esri World Imagery",
                        "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]
                    })
                
                fig_map = go.Figure()
        
                # Prepare risk/safe dataframes for layers
                # (Already defined above)
                
                # --- LAYER OPTIONAL: Major Roads (REAL-TIME FLOOD RISK) ---
                if show_roads:
                    # Load Roads (Lazy Load)
                    @st.cache_data
                    def load_roads_quick():
                        import geopandas as gpd
                        import os
                        path = "data/samarinda_roads.parquet"
                        if os.path.exists(path):
                            return gpd.read_parquet(path)
                        return None
                    
                    gdf_all_roads = load_roads_quick()
                    if gdf_all_roads is not None:
                         # Filter Major Roads only for performance
                         major_types = ['motorway', 'trunk', 'primary', 'secondary']
                         gdf_major = gdf_all_roads[gdf_all_roads['highway'].isin(major_types)].copy()
                         
                         if not gdf_major.empty:
                             # --- REAL-TIME FLOOD RISK CALCULATION ---
                             # Calculate current water level
                             water_level_absolute = sim_tide_level - config.TIDE_DATUM_OFFSET
                             
                             # Get current rainfall contribution
                             # Use max recent rainfall as proxy for accumulation
                             current_rain_24h = hourly_risk_df['rain_rolling_24h'].max()
                             rain_contrib = (current_rain_24h / 1000.0) * 1.5  # Convert mm to m, factor for poor drainage
                             
                             total_water_level = water_level_absolute + rain_contrib
                             
                             # Classify roads by CURRENT FLOOD RISK
                             # Red: Water > Road Elevation (FLOODING NOW)
                             # Orange: Water within 0.5m of road (HIGH RISK)
                             # Yellow: Water within 1m of road (AT RISK)
                             # Green: Safe (Water well below road)
                             
                             def get_flood_risk_category(mean_elev, water_lvl):
                                 if pd.isna(mean_elev):
                                     return 'unknown'
                                 
                                 margin = mean_elev - water_lvl
                                 
                                 if margin <= 0:
                                     return 'flooding'  # Currently flooded
                                 elif margin < 0.5:
                                     return 'high_risk'  # About to flood
                                 elif margin < 1.0:
                                     return 'at_risk'  # Watch closely
                                 else:
                                     return 'safe'
                             
                             elev_col = 'mean_elev' if 'mean_elev' in gdf_major.columns else None
                             
                             flood_colors = {
                                 'flooding': '#D32F2F',    # Dark Red - ALERT
                                 'high_risk': '#FF6F00',   # Deep Orange
                                 'at_risk': '#FDD835',     # Yellow
                                 'safe': '#43A047',        # Green
                                 'unknown': '#9E9E9E'      # Grey
                             }
                             flood_labels = {
                                 'flooding': '🚨 TERGENANG SEKARANG',
                                 'high_risk': '⚠️ Risiko Tinggi (<50cm)',
                                 'at_risk': '🔶 Perlu Waspada (<1m)',
                                 'safe': '✅ Aman',
                                 'unknown': '❓ Data Tidak Tersedia'
                             }
                             
                             if elev_col:
                                 gdf_major['flood_risk'] = gdf_major[elev_col].apply(
                                     lambda x: get_flood_risk_category(x, total_water_level)
                                 )
                             else:
                                 gdf_major['flood_risk'] = 'unknown'
                             
                             # Show flood context
                             st.info(f"💧 **Level Air Saat Ini**: {total_water_level:.2f}m (Pasang: {water_level_absolute:.2f}m + Hujan: {rain_contrib:.2f}m)")
                             
                             # Render each category as separate trace
                             for cat in ['flooding', 'high_risk', 'at_risk', 'safe', 'unknown']:
                                 gdf_cat = gdf_major[gdf_major['flood_risk'] == cat]
                                 if gdf_cat.empty:
                                     continue
                                     
                                 x_coords = []
                                 y_coords = []
                                 for geom in gdf_cat.geometry:
                                     if geom.geom_type == 'LineString':
                                         xs, ys = geom.xy
                                         x_coords.extend(list(xs))
                                         x_coords.append(None)
                                         y_coords.extend(list(ys))
                                         y_coords.append(None)
                                 
                                 # Make flooded roads more visible
                                 line_width = 5 if cat == 'flooding' else 3
                                 
                                 fig_map.add_trace(go.Scattermapbox(
                                     lat=y_coords,
                                     lon=x_coords,
                                     mode='lines',
                                     line=dict(width=line_width, color=flood_colors[cat]),
                                     name=flood_labels[cat],
                                     hoverinfo='skip'
                                 ))

                # Colorscale for Heatmap: Yellow (Low Risk) -> Red -> Black
                custom_heatmap_colors = [
                    [0.0, 'rgba(255, 235, 59, 0.0)'], # Start Transparent
                    [0.1, 'rgba(255, 235, 59, 0.6)'], # Waspada (Kuning)
                    [0.5, 'rgba(255, 0, 0, 0.8)'],    # Bahaya (Merah)
                    [1.0, 'rgba(0, 0, 0, 0.95)']      # Ekstrem (Hitam)
                ]
        
                # --- LAYER 0: Kelurahan Boundaries (Polygon Outline) ---
                if True: # Always show boundaries in Plotly mode
                    fig_map.add_trace(go.Choroplethmapbox(
                        geojson=geojson_data,
                        locations=df_map['NAMOBJ'],
                        z=[0] * len(df_map),  # Dummy value for uniform styling
                        featureidkey="properties.NAMOBJ",
                        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],  # Transparent fill
                        marker_opacity=0,
                        marker_line_width=1.5,
                        marker_line_color='#3498db',  # Blue outline
                        showscale=False,
                        hoverinfo='text',
                        text=df_map['NAMOBJ']
                    ))
        
                # --- LAYER 1: Heatmap for Risks (Hotspots) ---
                if not df_risk.empty:
                    fig_map.add_trace(go.Densitymapbox(
                        lat=df_risk['lat_center'],
                        lon=df_risk['lon_center'],
                        z=df_risk['heatmap_intensity'],
                        radius=40,
                        colorscale=custom_heatmap_colors,
                        zmin=0,
                        zmax=1,
                        opacity=0.8,
                        hoverinfo='text',
                        text=df_risk.apply(lambda x: f"<b>{x['NAMOBJ']}</b><br>Status: {x['status_text']}<br>Level: {x['heatmap_intensity']:.0%}<br>Estimasi: {x['depth_est']}", axis=1)
                    ))
        
                # --- LAYER 2: Scatter Markers for SAFE areas ---
                if not df_safe.empty:
                    fig_map.add_trace(go.Scattermapbox(
                        lat=df_safe['lat_center'],
                        lon=df_safe['lon_center'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color='#00C853',
                            opacity=0.6
                        ),
                        text=df_safe.apply(lambda x: f"<b>{x['NAMOBJ']}</b><br>Status: AMAN<br>Elevasi: {x['mean_elev']:.1f} m", axis=1),
                        hoverinfo='text'
                    ))
        
                # Helper invisible point to ensure centering if absolutely no data (edge case)
                if df_risk.empty and df_safe.empty:
                    fig_map.add_trace(go.Scattermapbox(
                        lat=[-0.498], lon=[117.154], 
                        mode='markers', marker=dict(size=0, opacity=0)
                    ))
        
                # --- LAYER 3: Radar Layer (RainViewer) ---
                radar_info = fetch_radar_timestamp()
                r_ts = None
                if True: # Always attempt radar in default view
                    if radar_info:
                        r_host, r_ts = radar_info
                        layers.append({
                            "below": 'traces',
                            "sourcetype": "raster",
                            "sourceattribution": "RainViewer Radar",
                            "source": [
                                f"{r_host}/v2/radar/{r_ts}/256/{{z}}/{{x}}/{{y}}/2/1_1.png"
                            ],
                            "opacity": 0.6,
                            "minzoom": 0,
                            "maxzoom": 10
                        })
                
                fig_map.update_layout(
                    mapbox_style=mapbox_style, 
                    mapbox_layers=layers,
                    mapbox_zoom=10.5, # Slightly zoomed in for heatmap view
                    mapbox_center={"lat": -0.498, "lon": 117.154},
                    margin={"r":0,"t":0,"l":0,"b":0},
                    height=550,
                    showlegend=False,
                    coloraxis_showscale=False
                )
                
                st.plotly_chart(fig_map, use_container_width=True, key=f"map_heat_{r_ts or 'none'}")
                
                # Legend (Custom HTML)
                st.markdown("""
                <div style='display: flex; gap: 12px; justify-content: center; font-size: 12px; margin-bottom: 20px; flex-wrap: wrap;'>
                    <div style='display: flex; align-items: center; gap: 4px;'><span style='width: 10px; height: 10px; background: black; border-radius: 50%; display: inline-block;'></span> <b>EKSTREM</b>: Tenggelam</div>
                    <div style='display: flex; align-items: center; gap: 4px;'><span style='width: 10px; height: 10px; background: red; border-radius: 50%; display: inline-block;'></span> <b>BAHAYA</b>: Risiko Tinggi</div>
                    <div style='display: flex; align-items: center; gap: 4px;'><span style='width: 10px; height: 10px; background: #FFEB3B; border-radius: 50%; display: inline-block;'></span> <b>WASPADA</b>: Risiko Rendah</div>
                    <div style='display: flex; align-items: center; gap: 4px;'><span style='width: 10px; height: 10px; background: #00C853; border-radius: 50%; display: inline-block;'></span> <b>AMAN</b>: Tidak Berisiko</div>
                    <div style='display: flex; align-items: center; gap: 4px;'><span style='width: 10px; height: 10px; border: 2px solid #3498db; border-radius: 2px; display: inline-block;'></span> <b>Batas Kelurahan</b></div>
                </div>
                """, unsafe_allow_html=True)
            
            with st.expander("Lihat Detail Elevasi Kelurahan"):
                top_risk = df_map.sort_values('mean_elev', ascending=True).head(10)
                col_map1, col_map2 = st.columns([2, 1])
                with col_map1:
                    st.dataframe(
                        top_risk[['NAMOBJ', 'mean_elev', 'risk_pct']].rename(columns={'NAMOBJ': 'Kelurahan', 'risk_pct': 'Persentase Dataran Rendah (%)', 'mean_elev': 'Elevasi Rata-rata (m)'}),
                        use_container_width=True, hide_index=True
                    )
                with col_map2:
                    st.info("ℹ️ **Analisis Data DEM**: Heatmap di atas digerakkan oleh satu titik pusat (Centroid) per Kelurahan. Area gelap menunjukkan pusat kelurahan yang memiliki rata-rata elevasi rendah dan rentan terhadap pasang air laut.")

        with tab2:
            st.markdown("### 🛣️ Analisis Dampak Jalan (Road Impact)")
            
            # Load Road Data (Cached)
            @st.cache_data
            def load_roads_data():
                import geopandas as gpd
                import os
                path = "data/samarinda_roads.parquet"
                if os.path.exists(path):
                    return gpd.read_parquet(path)
                return None
            
            gdf_roads = load_roads_data()
            
            if gdf_roads is None:
                st.warning("⚠️ Data jaringan jalan belum tersedia. Jalankan `python scripts/fetch_osm_roads.py`.")
            else:
                # User Input
                road_names = sorted(gdf_roads['name'].unique().tolist())
                selected_road = st.selectbox("🔍 Cari Nama Jalan:", options=road_names)
                
                if selected_road:
                    # Filter Road
                    road_geom = gdf_roads[gdf_roads['name'] == selected_road]
                    
                    # --- NEW: Physical Elevation Analysis ---
                    # Use the pre-calculated DEM elevation from the parquet file
                    avg_road_elev = road_geom['mean_elev'].mean() if 'mean_elev' in road_geom.columns else 5.0
                    min_road_elev = road_geom['min_elev'].min() if 'min_elev' in road_geom.columns else 0.0
                    
                    # Calculate Water Level Check
                    # Water Level > Road Elevation = FLOOD
                    # TIDE_DATUM_OFFSET converts Gauge Height to Approx Ground Relative
                    # We add a buffer for Rainfall Accumulation (Simulated)
                    
                    water_level_absolute = sim_tide_level - config.TIDE_DATUM_OFFSET
                    # Rainfall Contribution (Rough Estimate: 100mm rain ~ +0.1m accumulation in bad drainage)
                    rain_contrib = (hourly_risk_df['rain_rolling_24h'].max() / 1000.0) * 2 # Factor 2 for drainage fail
                    
                    total_flood_level = water_level_absolute + rain_contrib
                    
                    # Validate against Road
                    flood_depth_on_road = total_flood_level - avg_road_elev
                    
                    st.markdown(f"**Analisis Elevasi Fisik (DEM):**")
                    c_e1, c_e2, c_e3 = st.columns(3)
                    c_e1.metric("Elevasi Jalan", f"{avg_road_elev:.2f} m", help="Rata-rata ketinggian jalan dari DEM")
                    c_e2.metric("Level Air (Est)", f"{total_flood_level:.2f} m", help="Pasang + Akumulasi Hujan")
                    
                    is_flooded_physically = flood_depth_on_road > 0
                    
                    if is_flooded_physically:
                        c_e3.metric("Status", "TERGENANG", f"{flood_depth_on_road*100:.1f} cm", delta_color="inverse")
                        st.error(f"⚠️ **PERINGATAN BAHAYA**: Level air ({total_flood_level:.2f}m) melampaui tinggi jalan ({avg_road_elev:.2f}m).")
                    else:
                        c_e3.metric("Status", "AMAN", f"Margin {-flood_depth_on_road*100:.1f} cm")
                        st.success(f"✅ **DAPAT DILALUI**: Jalan lebih tinggi dari prediksi genangan.")

                    # --- Visual Overlay (Validation) ---
                    # Only calculate intersection if we suspect risk or just for visual context
                    # Prepare Flood Polygons (High Risk Only)
                    df_risk_poly = df_risk[df_risk['heatmap_intensity'] > 0.3]
                    
                    if not df_risk_poly.empty and geojson_data:
                        import geopandas as gpd
                        from shapely.geometry import shape
                        
                        # Convert to GDF
                        features = []
                        for f in geojson_data['features']:
                            props = f['properties']
                            name = props.get('NAMOBJ')
                            if name in df_risk_poly['NAMOBJ'].values:
                                # Get Risk Info
                                risk_row = df_risk_poly[df_risk_poly['NAMOBJ'] == name].iloc[0]
                                features.append({
                                    'geometry': shape(f['geometry']),
                                    'depth_est': risk_row['depth_est'],
                                    'intensity': risk_row['heatmap_intensity'],
                                    'status': risk_row['status_text']
                                })
                        
                        if features:
                            gdf_flood = gpd.GeoDataFrame(features, crs="EPSG:4326")
                            
                            # INTERSECTION ANALYSIS
                            # Ensure CRS matches
                            if gdf_roads.crs != gdf_flood.crs:
                                gdf_flood = gdf_flood.to_crs(gdf_roads.crs)
                            
                            # Spatial Join / Overlay
                            # Clip roads by flood polygons
                            try:
                                inundated = gpd.overlay(road_geom, gdf_flood, how='intersection')
                                
                                # Visualize Map
                                import folium
                                from streamlit_folium import st_folium
                                
                                center_lat = road_geom.geometry.centroid.y.mean()
                                center_lon = road_geom.geometry.centroid.x.mean()
                                
                                m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
                                
                                # Road (Blue)
                                folium.GeoJson(
                                    road_geom,
                                    style_function=lambda x: {'color': 'blue', 'weight': 5},
                                    tooltip=f"Jalan {selected_road} (Elev: {avg_road_elev:.1f}m)"
                                ).add_to(m)
                                
                                # Flood (Red)
                                folium.GeoJson(
                                    gdf_flood,
                                    style_function=lambda x: {'fillColor': 'red', 'color': 'red', 'weight': 0, 'fillOpacity': 0.3},
                                    tooltip="Area Risiko Model"
                                ).add_to(m)

                                if not inundated.empty:
                                    # Overlap (Orange)
                                    folium.GeoJson(
                                        inundated,
                                        style_function=lambda x: {'color': 'orange', 'weight': 6, 'dashArray': '5, 5'},
                                        tooltip="Area Berpotongan"
                                    ).add_to(m)
                                
                                st_folium(m, height=300, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Gagal melakukan overlay spatial: {e}")
                                
                        else:
                             st.success("Visualisasi: Area sekitar aman dari poligon model.")
                    else:
                        st.info("ℹ️ Tidak ada risiko banjir signifikan saat ini untuk dianalisis.")

        with tab3:
            st.info("Dashboard Statistik Wilayah (Coming Soon)")


def render_folium_heatmap(model_pack, hourly_risk_df: pd.DataFrame, geojson_data: dict):
    """
    Render an interactive Folium heatmap showing flood risk predictions.
    Uses st.components.v1.html for better compatibility.
    """
    import folium
    from folium.plugins import HeatMap
    import streamlit.components.v1 as components
    import numpy as np
    import model_utils
    
    st.subheader("🗺️ Peta Risiko Banjir Interaktif (Folium)")
    st.caption("Heatmap berbasis prediksi model ML di berbagai titik koordinat Kota Samarinda")
    
    # Grid of coordinates covering Samarinda
    # Bounds: Lat -0.42 to -0.58, Lon 117.05 to 117.22
    lat_min, lat_max = -0.58, -0.42
    lon_min, lon_max = 117.05, 117.22
    
    # Create grid (20x20 = 400 points)
    n_points = 20
    lats = np.linspace(lat_min, lat_max, n_points)
    lons = np.linspace(lon_min, lon_max, n_points)
    
    # Get current weather conditions from hourly_risk_df
    if hourly_risk_df is not None and not hourly_risk_df.empty:
        current_row = hourly_risk_df.iloc[-1] if len(hourly_risk_df) > 0 else None
        rain_24h = current_row.get('rain_rolling_24h', 0) if current_row is not None else 0
        tide = current_row.get('est', 0) if current_row is not None else 0
        rain_intensity = current_row.get('precipitation', 0) if current_row is not None else 0
    else:
        rain_24h, tide, rain_intensity = 0, 0, 0
    
    # Generate risk predictions for each grid point
    heat_data = []
    kelurahan_risks = []
    
    # Use GeoJSON kelurahan data with elevation-based risk (spatially accurate)
    if geojson_data and 'features' in geojson_data:
        for feature in geojson_data['features']:
            props = feature.get('properties', {})
            geom = feature.get('geometry', {})
            
            kelurahan_name = props.get('NAMOBJ', 'Unknown')
            
            # Get centroid
            if geom.get('type') == 'Polygon':
                coords = geom['coordinates'][0]
                centroid_lon = sum(c[0] for c in coords) / len(coords)
                centroid_lat = sum(c[1] for c in coords) / len(coords)
                
                # Get elevation-based risk from pre-calculated data
                mean_elev = props.get('mean_elev', 10)
                risk_pct = props.get('risk_pct', 0)  # Percentage of low-lying area
                
                # Calculate spatial risk based on:
                # 1. Elevation (lower = higher risk)
                # 2. Percentage of dataran rendah (risk_pct)
                # 3. Current weather conditions
                
                # Elevation factor: areas below 3m are high risk
                elev_factor = max(0, min(1, (3 - mean_elev) / 3))
                
                # Dataran rendah factor
                lowland_factor = risk_pct / 100
                
                # Weather amplifier based on current conditions
                weather_factor = 1.0
                if rain_24h > 50:  # Heavy rain
                    weather_factor = 1.5
                elif rain_24h > 20:
                    weather_factor = 1.2
                    
                if tide > 2.5:  # High tide
                    weather_factor *= 1.3
                elif tide > 2.0:
                    weather_factor *= 1.1
                
                # Combined risk score
                base_risk = (elev_factor * 0.4 + lowland_factor * 0.6)
                final_risk = min(1.0, base_risk * weather_factor)
                
                kelurahan_risks.append({
                    'name': kelurahan_name,
                    'lat': centroid_lat,
                    'lon': centroid_lon,
                    'risk': final_risk,
                    'elevation': mean_elev
                })
                
                heat_data.append([centroid_lat, centroid_lon, final_risk])
    
    # Create Folium Map
    m = folium.Map(
        location=[-0.50, 117.15],
        zoom_start=12,
        tiles='CartoDB positron'
    )
    
    # Add multiple tile layers
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Citra Satelit'
    ).add_to(m)
    
    # Add Risk Markers with modern styling
    if heat_data:
        # Find kelurahan info for tooltips
        kelurahan_info = {(k['lat'], k['lon']): k for k in kelurahan_risks} if kelurahan_risks else {}
        
        for lat, lon, risk in heat_data:
            # Get kelurahan name if available
            info = kelurahan_info.get((lat, lon), {})
            name = info.get('name', 'Unknown')
            elev = info.get('elevation', 0)
            
            # Determine color and label based on risk level
            if risk >= 0.85:
                color = '#dc2626'
                bg_gradient = 'linear-gradient(135deg, #dc2626, #7f1d1d)'
                label = 'EKSTREM'
                pulse_color = '#ef4444'
            elif risk >= 0.7:
                color = '#ea580c'
                bg_gradient = 'linear-gradient(135deg, #ea580c, #c2410c)'
                label = 'TINGGI'
                pulse_color = '#f97316'
            elif risk >= 0.5:
                color = '#d97706'
                bg_gradient = 'linear-gradient(135deg, #fbbf24, #d97706)'
                label = 'SEDANG'
                pulse_color = '#fbbf24'
            elif risk >= 0.3:
                color = '#65a30d'
                bg_gradient = 'linear-gradient(135deg, #84cc16, #65a30d)'
                label = 'RENDAH'
                pulse_color = '#84cc16'
            else:
                color = '#16a34a'
                bg_gradient = 'linear-gradient(135deg, #22c55e, #16a34a)'
                label = 'AMAN'
                pulse_color = '#22c55e'
            
            # Create modern custom icon with pulsing effect
            icon_html = f'''
            <div style="position: relative;">
                <div style="
                    width: 40px;
                    height: 40px;
                    background: {bg_gradient};
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 4px 15px {color}80;
                    border: 3px solid white;
                    animation: pulse-{label.lower()} 2s infinite;
                    cursor: pointer;
                ">
                    <span style="color: white; font-weight: bold; font-size: 11px;">
                        {int(risk*100)}%
                    </span>
                </div>
            </div>
            <style>
                @keyframes pulse-{label.lower()} {{
                    0% {{ box-shadow: 0 0 0 0 {pulse_color}80; }}
                    70% {{ box-shadow: 0 0 0 15px {pulse_color}00; }}
                    100% {{ box-shadow: 0 0 0 0 {pulse_color}00; }}
                }}
            </style>
            '''
            
            # Rich popup content
            popup_html = f'''
            <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 200px;">
                <div style="background: {bg_gradient}; color: white; padding: 12px; border-radius: 8px 8px 0 0; text-align: center;">
                    <h4 style="margin: 0; font-size: 14px;">📍 {name}</h4>
                </div>
                <div style="padding: 12px; background: #f8fafc;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: #64748b;">Status:</span>
                        <span style="font-weight: bold; color: {color};">{label}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: #64748b;">Risiko:</span>
                        <span style="font-weight: bold;">{risk*100:.0f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #64748b;">Elevasi:</span>
                        <span style="font-weight: bold;">{elev:.1f} m</span>
                    </div>
                    <div style="margin-top: 10px; background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                        <div style="width: {risk*100}%; height: 100%; background: {bg_gradient};"></div>
                    </div>
                </div>
            </div>
            '''
            
            # Add marker with custom icon
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    html=icon_html,
                    icon_size=(40, 40),
                    icon_anchor=(20, 20)
                ),
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"<b>{name}</b><br>Risiko: {risk*100:.0f}%"
            ).add_to(m)
            
    # Add RainViewer Radar Layer (Live/Latest)
    # Get latest host and timestamp if available
    radar_host, radar_ts = fetch_radar_timestamp()
    if radar_host and radar_ts:
        folium.TileLayer(
            tiles=f"{radar_host}/v2/radar/{radar_ts}/256/{{z}}/{{x}}/{{y}}/2/1_1.png",
            attr="RainViewer",
            name="🌧️ Radar Hujan (RainViewer)",
            overlay=True,
            opacity=0.7
        ).add_to(m)

    # Add Kelurahan boundaries
    if geojson_data:
        def style_function(feature):
            return {
                'fillColor': 'transparent',
                'color': '#2980b9',
                'weight': 2,
                'fillOpacity': 0
            }
        
        folium.GeoJson(
            geojson_data,
            name='Batas Kelurahan',
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['NAMOBJ', 'mean_elev', 'risk_pct'],
                aliases=['Kelurahan:', 'Elevasi (m):', 'Risiko Dataran Rendah (%):'],
                style='font-size: 12px; font-weight: bold;'
            )
        ).add_to(m)
        
    # --- Impact Analysis Integration ---
    try:
        import impact_analysis
        import geopandas as gpd
        from shapely.geometry import Polygon
        
        # Create GeoDataFrame from risks
        if kelurahan_risks:
            risk_polys = []
            for k in kelurahan_risks:
                if k['risk'] > 0.5: # Consider high risk for impact
                     # Create a small buffer around the centroid as a proxy for the risk area
                     # In a real scenario, use the actual kelurahan geometry if available in props
                     p = Polygon([(k['lon']-0.005, k['lat']-0.005), 
                                  (k['lon']+0.005, k['lat']-0.005), 
                                  (k['lon']+0.005, k['lat']+0.005), 
                                  (k['lon']-0.005, k['lat']+0.005)])
                     risk_polys.append({'geometry': p, 'risk': k['risk']})
            
            if risk_polys:
                risk_gdf = gpd.GeoDataFrame(risk_polys)
                risk_gdf.crs = "EPSG:4326"
                
                # Analyze Impact
                impact = impact_analysis.analyze_impact(risk_gdf)
                
                # Show Impact Stats
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.9); padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #dc2626; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0 0 10px 0; color: #dc2626;">🚨 Estimasi Dampak Banjir</h4>
                    <div style="display: flex; gap: 20px;">
                        <div>
                            <span style="font-size: 24px; font-weight: bold; color: #1e293b;">{impact['total_affected']}</span><br>
                            <span style="font-size: 12px; color: #64748b;">Total Bangunan Terdampak</span>
                        </div>
                        <div>
                            <span style="font-size: 24px; font-weight: bold; color: #d97706;">{impact['schools_affected']}</span><br>
                            <span style="font-size: 12px; color: #64748b;">Sekolah</span>
                        </div>
                        <div>
                            <span style="font-size: 24px; font-weight: bold; color: #dc2626;">{impact['hospitals_affected']}</span><br>
                            <span style="font-size: 12px; color: #64748b;">Fasilitas Kesehatan</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        print(f"Impact Analysis Error: {e}")

    
    # Add Legend as a custom control (without LayerControl to avoid duplicate ID error)
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background: white; padding: 10px; border-radius: 5px; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.3); font-size: 12px;">
        <b>Level Risiko Banjir</b><br>
        <i style="background: darkred; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></i> Ekstrem<br>
        <i style="background: red; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></i> Tinggi<br>
        <i style="background: orange; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></i> Sedang<br>
        <i style="background: yellow; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></i> Rendah<br>
        <i style="background: green; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></i> Aman
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Display map using components.html for better compatibility
    map_html = m._repr_html_()
    components.html(map_html, height=550, scrolling=False)
    
    # Info
    st.caption(f"📊 Data: {len(heat_data)} titik prediksi | Kondisi: Hujan {rain_24h:.1f}mm/24h, Pasang {tide:.2f}m")


def render_shap_explanation(risk_df: pd.DataFrame):
    """
    Render SHAP-based model explainability visualization.
    Shows which features contribute most to flood risk predictions.
    """
    import model_utils
    import shap_explainer
    
    st.subheader("🧠 Explainable AI - Interpretasi Model")
    st.caption("Analisis faktor-faktor yang berkontribusi terhadap prediksi risiko banjir menggunakan SHAP (SHapley Additive exPlanations)")
    
    # Load model
    model_pack = model_utils.load_model()
    
    if model_pack is None:
        st.warning("⚠️ Model belum dimuat. Tidak dapat menampilkan explainability.")
        return
    
    # Get current conditions from risk_df
    if risk_df is not None and not risk_df.empty:
        current = risk_df.iloc[-1]
        rain_24h = current.get('rain_rolling_24h', 0)
        input_data = {
            "rain_sum_imputed": rain_24h,
            "rain_intensity_max": current.get('precipitation', 0),
            "rain_rolling_3h": rain_24h / 8,
            "pasut_msl_max": current.get('est', 0),
            "soil_moisture_surface_mean": 0.45,
            "soil_moisture_root_mean": 0.45,
            "hujan_lag1": rain_24h * 0.8, "hujan_lag2": rain_24h * 0.6, "hujan_lag3": rain_24h * 0.4,
            "hujan_lag4": rain_24h * 0.2, "hujan_lag5": 0, "hujan_lag6": 0, "hujan_lag7": 0,
            "api_7day": rain_24h * 2.5  # Antecedent Precipitation Index
        }
    else:
        input_data = {
            "rain_sum_imputed": 20,
            "rain_intensity_max": 5,
            "rain_rolling_3h": 7.5,
            "pasut_msl_max": 2.0,
            "soil_moisture_surface_mean": 0.45,
            "soil_moisture_root_mean": 0.45,
            "hujan_lag1": 10, "hujan_lag2": 5, "hujan_lag3": 2,
            "hujan_lag4": 0, "hujan_lag5": 0, "hujan_lag6": 0, "hujan_lag7": 0,
            "api_7day": 50  # Antecedent Precipitation Index
        }
    
    # Get explanation
    with st.spinner("Menghitung SHAP values..."):
        explanation = shap_explainer.explain_prediction(model_pack, input_data)
    
    if explanation is None or "error" in explanation:
        st.info("💡 SHAP belum tersedia. Menampilkan feature importance dari model.")
        
        # Fallback to model feature importance
        model = model_pack.get('model')
        feature_names = model_pack.get('feature_names', [])
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=[f.replace("_", " ").title() for f in feature_names[:10]],
                x=importances[:10],
                orientation='h',
                marker=dict(
                    color=importances[:10],
                    colorscale='RdYlGn_r'
                )
            ))
            
            fig.update_layout(
                title="📊 Feature Importance (Model-Based)",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=400,
                template="plotly_dark",
                yaxis=dict(autorange="reversed")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        return
    
    # Get chart data
    chart_data = shap_explainer.get_feature_importance_chart_data(explanation)
    
    if chart_data:
        # Create SHAP waterfall-style bar chart
        features = [d['feature'] for d in chart_data]
        contributions = [d['contribution'] for d in chart_data]
        colors = ['#dc2626' if c > 0 else '#16a34a' for c in contributions]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=features,
            x=contributions,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{c:+.3f}" for c in contributions],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="🎯 Kontribusi Fitur terhadap Prediksi Risiko",
            xaxis_title="Kontribusi SHAP (+ = Meningkatkan Risiko)",
            yaxis_title="Fitur",
            height=450,
            template="plotly_dark",
            yaxis=dict(autorange="reversed"),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔺 Top Faktor Peningkat Risiko")
            for feat, val in explanation.get('top_positive', [])[:3]:
                st.markdown(f"**{feat.replace('_', ' ').title()}**: +{val:.3f}")
        
        with col2:
            st.markdown("### 🔻 Top Faktor Penurun Risiko")
            for feat, val in explanation.get('top_negative', [])[:3]:
                st.markdown(f"**{feat.replace('_', ' ').title()}**: {val:.3f}")
        
        # Explanation
        st.info("""
        **Cara Membaca Grafik:**
        - **Merah (+)**: Fitur ini meningkatkan probabilitas banjir
        - **Hijau (-)**: Fitur ini menurunkan probabilitas banjir
        - Semakin panjang bar, semakin besar pengaruhnya
        """)
