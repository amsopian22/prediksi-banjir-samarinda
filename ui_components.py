
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import config
from datetime import datetime

def load_custom_css():
    """
    Injects custom CSS for modern glassmorphism look and feel.
    """
    st.markdown("""
        <style>
        /* Import Google Font: Outfit */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Main Container Background - Subtle Dark Theme Override */
        .stApp {
            background-color: #0e1117;
            background-image: radial-gradient(circle at 50% 0%, #1c2541 0%, #0b1021 100%);
        }
        
        /* Card Styling (Glassmorphism) */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 30px;
            margin-bottom: 25px;
            min-height: 200px; /* Ensure visual balance */
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        /* Typography */
        h1, h2, h3, h4 {
            color: white !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }
        
        p, span, div {
            color: #e0e0e0;
        }
        
        /* Streamlit Metrics Override */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: white !important;
        }
        [data-testid="stMetricLabel"] {
            color: #a0a0a0 !important;
            font-size: 0.9rem !important;
        }

        /* Hero Status Banner */
        .hero-banner {
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 40px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }
        
        .hero-status-text {
            font-size: 4rem; 
            font-weight: 800;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin: 0;
            line-height: 1.1;
            text-shadow: 0 4px 20px rgba(0,0,0,0.6);
        }
        
        .hero-subtext {
            font-size: 1.3rem;
            color: #ccc;
            margin-top: 15px;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }

        /* Metric Cards Redesigned */
        .metric-card {
            background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
            border-radius: 16px;
            padding: 2px; /* Border gradient wrapper */
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            height: 100%;
            min-height: 140px;
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-inner {
            background: #141824; /* Dark inner */
            border-radius: 14px;
            padding: 20px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center; /* Center content vertically */
            align-items: flex-start;
        }

        .metric-title {
            color: #8b9bb4;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: white;
            margin-bottom: 5px;
        }
        
        .metric-sub {
            font-size: 0.8rem;
            color: #64748b;
        }
        
        /* Pulse Animation */
        .pulse-red {
            animation: pulse-animation 2s infinite;
        }
        
        @keyframes pulse-animation {
            0% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0.4); }
            70% { box-shadow: 0 0 0 15px rgba(255, 82, 82, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0); }
        }
        
        /* Layout Utilities */
        .row-gap {
            display: flex; 
            gap: 20px;
        }
        </style>
    """, unsafe_allow_html=True)


def render_executive_summary(assessment: dict):
    """
    Renders the Executive Summary section as a Hero Banner.
    """
    # Extract Data
    level = assessment.get("level", "UNKNOWN")
    label = assessment.get("label", "Unknown")
    color = assessment.get("color", "gray")
    rec_text = assessment.get("recommendation", "Tidak ada data.")
    
    # Map color names to hex/palette
    badge_color = "gray"
    if color == "green": badge_color = config.COLOR_PALETTE["status_safe"]
    elif color == "yellow": badge_color = config.COLOR_PALETTE["status_warning"]
    elif color == "orange": badge_color = "#ff9800" # Orange
    elif color == "red": badge_color = config.COLOR_PALETTE["status_danger"]

    # Icons & Pulse
    icon = "🛡️"
    pulse_class = ""
    bg_gradient = "linear-gradient(90deg, #1c2541 0%, #0b1021 100%)"
    
    if level == "NORMAL":
        icon = "✅"
    elif level == "WASPADA":
        icon = "⚠️"
    elif level == "SIAGA":
        icon = "📢"
        pulse_class = "pulse"
        bg_gradient = "linear-gradient(90deg, rgba(255, 109, 0, 0.15) 0%, rgba(11, 16, 33, 0.8) 100%)"
    elif level == "AWAS":
        icon = "🚨"
        pulse_class = "pulse-red"
        bg_gradient = "linear-gradient(90deg, rgba(213, 0, 0, 0.2) 0%, rgba(11, 16, 33, 0.8) 100%)"

    st.markdown(f"""
        <div class="hero-banner {pulse_class}" style="background: {bg_gradient}; border-left: 5px solid {badge_color};">
            <div style="margin-bottom: 10px; font-size: 0.9rem; letter-spacing: 3px; color: #8b9bb4; text-transform: uppercase;">STATUS SISTEM PERINGATAN DINI</div>
            <h1 class="hero-status-text" style="color: {badge_color};">{label}</h1>
            <div class="hero-subtext">
                {icon} {rec_text}
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_risk_context(assessment: dict):
    """
    Displays the 'Why' (Reasoning) and 'Action' (Recommendation) in a structured way.
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="margin-top:0;">🔍 Analisis Penyebab</h3>
            <p style="font-size: 1.1rem; font-weight: 500; color: #ffcc80;">{assessment.get('main_factor', '-')}</p>
            <p style="font-size: 0.9rem; color: #b0bec5;">{assessment.get('reasoning', '-')}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="margin-top:0;">📋 Rekomendasi Tindakan</h3>
            <p style="font-size: 1rem; color: #e0e0e0; line-height: 1.6;">{assessment.get('recommendation', '-')}</p>
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
                        subplot_titles=("Curah Hujan & Pasang Surut", "Probabilitas Risiko Banjir (%)"),
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

    # --- ROW 2: Flood Probability (Area) ---
    fig.add_trace(go.Scatter(
        x=hourly_risk_df['time'],
        y=hourly_risk_df['probability'] * 100,
        name='Risiko Banjir (%)',
        fill='tozeroy',
        mode='lines',
        line=dict(color='#ff5252')
    ), row=2, col=1)
    
    # Logic Threshold
    fig.add_hline(y=50, line_dash="dot", line_color="orange", annotation_text="Waspada (50%)", row=2, col=1)

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
    fig.update_yaxes(title_text="Risiko (%)", row=2, col=1, range=[0, 100])
    
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=600) # Cache for 10 minutes
def fetch_radar_timestamp():
    """Fetch the latest available radar timestamp from RainViewer API."""
    import requests
    try:
        url = "https://api.rainviewer.com/public/weather-maps.json"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Get the very latest past timestamp
            if "radar" in data and "past" in data["radar"] and len(data["radar"]["past"]) > 0:
                latest = data["radar"]["past"][-1]
                return latest["time"]
    except Exception as e:
        pass # Fail silently and return None
    return None

def render_map_simulation(geojson_data: dict, hourly_risk_df: pd.DataFrame, lat: float, lon: float, selected_date=None):
    """
    Renders the Dynamic Inundation Map with Time Slider.
    """
    import os
    if not geojson_data:
        st.warning("Data GeoJSON peta tidak tersedia.")
        return

    feats = [f['properties'] for f in geojson_data['features']]
    df_map = pd.DataFrame(feats)
    
    st.divider()
    st.subheader("🗺️ Peta Simulasi & Dampak Genangan")
    st.info("💡 **Petunjuk**: Peta ini dinamis. Gunakan **slider waktu** di bawah untuk melihat bagaimana air pasang akan menggenangi wilayah rendah dari jam ke jam.")

    # Dynamic Slider for Tide Simulation
    # Check if selected_date is provided
    if selected_date:
        # Filter for specific date
        target_start = pd.to_datetime(selected_date).tz_localize(hourly_risk_df['time'].dt.tz)
        target_end = target_start + pd.Timedelta(days=1)
        future_tide_df = hourly_risk_df[(hourly_risk_df['time'] >= target_start) & (hourly_risk_df['time'] < target_end)]
    else:
        # Fallback to next 48h but starting from CURRENT HOUR FLOOR
        # Example: Now is 11:30. We want 11:00 to be included.
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
            selected_time_str = st.select_slider(
                "⏳ **Pilih Waktu Simulasi**:", 
                options=time_options, 
                value=time_options[default_idx], # Set Default to Next Hour
                key='map_simulation_slider'
            )
        
        # Get tide level for selected time
        selected_idx = future_tide_df.index[future_tide_df['time'].dt.strftime('%d %b %H:%M') == selected_time_str][0]
        selected_row = future_tide_df.loc[selected_idx]
        sim_tide_level = selected_row['est']
        
        # Calculate Trend (vs previous hour logic if available in dataframe)
        # If it's the first item, compare with next (reverse) or just 0
        tide_trend = 0
        if selected_idx > future_tide_df.index[0]:
             # We need to rely on the fact that the dataframe is sorted by time
             prev_level = future_tide_df.loc[selected_idx - 1, 'est'] if (selected_idx - 1) in future_tide_df.index else sim_tide_level
             tide_trend = sim_tide_level - prev_level
        elif selected_idx < future_tide_df.index[-1]:
             # Use next data to see slope
             next_level = future_tide_df.loc[selected_idx + 1, 'est']
             tide_trend = sim_tide_level - next_level # Just approximation

        # Status Logic
        is_danger = sim_tide_level > config.THRESHOLD_TIDE_PHYSICAL_DANGER
        
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

        # Display Current Tide Context (Custom HTML for single arrow control)
        with col_ctrl2:
            st.markdown(f"""
            <div style="text-align: center;">
                <p style="color: #808495; font-size: 0.75rem; margin: 0 0 0.1rem 0; font-weight: 400;">Tinggi Pasut</p>
                <p style="font-size: 1.75rem; font-weight: 700; color: white; margin: 0; line-height: 1.2;">{sim_tide_level:.2f} m</p>
                <p style="color: {color}; font-size: 0.85rem; margin: 0.1rem 0 0 0; font-weight: 500;">{arrow} {text}</p>
            </div>
            """, unsafe_allow_html=True)

        # Calculate Vulnerability (Reverted to Mean Elevation)
        # Rule: 
        # Red (Danger): Mean Elev < Tide Level (Flooded)
        # Yellow (Warning): Mean Elev < Tide Level + 1m (Risk)
        # Green (Safe): Mean Elev > Tide Level + 1m
        
        def get_status_details(row):
            elev = row['mean_elev']
            adj_tide = sim_tide_level - config.TIDE_DATUM_OFFSET
            depth = adj_tide - elev
            
            if elev < adj_tide:
                return 1.0, "BAHAYA (TENGGELAM)", f"{depth*100:.0f} cm"
            elif elev < (adj_tide + 1.0):
                return 0.5, "WASPADA (RISIKO)", "Belum Tergenang"
            else:
                return 0.0, "AMAN", "Kering"
                
        # Apply logic to create multiple columns
        df_map[['sim_score', 'status_text', 'depth_est']] = df_map.apply(
            lambda x: pd.Series(get_status_details(x)), axis=1
        )
        
        # Custom Colorscale
        custom_colorscale = [
            [0.0, "green"],
            [0.5, "yellow"],
            [1.0, "red"]
        ]

        # Create Dynamic Map
        fig_map = go.Figure(go.Choroplethmapbox(
            geojson=geojson_data,
            locations=df_map['NAMOBJ'],
            z=df_map['sim_score'],
            featureidkey="properties.NAMOBJ",
            colorscale=custom_colorscale,
            zmin=0,
            zmax=1,
            marker_opacity=0.6,
            marker_line_width=1,
            text=df_map['status_text'],
            hovertemplate=(
                "<b>Negara/Kelurahan: %{location}</b><br>" +
                "Status: <b>%{text}</b><br>" +
                "Estimasi Genangan: <b>%{customdata[1]}</b><br>" +
                "Elevasi Tanah: %{customdata[0]:.1f} mdpl<br>" +
                "<extra></extra>"
            ),
            customdata=df_map[['mean_elev', 'depth_est']]
        ))
        
        # Add Rain Radar Layer (if available)
        # Map Style Toggle
        map_style = st.radio("Tampilan Peta:", ["Satelit (Citra)", "Jalan (Label/Alamat)"], horizontal=True)
        
        # Define Layers based on selection
        if map_style == "Satelit (Citra)":
            mapbox_style = "white-bg"
            layers = [{
                "below": 'traces',
                "sourcetype": "raster",
                "sourceattribution": "Esri World Imagery",
                "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]
            }]
        else:
            # OpenStreetMap for clear addresses
            mapbox_style = "open-street-map" 
            layers = []
            
        radar_ts = fetch_radar_timestamp()
        if radar_ts:
            layers.append({
                "below": 'traces',
                "sourcetype": "raster",
                "sourceattribution": "RainViewer Radar",
                "source": [
                    f"https://tile.rainviewer.com/{radar_ts}/256/{{z}}/{{x}}/{{y}}/2/1_1.png"
                ],
                "opacity": 0.7
            })

        fig_map.update_layout(
            mapbox_style=mapbox_style, 
            mapbox_layers=layers,
            mapbox_zoom=10.8, # Optimized for Full Samarinda View
            mapbox_center={"lat": -0.498, "lon": 117.154}, # Fixed Center
            margin={"r":0,"t":0,"l":0,"b":0},
            height=500,
            showlegend=False,
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Legend Explanation
        st.markdown("""
        <div style='display: flex; gap: 20px; justify-content: center; font-size: 14px; margin-bottom: 20px;'>
            <div style='background-color: #ffe6e6; color: #330000; padding: 5px 10px; border-radius: 5px; border: 1px solid #ffcccc;'><span style='color: #d50000;'>■</span> <b>TENGGELAM</b> (Elevasi < Pasut)</div>
            <div style='background-color: #fffde7; color: #333300; padding: 5px 10px; border-radius: 5px; border: 1px solid #fff9c4;'><span style='color: #fbc02d;'>■</span> <b>WASPADA</b> (Selisih < 1m)</div>
            <div style='background-color: #e8f5e9; color: #003300; padding: 5px 10px; border-radius: 5px; border: 1px solid #c8e6c9;'><span style='color: #2e7d32;'>■</span> <b>AMAN</b> (Dataran Tinggi)</div>
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
            st.info("ℹ️ **Analisis Data DEM**: Peta ini menampilkan persentase wilayah di setiap Kelurahan yang berada di dataran rendah (Elevasi < 7 mdpl). Wilayah ini memiliki potensi genangan tertinggi saat terjadi pasang air laut > 2.5 meter.")
