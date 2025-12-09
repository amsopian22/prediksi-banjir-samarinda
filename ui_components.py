
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import config
from datetime import datetime

def load_custom_css():
    """
    Injects custom CSS for modern glassmorphism look and feel.
    """
    st.markdown(f"""
        <style>
        /* Import Google Font: Outfit */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {{
            font-family: 'Outfit', sans-serif;
        }}
        
        /* Main Container Background - Subtle Dark Theme Override */
        .stApp {{
            background-color: #0e1117;
            background-image: radial-gradient(circle at 50% 0%, #1c2541 0%, #0b1021 100%);
        }}
        
        /* Card Styling (Glassmorphism) */
        .glass-card {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 24px;
            margin-bottom: 20px;
        }}
        
        /* Typography */
        h1, h2, h3, h4 {{
            color: white !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }}
        
        p, span, div {{
            color: #e0e0e0;
        }}
        
        /* Streamlit Metrics Override */
        [data-testid="stMetricValue"] {{
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: white !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: #a0a0a0 !important;
            font-size: 0.9rem !important;
        }}

        /* Hero Banner Styling */
        .hero-title {{
            font-size: 3rem; 
            font-weight: 800; 
            background: -webkit-linear-gradient(0deg, #ffffff, #a0c4ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }}
        
        .status-badge {{
            padding: 8px 16px; 
            border-radius: 50px; 
            font-weight: 700; 
            font-size: 1.2rem;
            display: inline-block;
            margin-bottom: 16px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        
        .pulse {{
            box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7);
            animation: pulse-red 2s infinite;
        }}
        
        @keyframes pulse-red {{
            0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 82, 82, 0.7); }}
            70% {{ transform: scale(1); box-shadow: 0 0 0 10px rgba(255, 82, 82, 0); }}
            100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 82, 82, 0); }}
        }}

        /* Custom Table Styling */
        .dataframe {{
            font-size: 0.9rem !important;
        }}
        
        </style>
    """, unsafe_allow_html=True)

def render_executive_summary(curr_prob: float, curr_tide: float, tma_benanga: float, hourly_risk_df: pd.DataFrame):
    """
    Renders the Executive Summary section as a Hero Banner.
    """
    # Determine Level & Recommendation
    prob_pct = curr_prob * 100
    
    # Default Levels
    if prob_pct < 20:
        level = "NORMAL"
        badge_color = config.COLOR_PALETTE["status_safe"]
        rec_text = "✅ <b>Kondisi Kondusif</b>. Lakukan pemantauan rutin. Tidak ada ancaman banjir signifikan."
        icon = "🛡️"
        pulse_class = ""
    elif prob_pct < 50:
        level = "WASPADA"
        badge_color = config.COLOR_PALETTE["status_warning"]
        rec_text = "⚠️ <b>Monitor Intensif</b>. Cek kondisi pintu air dan drainase utama. Perintahkan tim lapangan standby."
        icon = "⚠️"
        pulse_class = ""
    elif prob_pct < 80:
        level = "SIAGA"
        badge_color = "orange"
        rec_text = "🟠 <b>Siagakan Personil</b>. Aktifkan pompa pengendali banjir. Informasikan Camat & Lurah di wilayah rawan."
        icon = "📢"
        pulse_class = "pulse"
    else:
        level = "AWAS"
        badge_color = config.COLOR_PALETTE["status_danger"]
        rec_text = "🚨 <b>BAHAYA BANJIR</b>. <b>AKTIFKAN POSKO DARURAT</b>. Segera mobilisasi evakuasi warga di titik rendah (< 5 mdpl)."
        icon = "🚨"
        pulse_class = "pulse"
    
    # --- HARD RULE OVERRIDES ---
    # 1. Physical Tide
    if curr_tide >= config.THRESHOLD_TIDE_PHYSICAL_DANGER:
        level = "AWAS (ROB)"
        badge_color = config.COLOR_PALETTE["status_danger"]
        rec_text = f"🚨 <b>PERINGATAN FISIK</b>: Tinggi pasang (<b>{curr_tide:.2f}m</b>) sangat tinggi. Banjir Rob tak terelakkan. <b>EVAKUASI WARGA</b>."
        pulse_class = "pulse"
    elif curr_tide >= config.THRESHOLD_TIDE_LOW_RISK and curr_tide < config.THRESHOLD_TIDE_PHYSICAL_DANGER:
        level = "SIAGA (PASANG)"
        badge_color = "orange"
        rec_text = f"🟠 <b>PERINGATAN FISIK</b>: Pasang naik (<b>{curr_tide:.2f}m</b>). Genangan air di jalan rendah."
        pulse_class = ""

    # 2. Benanga Dam
    if tma_benanga >= config.THRESHOLD_BENANGA_BAHAYA:
         level = "AWAS (BENANGA)"
         badge_color = config.COLOR_PALETTE["status_critical"]
         rec_text = f"🚨 <b>BAHAYA KRITIS</b>: Bendungan Benanga (<b>{tma_benanga:.2f}m</b>) LIMPAS. Banjir kiriman besar tiba 4-6 jam."
         pulse_class = "pulse"
    elif tma_benanga >= config.THRESHOLD_BENANGA_SIAGA:
         if "AWAS" not in level:
             level = "SIAGA (BENANGA)"
             badge_color = "orange"
             rec_text = f"🟠 <b>HULU KRITIS</b>: TMA Benanga (<b>{tma_benanga:.2f}m</b>) tinggi. Debit kiriman meningkat."

    # Render Hero HTML
    st.markdown(f"""
        <div class="glass-card" style="text-align: center; padding: 40px 20px;">
            <div style="font-size: 1.2rem; margin-bottom: 10px; color: #a0c4ff; text-transform: uppercase; letter-spacing: 2px;">Status Peringatan Dini</div>
            <div class="{pulse_class} status-badge" style="background-color: {badge_color}; color: white; display: inline-flex; align-items: center; gap: 10px; padding: 10px 30px;">
                <span style="font-size: 1.5rem;">{icon}</span> {level}
            </div>
            <div style="margin-top: 20px; font-size: 1.1rem; max-width: 800px; margin-left: auto; margin-right: auto; line-height: 1.6;">
                {rec_text}
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_metrics(curr_status: str, total_rain_24h: float, curr_tide: float, tide_status: str, tma_benanga: float):
    """
    Renders key metrics using custom HTML cards.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    def metric_card(title, value, subtext, color="white"):
        return f"""
        <div class="glass-card" style="padding: 20px; text-align: left; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="color: #a0a0a0; font-size: 0.9rem; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;">{title}</div>
            <div style="font-size: 2rem; font-weight: 700; color: {color}; margin: 10px 0;">{value}</div>
            <div style="font-size: 0.85rem; color: #808080;">{subtext}</div>
        </div>
        """
        
    # Logic for Dynamic Colors
    rain_color = config.COLOR_PALETTE["status_warning"] if total_rain_24h > 50 else "white"
    tide_color = config.COLOR_PALETTE["status_danger"] if tide_status == "Bahaya" else "white"
    
    benanga_color = "white"
    if tma_benanga >= config.THRESHOLD_BENANGA_BAHAYA: benanga_color = config.COLOR_PALETTE["status_critical"]
    elif tma_benanga >= config.THRESHOLD_BENANGA_SIAGA: benanga_color = "orange"
        
    with col1:
        st.markdown(metric_card("Status Teknis", curr_status, "AI Prediction"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Curah Hujan (24h)", f"{total_rain_24h:.1f} <span style='font-size:1rem'>mm</span>", "Akumulasi Harian", rain_color), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("Tinggi Pasang", f"{curr_tide:.2f} <span style='font-size:1rem'>m</span>", tide_status, tide_color), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("TMA Benanga", f"{tma_benanga:.2f} <span style='font-size:1rem'>m</span>", "Level Hulu", benanga_color), unsafe_allow_html=True)

def render_hourly_chart(hourly_risk_df: pd.DataFrame):
    """
    Renders the Plotly chart for hourly risk.
    """
    st.divider()
    st.divider()
    st.subheader("📉 Grafik Tren Hujan & Pasang Surut (48 Jam)")
    
    # Dual Axis: Precipitation (Bar) & Tide (Line)
    fig = go.Figure()
    
    # 1. Rain Intensity (Bar)
    fig.add_trace(go.Bar(
        x=hourly_risk_df['time'],
        y=hourly_risk_df['precipitation'],
        name='Curah Hujan (mm)',
        marker_color='#5DADEC', # Blue Jeans Color
        opacity=0.6,
        yaxis='y1'
    ))
    
    # 2. Tide Level (Line)
    fig.add_trace(go.Scatter(
        x=hourly_risk_df['time'],
        y=hourly_risk_df['est'],
        name='Tinggi Pasang (m)',
        line=dict(color='#FFD700', width=3), # Gold/Yellow for brightness on dark bg
        yaxis='y2'
    ))

    # 3. Critical Tide Line (2.5m)
    fig.add_hline(y=config.THRESHOLD_TIDE_PHYSICAL_DANGER, line_dash="dash", line_color="red", 
                  annotation_text=f"Batas Bahaya ({config.THRESHOLD_TIDE_PHYSICAL_DANGER}m)", 
                  annotation_position="top right", yref='y2')
    
    fig.update_layout(
        title=dict(text="Prediksi Curah Hujan vs Pasang Surut", x=0.02, y=0.98, xanchor='left', yanchor='top'),
        yaxis=dict(
            title="Curah Hujan (mm)",
            side='left',
            showgrid=False
        ),
        yaxis2=dict(
            title="Tinggi Pasang (meter)",
            overlaying='y',
            side='right',
            showgrid=True,
            range=[0, 4.2] # Adjust based on historical max
        ),
        legend=dict(
            orientation='h',
            yanchor="top",
            y=-0.2, # Position further below
            xanchor="center",
            x=0.5
        ),
        hovermode="x unified",
        margin={"r":10,"t":50,"l":10,"b":60} # Increased margins generally
    )
    
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
        # Fallback to next 48h
        now = datetime.now(tz=hourly_risk_df['time'].dt.tz) if not hourly_risk_df.empty else datetime.now()
        future_tide_df = hourly_risk_df[hourly_risk_df['time'] >= now].head(48) # Full 48 hours
    
    if not future_tide_df.empty:
        # Create timestamp map for slider
        time_options = future_tide_df['time'].dt.strftime('%d %b %H:%M').tolist()
        
        # Determine Default Value (Next Hour)
        # If no date filter (live mode), default to next full hour
        default_idx = 0
        if not selected_date:
            now_dt = datetime.now(tz=hourly_risk_df['time'].dt.tz) if not hourly_risk_df.empty else datetime.now()
            # Find the option closest to (now + 1 hour) rounded to hour
            next_hour = (now_dt + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            next_hour_str = next_hour.strftime('%d %b %H:%M')
            
            if next_hour_str in time_options:
                default_idx = time_options.index(next_hour_str)
        
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
        radar_ts = fetch_radar_timestamp()
        
        layers = [
            {
                "below": 'traces',
                "sourcetype": "raster",
                "sourceattribution": "Esri World Imagery",
                "source": [
                    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ]
            }
        ]
        
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
            mapbox_style="white-bg", 
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
