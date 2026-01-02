"""
Add this function to ui_components.py to display 5 monitoring locations on Folium map
"""

import folium
from folium.plugins import MarkerCluster
import streamlit as st
from streamlit_folium import st_folium
import config

def render_monitoring_locations_map(model_pack, current_time=None):
    """
    Render interactive map with 5 monitoring locations showing real-time flood risk.
    
    Args:
        model_pack: Loaded model package
        current_time: Current datetime for predictions (optional)
    """
    st.subheader("📍 5 Titik Lokasi Monitoring Banjir Samarinda")
    
    # Create base map centered on Samarinda
    m = folium.Map(
        location=[config.LATITUDE, config.LONGITUDE],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add satellite layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Color mapping for risk levels
    def get_marker_color(probability):
        if probability >= 0.85:
            return 'red', '🔴', 'AWAS'
        elif probability >= 0.70:
            return 'orange', '🟠', 'SIAGA'
        elif probability >= 0.50:
            return 'yellow', '🟡', 'WASPADA'
        else:
            return 'green', '🟢', 'AMAN'
    
    # Predict for each location
    import model_utils
    import pandas as pd
    from datetime import datetime
    
    for loc_name, (lat, lon, runoff_coeff) in config.LOCATIONS.items():
        # Prepare input data for prediction
        # Note: In production, fetch real weather data for this location
        input_data = {
            "rain_sum_imputed": 1.8,  # Example: current rain from API
            "rain_intensity_max": 1.8,
            "pasut_msl_max": 3.0,  # Example: current tide from API
            "soil_moisture_surface_mean": 0.45,
            "soil_moisture_root_mean": 0.45,
            "runoff_coefficient": runoff_coeff,
            "rain_lag1": 1.8,
            "rain_lag2": 1.7,
            "rain_lag3": 0.2,
            "rain_lag4": 0.1,
            "rain_lag5": 0,
            "rain_lag6": 0,
            "rain_lag7": 0,
        }
        
        # Get prediction
        try:
            assessment = model_utils.predict_flood(model_pack, input_data)
            probability = assessment.get('probability', 0)
        except Exception as e:
            st.warning(f"Error predicting for {loc_name}: {e}")
            probability = 0
        
        # Get marker style
        color, emoji, level = get_marker_color(probability)
        
        # Create popup content
        popup_html = f"""
        <div style="font-family: Arial; width: 250px;">
            <h4 style="margin:0; color: #1f77b4;">{emoji} {loc_name}</h4>
            <hr style="margin: 5px 0;">
            <b>Status:</b> <span style="color: {color}; font-weight: bold;">{level}</span><br>
            <b>Probabilitas Banjir:</b> {probability*100:.1f}%<br>
            <b>Runoff Coeff:</b> {runoff_coeff}<br>
            <b>Koordinat:</b> {lat:.4f}, {lon:.4f}<br>
            <hr style="margin: 5px 0;">
            <small><i>Update: {datetime.now().strftime('%H:%M:%S')}</i></small>
        </div>
        """
        
        # Add marker with custom icon
        folium.CircleMarker(
            location=[lat, lon],
            radius=15,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{loc_name}: {probability*100:.0f}%",
            color=color,
            fillColor=color,
            fillOpacity=0.7,
            weight=3
        ).add_to(m)
        
        # Add label
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(html=f'''
                <div style="
                    font-size: 11px; 
                    color: white; 
                    background-color: {color}; 
                    padding: 2px 5px; 
                    border-radius: 3px;
                    font-weight: bold;
                    text-shadow: 1px 1px 2px black;
                    white-space: nowrap;
                    transform: translate(-50%, 20px);
                ">
                    {loc_name}
                </div>
            ''')
        ).add_to(m)
    
    # Add rivers overlay (approximate Mahakam and Karang Mumus)
    mahakam_coords = [
        [-0.55, 117.05], [-0.52, 117.13], [-0.50, 117.16], [-0.53, 117.22], [-0.55, 117.30]
    ]
    karang_mumus_coords = [
        [-0.40, 117.16], [-0.45, 117.165], [-0.48, 117.155], [-0.50, 117.15]
    ]
    
    folium.PolyLine(
        mahakam_coords,
        color='blue',
        weight=3,
        opacity=0.6,
        tooltip='Sungai Mahakam'
    ).add_to(m)
    
    folium.PolyLine(
        karang_mumus_coords,
        color='blue',
        weight=3,
        opacity=0.6,
        tooltip='Karang Mumus'
    ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin:0; font-weight:bold;">Level Risiko:</p>
        <p style="margin:3px 0;"><span style="color:red;">🔴</span> AWAS (≥85%)</p>
        <p style="margin:3px 0;"><span style="color:orange;">🟠</span> SIAGA (70-85%)</p>
        <p style="margin:3px 0;"><span style="color:gold;">🟡</span> WASPADA (50-70%)</p>
        <p style="margin:3px 0;"><span style="color:green;">🟢</span> AMAN (<50%)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Display map
    st_folium(m, width=1200, height=600)
    
    # Display summary table
    st.markdown("### 📊 Ringkasan Status Lokasi")
    
    # Create summary dataframe (example - in production, collect from predictions)
    summary_data = []
    for loc_name, (lat, lon, runoff_coeff) in config.LOCATIONS.items():
        # Mock data - replace with actual predictions
        prob = 0.98 if "Antasari" in loc_name else (0.85 if "Lembuswana" in loc_name else 0.3)
        _, emoji, level = get_marker_color(prob)
        summary_data.append({
            "Lokasi": f"{emoji} {loc_name}",
            "Status": level,
            "Risiko": f"{prob*100:.0f}%",
            "Runoff": runoff_coeff,
            "Rekomendasi": "🚨 Evakuasi" if prob >= 0.85 else "⚠️ Siaga" if prob >= 0.7 else "✅ Pantau"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
