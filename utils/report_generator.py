from fpdf import FPDF
import pandas as pd
import os

class FloodReport(FPDF):
    def header(self):
        # Header
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'SISTEM PERINGATAN DINI BANJIR - SAMARINDA', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 5, 'Laporan Situasi & Prediksi Genangan', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        # Footer
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(assessment: dict, weather_data: dict, selected_loc_name: str, tide_level: float):
    """
    Generates a PDF report for the current flood assessment.
    """
    pdf = FloodReport()
    pdf.add_page()
    
    # --- 1. STATUS SUMMARY ---
    level = assessment.get("level", "UNKNOWN")
    label = assessment.get("label", "Unknown")
    depth = assessment.get("depth_cm", 0)
    
    # Color logic for Status Title
    if level == "AMAN":
        pdf.set_text_color(0, 128, 0) # Green
    elif level == "WASPADA":
        pdf.set_text_color(255, 165, 0) # Orange
    elif level == "SIAGA":
        pdf.set_text_color(255, 69, 0) # Red-Orange
    elif level == "AWAS":
        pdf.set_text_color(255, 0, 0) # Red
    else:
        pdf.set_text_color(0, 0, 0)

    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 15, f"STATUS: {label}", 0, 1, 'C')
    pdf.set_text_color(0, 0, 0) # Reset color
    
    pdf.set_font("Arial", "", 12)
    timestamp = pd.Timestamp.now().strftime("%d %B %Y, %H:%M WITA")
    pdf.cell(0, 10, f"Lokasi: {selected_loc_name} | Waktu: {timestamp}", 0, 1, 'C')
    pdf.ln(5)
    
    # --- 2. KEY METRICS ---
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "DATA PENGAMATAN & PREDIKSI", 1, 1, 'L', fill=True)
    pdf.set_font("Arial", "", 11)
    
    col_width = 80
    row_height = 8
    
    data = [
        ("Estimasi Genangan", f"{depth:.1f} cm"),
        ("Curah Hujan (24 Jam)", f"{assessment.get('rain_input', 0):.1f} mm"),
        ("Tinggi Pasang Surut", f"{tide_level:.2f} m"),
        ("Kelembaban Tanah", f"{weather_data.get('soil_moisture', 0):.2f} m³/m³"),
        ("Faktor Utama", assessment.get("main_factor", "-"))
    ]
    
    for metric, value in data:
        pdf.cell(col_width, row_height, metric, 1)
        pdf.cell(0, row_height, value, 1, 1)
        
    pdf.ln(10)
    
    # --- 3. ANALYSIS & RECOMMENDATION ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "ANALISIS PENYEBAB", 1, 1, 'L', fill=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, assessment.get("reasoning", "-"))
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "REKOMENDASI TINDAKAN", 1, 1, 'L', fill=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, assessment.get("recommendation", "-"))
    
    # --- 4. DISCLAIMER ---
    pdf.ln(20)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, "Disclaimer: Laporan ini dihasilkan secara otomatis oleh sistem AI Prediksi Banjir Samarinda. Data merupakan estimasi berdasarkan model hidrologi dan prakiraan cuaca. Harap verifikasi dengan kondisi lapangan.")
    
    # Output
    return pdf.output(dest='S').encode('latin-1')
