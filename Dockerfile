# Menggunakan image Python 3.11 yang ringan
FROM python:3.11-slim

# Menentukan direktori kerja di dalam container
WORKDIR /app

# Menyalin file requirements dan menginstall dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh kode aplikasi ke dalam container
COPY . .

# Menjalankan Streamlit
# Cloud Run secara otomatis menyuntikkan variabel environment PORT (biasanya 8080)
CMD streamlit run dashboard.py --server.port=${PORT:-8080} --server.address=0.0.0.0