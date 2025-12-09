# Gunakan image Python dasar yang stabil
FROM python:3.11-slim

# Instal dependensi sistem yang hilang (libgomp1)
# Ini KRUSIAL untuk LightGBM agar tidak crash.
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Atur direktori kerja
WORKDIR /app

# Salin semua file (termasuk main.py dan folder saved_models)
COPY . /app

# Instal dependensi Python (dari requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan API menggunakan Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]