# 1. Gunakan image Python dasar yang stabil
FROM python:3.11-slim

# 2. Instal dependensi sistem yang hilang (OpenMP/libgomp1)
# Ini adalah langkah KRITIS untuk LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Atur direktori kerja
WORKDIR /app

# 4. Salin kode & model yang dibutuhkan ke dalam container
COPY . /app

# 5. Instal dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Jalankan API menggunakan Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]