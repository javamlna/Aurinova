# Gunakan base image Python slim
FROM python:3.11-slim

# Install dependencies system yang dibutuhkan oleh scipy, numpy, dll
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Set workdir di root /app
WORKDIR /app

# Copy semua isi folder backend ke dalam container
COPY . .

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose port aplikasi
EXPOSE 5000

# Jalankan aplikasi
CMD ["python", "app.py"]