FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy semua isi project-root ke /app di container
COPY . .

# Install dependencies dari backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Buka port 5000
EXPOSE 5000

# Jalankan Flask app dari backend/app.py
CMD ["python", "backend/app.py"]