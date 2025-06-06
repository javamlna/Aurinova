# Gunakan image Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy seluruh isi project ke /app (build context = root)
COPY . .

# Set working directory ke dalam folder backend
WORKDIR /app/backend

# Install dependencies backend
RUN pip install --no-cache-dir -r requirements.txt

# Expose port untuk backend
EXPOSE 5000

# Jalankan aplikasi backend
CMD ["python", "app.py"]