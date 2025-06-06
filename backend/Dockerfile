FROM python:3.11-slim

# Buat working dir
WORKDIR /app

# Copy isi folder backend ke dalam /app
COPY . .

# Copy frontend (relatif dari posisi Docker build context = root proyek)
COPY ../frontend ./frontend

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Jalankan Flask
CMD ["python", "app.py"]
