# Gunakan Python 3.12 sebagai base image
FROM python:3.12

# Set direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt dan install dependencies
COPY requirements.txt .

# Instal dependensi sistem yang dibutuhkan untuk OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode proyek ke dalam container
COPY . .

# Jalankan aplikasi dengan gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]