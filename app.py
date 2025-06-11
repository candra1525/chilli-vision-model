import os
import cv2
import numpy as np
import time
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Load API_KEY from .env file
CLOUD_NAME = os.getenv("CLOUD_NAME")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Load model YOLOv11
MODEL_PATH = "models/chilli_vision_best_model.pt"
try:
    model = YOLO(MODEL_PATH)
    model.to("cpu")  # Memindahkan model ke CPU
    print(f"Model berhasil dimuat. {MODEL_PATH}")
except Exception as e:
    print(f"Error memuat model: {e}")
    exit(1)

# Path ke file CSV
# CSV_PATH = "data/penyakit_tanaman_cabai_2_fix.csv"
CSV_PATH = "data/penyakit_cabai_fix.csv"

# Konfigurasi Cloudinary
cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=API_KEY,
    api_secret=API_SECRET
)

def get_disease_info(label):
    if not os.path.exists(CSV_PATH):
        print(f"File CSV tidak ditemukan: {CSV_PATH}")
        return None

    try:
        df = pd.read_csv(CSV_PATH, delimiter=';', encoding='utf-8', on_bad_lines='skip')

        if 'prediksi' not in df.columns:
            print("Kolom 'prediksi' tidak ditemukan dalam CSV.")
            return None

        matching_rows = df[df['prediksi'].astype(str).str.strip().str.lower() == label.strip().lower()]
        return matching_rows.iloc[0].to_dict() if not matching_rows.empty else None

    except Exception as e:
        print(f"Error membaca CSV: {e}")
        return None

@app.route("/detect", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400
    
    allowed_extensions = ["jpg", "jpeg", "png"]
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({"error": "Format file tidak didukung"}), 400

    # Simpan sementara di RAM
    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Gagal membaca gambar"}), 400

    img_height, img_width, _ = image.shape  

    # Mulai hitung waktu deteksi
    start_time = time.time()
    
    # Deteksi menggunakan YOLOv11-L
    results = model(image)[0]
    
    detection_time = time.time() - start_time
    detections = []
    
    if results.boxes is not None:
        for box in results.boxes:
            conf = float(box.conf[0])  
            if conf >= 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                label = model.names[int(box.cls[0])]  
                conf_percent = f"{conf * 100:.0f}%"
                disease_info = get_disease_info(label)
                detections.append({
                    "label": label,
                    "confidence": conf_percent,
                    "bbox": [x1, y1, x2, y2],
                    "disease_info": disease_info
                })
    
    if not detections:
        return jsonify({
            "message": "Tidak ada penyakit terdeteksi", 
            "detection_time": f"{detection_time:.2f} detik"
        }), 200

    # Anotasi hasil deteksi
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 6)
        text = f"{label} ({conf})"
        font_scale = max(0.5, img_width / 1000)
        text_thickness = max(1, int(font_scale * 2))
        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
        text_x, text_y = x1, max(y1 - 10, text_height + 10)

        cv2.rectangle(image, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness)

    # Simpan hasil anotasi sementara dalam memori
    _, buffer = cv2.imencode(".jpg", image)
    
    # Upload ke Cloudinary
    cloudinary_result = cloudinary.uploader.upload(buffer.tobytes(), folder="chilli_diseases")

    # Ambil daftar nama penyakit unik
    unique_disease_names = []
    for det in detections:
        disease_info = det.get("disease_info")
        if disease_info is not None:
            disease_name = disease_info.get("nama_penyakit", "").strip()
            if disease_name and disease_name not in unique_disease_names:
                unique_disease_names.append(disease_name)

    unique_name_disease = ", ".join(unique_disease_names)

    # Kumpulkan ringkasan deteksi unik
    detections_summary = []
    seen_labels = set()
    for det in detections:
        if det["label"] not in seen_labels:
            detections_summary.append({
                "label": det["label"],
                "disease_info": det["disease_info"]
            })
            seen_labels.add(det["label"])

    return jsonify({
        "message": "Deteksi berhasil",
        "detection_time": f"{detection_time:.2f} detik",
        "detections": detections,
        "detections_summary": detections_summary,  
        "unique_name_disease": unique_name_disease,
        "image_url": cloudinary_result["secure_url"]
    }), 200

@app.route("/", methods=["GET"])
def test():
    return jsonify({"message": "API is running"})

# Jalankan Flask server 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
