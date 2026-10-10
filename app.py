import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import joblib
from ultralytics import YOLO
from skimage.feature import graycomatrix, graycoprops

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Klasifikasi Tipat Bali - YOLOv8 vs Klasik JTIM",
    page_icon="🍙",
    layout="wide"
)

st.title("🍙 Sistem Klasifikasi & Deteksi Tipat Bali")
st.markdown("Aplikasi Pengolahan Citra Digital: Komparasi **YOLOv8 (Deep Learning)** dan **Metode Klasik JTIM (21 Fitur)**.")

# ==========================================
# 2. LOAD MODEL & UTILITIES
# ==========================================
@st.cache_resource
def load_yolo_model():
    # Daftar prioritas lokasi file bobot best.pt hasil training GPU
    possible_paths = [
        "runs/detect/yolov8_ketupat_optimal/weights/best.pt",
        "best.pt",
        "weights/best.pt",
        "runs/detect/train/weights/best.pt"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return YOLO(path), path
            
    return YOLO("yolov8n.pt"), None

@st.cache_resource
def load_klasik_model():
    model = joblib.load('model_klasik.pkl') if os.path.exists('model_klasik.pkl') else None
    scaler = joblib.load('scaler_klasik.pkl') if os.path.exists('scaler_klasik.pkl') else None
    return model, scaler

# Memanggil model dan menampilkan elemen UI di luar fungsi cache
yolo_model, model_path = load_yolo_model()
if model_path:
    st.toast(f"✓ Berhasil memuat model terspesialisasi: {model_path}", icon="✅")
else:
    st.error("⚠️ File `best.pt` tidak ditemukan! Aplikasi menggunakan model standar `yolov8n.pt` yang belum dilatih Tipat Bali.")

klasik_model, klasik_scaler = load_klasik_model()

# ==========================================
# 3. FUNGSI EKSTRAKSI 21 FITUR (JTIM)
# ==========================================
def extract_21_features(img_bgr):
    # 1. HSV (6 Fitur)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_h, std_h = np.mean(hsv[:, :, 0]), np.std(hsv[:, :, 0])
    mean_s, std_s = np.mean(hsv[:, :, 1]), np.std(hsv[:, :, 1])
    mean_v, std_v = np.mean(hsv[:, :, 2]), np.std(hsv[:, :, 2])
    f_hsv = [mean_h, std_h, mean_s, std_s, mean_v, std_v]

    # 2. GLCM (13 Fitur)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').ravel()
    correlation = graycoprops(glcm, 'correlation').ravel()
    energy = graycoprops(glcm, 'energy').ravel()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    f_glcm = [*contrast, *correlation, *energy, homogeneity]

    # 3. Bentuk (2 Fitur)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        metric = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        if len(cnt) >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            a, b = ma / 2, MA / 2
            eccentricity = np.sqrt(1 - (b**2 / a**2)) if a > 0 else 0
        else:
            eccentricity = 0
    else:
        metric, eccentricity = 0, 0
    f_shape = [metric, eccentricity]

    return f_hsv + f_glcm + f_shape, f_hsv, f_glcm, f_shape

# ==========================================
# 4. SIDEBAR NAVIGATION & SETTINGS
# ==========================================
st.sidebar.header("⚙️ Pengaturan & Metode")
engine_choice = st.sidebar.radio(
    "Pilih Mode Operasi:",
    ["YOLOv8 Object Detection", "Metode Klasik JTIM (21 Fitur)", "Komparasi & Evaluasi Model"]
)

st.sidebar.markdown("---")
if engine_choice == "YOLOv8 Object Detection":
    st.sidebar.subheader("🎛️ Parameter YOLOv8")
    conf_threshold = st.sidebar.slider(
        "Batas Kepercayaan (Confidence):", 
        min_value=0.05, 
        max_value=1.00, 
        value=0.25, 
        step=0.05,
        help="Turunkan nilai ini jika objek Tipat tidak terdeteksi, atau naikkan jika terlalu banyak deteksi palsu."
    )

# ==========================================
# 5. MODE 1: YOLOV8 OBJECT DETECTION
# ==========================================
if engine_choice == "YOLOv8 Object Detection":
    st.subheader("🎯 Deteksi Otomatis & Bounding Box (YOLOv8)")
    st.caption("Unggah foto utuh berisi Tipat Bali. Model akan mendeteksi posisi dan jenis tipat secara otomatis.")
    
    uploaded_file = st.file_uploader("Unggah foto Tipat Bali:", type=["jpg", "jpeg", "png"], key="yolo_up")

    if uploaded_file is not None:
        # Konversi gambar ke RGB untuk mengatasi gambar RGBA (4 channels)
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)

        with col2:
            st.write("### Hasil Deteksi:")
            img_np = np.array(image)
            
            # Run YOLOv8 inference dengan parameter confidence dari slider
            results = yolo_model(img_np, conf=conf_threshold)
            res_plotted = results[0].plot()
            st.image(res_plotted, caption=f"Hasil Bounding Box YOLOv8 (Conf: {conf_threshold})", use_container_width=True)

# ==========================================
# 6. MODE 2: METODE KLASIK JTIM
# ==========================================
elif engine_choice == "Metode Klasik JTIM (21 Fitur)":
    st.subheader("🔬 Ekstraksi 21 Fitur Manual & Logistic Regression")
    st.caption("Unggah foto objek Tipat yang sudah di-crop (terpotong pas di objeknya) untuk ekstraksi fitur warna, tekstur, dan bentuk.")
    
    if klasik_model is None or klasik_scaler is None:
        st.warning("⚠️ File `model_klasik.pkl` / `scaler_klasik.pkl` tidak ditemukan. Jalankan `python klasik_jtim.py` terlebih dahulu.")
    else:
        uploaded_file = st.file_uploader("Unggah foto objek Tipat (cropped):", type=["jpg", "jpeg", "png"], key="klasik_up")

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(image, caption="Objek Tipat (Crop)", use_container_width=True)

                features, f_hsv, f_glcm, f_shape = extract_21_features(img_np)
                scaled_feat = klasik_scaler.transform([features])
                pred = klasik_model.predict(scaled_feat)[0]

                st.success(f"**Prediksi Klasik:** {pred}")

            with col2:
                st.write("### 📊 Tabel Nilai Ekstraksi 21 Fitur")

                df_features = pd.DataFrame({
                    "Kategori Fitur": ["HSV (Warna)"]*6 + ["GLCM (Tekstur)"]*13 + ["Bentuk"]*2,
                    "Nama Fitur": [
                        "Mean H", "Std H", "Mean S", "Std S", "Mean V", "Std V",
                        "Contrast 0°", "Contrast 45°", "Contrast 90°", "Contrast 135°",
                        "Corr 0°", "Corr 45°", "Corr 90°", "Corr 135°",
                        "Energy 0°", "Energy 45°", "Energy 90°", "Energy 135°",
                        "Homogeneity",
                        "Metric (Circularity)", "Eccentricity"
                    ],
                    "Nilai Ekstraksi": [round(f, 4) for f in features]
                })

                st.dataframe(df_features, use_container_width=True, height=400)

# ==========================================
# 7. MODE 3: KOMPARASI & EVALUASI
# ==========================================
elif engine_choice == "Komparasi & Evaluasi Model":
    st.subheader("📈 Perbandingan Performa Metode Klasik vs Deep Learning")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### 1. Metode Klasik JTIM")
        st.markdown("""
        * **Ekstraksi Fitur:** 21 Fitur Manual (6 HSV, 13 GLCM, 2 Bentuk)
        * **Klasifikasi:** Logistic Regression
        * **Akurasi Test Set:** **100%**
        """)
        if os.path.exists("confusion_matrix_klasik.png"):
            st.image("confusion_matrix_klasik.png", caption="Confusion Matrix - Metode Klasik JTIM", use_container_width=True)

    with col2:
        st.write("### 2. Deep Learning YOLOv8")
        st.markdown("""
        * **Ekstraksi Fitur:** Otomatis via CNN (CSPDarknet/PAN-FPN)
        * **Tipe Deteksi:** Bounding Box + Class + Confidence Score
        * **mAP@50:** **98.26%**
        """)
        if os.path.exists("runs/detect/yolov8_ketupat_optimal/confusion_matrix.png"):
            st.image("runs/detect/yolov8_ketupat_optimal/confusion_matrix.png", caption="Confusion Matrix - YOLOv8", use_container_width=True)
        elif os.path.exists("runs/detect/train/confusion_matrix.png"):
            st.image("runs/detect/train/confusion_matrix.png", caption="Confusion Matrix - YOLOv8", use_container_width=True)