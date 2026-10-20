import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Deteksi Tipat Bali - YOLOv8",
    page_icon="🍙",
    layout="wide"
)

st.title("🍙 Deteksi Otomatis Tipat Bali")
st.markdown("Aplikasi Pengolahan Citra Digital berbasis **Deep Learning (YOLOv8)**.")

# ==========================================
# 2. LOAD MODEL YOLOV8 (VALIDASI KETAT)
# ==========================================
@st.cache_resource
def load_yolo_model():
    possible_paths = [
        "runs/detect/yolov8_ketupat_optimal/weights/best.pt",  # Model lama yang aktif
        "runs/detect/yolov8_ketupat_v3-2/weights/best.pt",
        "best.pt"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return YOLO(path), path
            
    return None, None
yolo_model, model_path = load_yolo_model()

# Memastikan hanya model buatanmu yang digunakan
if yolo_model is not None:
    st.toast(f"✓ Model khusus aktif: {model_path}", icon="✅")
else:
    st.error("🚨 KRITIS: File `best.pt` hasil training kamu tidak ditemukan!")
    st.info("Salin file `best.pt` dari folder `runs/detect/.../weights/` lalu tempel (paste) ke folder utama proyek `Ketupat`.")
    st.stop()

# ==========================================
# 3. SIDEBAR PARAMETER
# ==========================================
st.sidebar.header("⚙️ Pengaturan Model")
conf_threshold = st.sidebar.slider(
    "Batas Kepercayaan (Confidence):", 
    min_value=0.05, 
    max_value=1.00, 
    value=0.25, 
    step=0.05,
    help="Turunkan nilai ini jika Tipat tidak terdeteksi, atau naikkan jika ada deteksi palsu."
)

# ==========================================
# 4. TAMPILAN UTAMA & DETEKSI
# ==========================================
st.caption("Unggah foto utuh berisi Tipat Bali. Model akan mendeteksi posisi dan jenis tipat secara otomatis.")

uploaded_file = st.file_uploader("Unggah foto Tipat Bali:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Fix RGBA: Konversi ke RGB agar tidak error 4-channels pada file PNG
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Gambar Asli", use_container_width=True)

    with col2:
        st.write("### Hasil Deteksi:")
        img_np = np.array(image)
        
        # 2. Run YOLOv8 inference dengan TTA (augment=True) untuk hasil presisi tinggi
        results = yolo_model(img_np, conf=conf_threshold, augment=True)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption=f"Hasil Bounding Box YOLOv8 (Conf: {conf_threshold})", use_container_width=True)