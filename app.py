import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Deteksi Tipat Bali", layout="centered")
st.title("Aplikasi Deteksi Jenis Tipat Bali")

# Sidebar
st.sidebar.header("Pengaturan Deteksi")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.01, 
    max_value=1.00, 
    value=0.15, 
    step=0.01
)

# Load Model
@st.cache_resource
def load_model():
    model_path = r'runs\detect\yolov8_ketupat_optimal\weights\best.pt'
    return YOLO(model_path)

model = load_model()

uploaded_file = st.file_uploader("Pilih gambar tipat...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)

    # Jalankan Prediksi Langsung tanpa Tombol Tambahan
    results = model.predict(source=image, conf=conf_threshold)
    res_plotted = results[0].plot()

    with col2:
        st.subheader("Hasil Deteksi")
        st.image(res_plotted, use_container_width=True)

    # Detail Bounding Box
    boxes = results[0].boxes
    if len(boxes) > 0:
        st.success(f"Terdeteksi {len(boxes)} objek:")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            st.write(f"- **{class_name}** (Confidence: {conf*100:.2f}%)")
    else:
        st.warning("Objek tidak terdeteksi pada threshold saat ini.")