import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. KONFIGURASI PATH DATASET COCO
# ==========================================
DATASET_DIR = r"Ketupat_COCO"  # <-- Ganti dengan nama folder dataset COCO kamu
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train")
ANOTASI_COCO_PATH = os.path.join(DATASET_DIR, "train", "_annotations.coco.json")

print("=== MEMULAI PROSES KLASIFIKASI KLASIK (JTIM) - FORMAT COCO ===")

# ==========================================
# 2. LOAD ANOTASI COCO & MAPPING KELAS
# ==========================================
if os.path.exists(ANOTASI_COCO_PATH):
    with open(ANOTASI_COCO_PATH, 'r') as f:
        coco_data = json.load(f)
    print("✓ Berhasil memuat anotasi COCO JSON.")
else:
    print(f"[X] File anotasi tidak ditemukan di {ANOTASI_COCO_PATH}")
    print("Pastikan folder Ketupat_COCO sudah diekstrak dan berisi _annotations.coco.json")
    exit()

# Map ID kategori ke nama kelas
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
images = {img['id']: img['file_name'] for img in coco_data['images']}

# ==========================================
# 3. EXTRACTION FUNCTIONS (HSV, GLCM, BENTUK)
# ==========================================

def extract_hsv_features(img_bgr):
    """Ekstraksi 6 Fitur Warna HSV (Mean & Std Dev)"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_h, std_h = np.mean(hsv[:, :, 0]), np.std(hsv[:, :, 0])
    mean_s, std_s = np.mean(hsv[:, :, 1]), np.std(hsv[:, :, 1])
    mean_v, std_v = np.mean(hsv[:, :, 2]), np.std(hsv[:, :, 2])
    return [mean_h, std_h, mean_s, std_s, mean_v, std_v]

def extract_glcm_features(img_bgr):
    """Ekstraksi 13 Fitur Tekstur GLCM (Contrast, Correlation, Energy 4 Sudut + Homogeneity)"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast').ravel()       # 4 fitur
    correlation = graycoprops(glcm, 'correlation').ravel() # 4 fitur
    energy = graycoprops(glcm, 'energy').ravel()           # 4 fitur
    homogeneity = graycoprops(glcm, 'homogeneity').mean()  # 1 fitur
    
    return [*contrast, *correlation, *energy, homogeneity]

def extract_shape_features(crop_bgr):
    """Ekstraksi 2 Fitur Bentuk (Metric & Eccentricity)"""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0, 0]
    
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
        
    return [metric, eccentricity]

# ==========================================
# 4. CROP OBJEK & EKSTRAKSI 21 FITUR
# ==========================================
X_list = []
y_list = []

print("✓ Memproses gambar dan anotasi COCO...")

for ann in coco_data['annotations']:
    img_id = ann['image_id']
    cat_id = ann['category_id']
    bbox = ann['bbox']  # [x, y, width, height] dalam satuan piksel
    
    if img_id not in images:
        continue
        
    img_name = images[img_id]
    img_path = os.path.join(TRAIN_IMG_DIR, img_name)
    
    if not os.path.exists(img_path):
        continue
        
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    # Crop objek berdasarkan Bounding Box COCO
    x, y, w, h = [int(v) for v in bbox]
    x, y = max(0, x), max(0, y)
    crop = img[y:y+h, x:x+w]
    
    if crop.size == 0:
        continue
        
    # Ekstraksi 21 Fitur
    f_hsv = extract_hsv_features(crop)     # 6 fitur
    f_glcm = extract_glcm_features(crop)   # 13 fitur
    f_shape = extract_shape_features(crop) # 2 fitur
    
    all_features = f_hsv + f_glcm + f_shape
    
    X_list.append(all_features)
    y_list.append(categories[cat_id])

X = np.array(X_list)
y = np.array(y_list)

print(f"✓ Berhasil mengekstraksi {X.shape[0]} sampel dengan {X.shape[1]} fitur.")

# ==========================================
# 5. DATA CLEANING & PREPARATION
# ==========================================
X = np.nan_to_num(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 6. TRAINING LOGISTIC REGRESSION
# ==========================================
print("✓ Melatih model Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# ==========================================
# 7. EVALUASI DAN CONFUSION MATRIX
# ==========================================
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*40)
print(f" AKURASI MODEL KLASIK: {acc * 100:.2f}%")
print("="*40 + "\n")

print("--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix - Metode Klasik COCO (Logistic Regression)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix_klasik.png')
print("✓ Confusion Matrix disimpan ke 'confusion_matrix_klasik.png'")
plt.show() 

import joblib

# Simpan model dan scaler agar bisa dipanggil di Streamlit app.py
joblib.dump(model, 'model_klasik.pkl')
joblib.dump(scaler, 'scaler_klasik.pkl')
print("✓ Model dan Scaler Klasik berhasil disimpan!")