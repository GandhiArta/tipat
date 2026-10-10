from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Load Pretrained YOLOv8 Nano
    model = YOLO('yolov8n.pt')

    # 2. Training dengan Konfigurasi Optimal
    results = model.train(
        data='Ketupat.v2i.yolov8/data.yaml',  # Path ke file data.yaml
        epochs=100,                            # Ditingkatkan agar konvergensi optimal
        patience=20,                           # Early stopping jika tidak ada peningkatan dalam 20 epoch
        imgsz=640,                             # Resolusi standar YOLOv8
        batch=16,                              # Ukuran batch stabil untuk GPU
        device=0,                              # GPU NVIDIA
        workers=4,                             # Mempercepat pembacaan dataset
        seed=42,                               # Menjaga konsistensi eksperimen (reproducible)
        optimizer='SGD',                       # Optimizer standar yang stabil
        lr0=0.01,                              # Learning rate awal
        plots=True,                            # Menyimpan grafik evaluasi (mAP, Loss, Confusion Matrix)
        save=True,                             # Menyimpan bobot checkpoint
        name='yolov8_ketupat_optimal'          # Nama folder hasil eksport
    )

    # 3. Validasi Akhir pada Split Validation
    metrics = model.val()
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")