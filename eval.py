from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Muat model terbaik yang ingin dievaluasi
    model = YOLO('runs/detect/yolov8_ketupat_optimal/weights/best.pt')

    # 2. Jalankan proses validasi/evaluasi
    metrics = model.val(
        data='Ketupat.v2i.yolov8/data.yaml',
        imgsz=800,
        batch=8,
        split='val'  # Bisa diganti 'test' jika ingin menguji dataset test
    )

    # 3. Tampilkan metrik utama di terminal
    print("\n--- HASIL EVALUASI MODEL ---")
    print(f"Precision (P) : {metrics.box.mp:.4f}")
    print(f"Recall (R)    : {metrics.box.mr:.4f}")
    print(f"mAP@50        : {metrics.box.map50:.4f}")
    print(f"mAP@50-95     : {metrics.box.map:.4f}")