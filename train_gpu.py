from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Gunakan model yolov8s.pt (Small) untuk kapasitas fitur lebih tinggi
    model = YOLO('yolov8s.pt')

    # 2. Jalankan pelatihan optimal
    model.train(
        data='Ketupat.v2i.yolov8/data.yaml',
        epochs=100,             # Jumlah epoch ditingkatkan
        imgsz=800,              # Resolusi 800 agar detail tekstur janur jelas
        batch=8,                # Sesuaikan dengan VRAM GPU
        device='cpu',               # Menggunakan GPU RTX
        patience=20,            # Early stopping jika tidak ada peningkatan
        
        # Augmentasi Khusus Deteksi Objek Berdekatan
        mosaic=1.0,             # Menggabungkan 4 gambar agar belajar mendeteksi objek bergelombol
        degrees=15.0,           # Augmentasi rotasi
        hsv_v=0.4,              # Augmentasi pencahayaan
        scale=0.5,              # Variasi skala dekat/jauh
        
        name='yolov8_ketupat_v3' # Nama folder hasil training baru
    )