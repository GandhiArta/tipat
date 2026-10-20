from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s.pt')

    model.train(
        data='Ketupat.v2i.yolov8/data.yaml',
        epochs=100,
        imgsz=800,
        batch=8,
        device=0,
        
        #PARAMETER AUGMENTASI YOLOV8
        mosaic=1.0,       #Menggabungkan 4 gambar jadi 1 (Sangat bagus untuk objek berdekatan/bergelombol)
        mixup=0.1,        #Menumpuk 2 gambar transparan (Membantu model mengenali objek saling menutupi)
        degrees=15.0,     #Memutar/rotasi gambar acak sebesar -15 hingga +15 derajat
        fliplr=0.5,       #Membalik gambar secara horizontal (kiri-kanan) sebesar 50% peluang
        scale=0.5,        #Mengubah skala ukuran objek (zoom in/out acak 50%)
        hsv_h=0.015,      #Variasi rona warna janur/daun (Hue)
        hsv_s=0.7,        #Variasi kejenuhan warna janur (Saturation)
        hsv_v=0.4,        #Variasi pencahayaan/kecerahan (Value) — Bagus untuk foto kondisi gelap/malam
        
        name='yolov8_ketupat_v4'
    )