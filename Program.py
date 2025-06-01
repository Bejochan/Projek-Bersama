import cv2
import torch
import numpy as np
from ultralytics import YOLO

def detect_mask_webcam():
    """
    Melakukan deteksi masker wajah secara real-time menggunakan YOLOv5 dan webcam.
    """
    
    # 1. Muat model YOLOv5
    # Jika Anda melatih model sendiri, ganti 'yolov5s.pt' dengan path ke model Anda (misalnya, 'runs/train/expX/weights/best.pt')
    try:
        model = YOLO('yolov5su.pt') 
        print("Model YOLOv5 berhasil dimuat.")
    except Exception as e:
        print(f"Gagal memuat model YOLOv5: {e}")
        print("Pastikan Anda memiliki koneksi internet untuk mengunduh model atau path model Anda benar.")
        return

    # Anda bisa menentukan kelas-kelas yang ingin dideteksi jika Anda melatih model kustom
    # Contoh untuk deteksi masker:
    # classes = ['no_mask', 'mask'] 
    # model.names akan berisi nama kelas dari model yang dimuat secara otomatis
    
    # 2. Inisialisasi webcam
    cap = cv2.VideoCapture(0)  # 0 biasanya adalah ID untuk webcam internal laptop Anda
    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam. Pastikan webcam terhubung dan tidak digunakan oleh aplikasi lain.")
        return

    print("Memulai deteksi real-time. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil frame dari webcam.")
            break

        # 3. Lakukan inferensi (deteksi) pada frame
        # Conf: Ambang batas kepercayaan. Objek dengan kepercayaan di bawah nilai ini akan diabaikan.
        # IOU: Intersection Over Union. Ambang batas untuk Non-Maximum Suppression.
        results = model(frame, conf=0.5, iou=0.45) 
        
        # 4. Proses hasil deteksi
        # Iterasi melalui setiap objek yang terdeteksi
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # Koordinat bounding box (x1, y1, x2, y2)
            scores = r.boxes.conf.cpu().numpy()  # Skor kepercayaan
            class_ids = r.boxes.cls.cpu().numpy()  # ID kelas

            for box, score, class_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(class_id)] # Dapatkan nama kelas dari ID
                confidence = float(score)

                # Tentukan warna berdasarkan kelas (misalnya, hijau untuk masker, merah untuk tidak bermasker)
                color = (0, 255, 0) if label == 'mask' else (0, 0, 255) # Asumsi 'mask' adalah kelas untuk masker, 'no_mask' untuk tanpa masker
                
                # Gambar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Teks label dan kepercayaan
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 5. Tampilkan frame dengan hasil deteksi
        cv2.imshow('Deteksi Masker Real-time (YOLOv5)', frame)

        # 6. Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. Bersihkan sumber daya
    cap.release()
    cv2.destroyAllWindows()
    print("Deteksi real-time dihentikan.")

detect_mask_webcam()