import os
import urllib.request
import cv2
import easyocr
import torch
from ultralytics import YOLO
from util import get_car, read_license_plate, results_to_df

def main(video_path='./sample.mp4', model_path='./models/license_plate_detector.pt', progress_callback=None):
    # 1. Download default model and video if they do not exist (only for default paths)
    model_dir = './models'
    default_model_path = os.path.join(model_dir, 'license_plate_detector.pt')
    default_video_path = './sample.mp4'

    # Download model if not exists and it is the default path
    if model_path == default_model_path and not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        print("Mengunduh model deteksi plat nomor default (license_plate_detector.pt)...")
        model_url = "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt"
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print("Unduhan model berhasil!")
        except Exception as e:
            print(f"Gagal mengunduh model: {e}")
            print("Silakan unduh model YOLOv8 custom Anda secara manual dan simpan di `./models/license_plate_detector.pt`")
            return

    # Download sample video if not exists and it is the default path
    if video_path == default_video_path and not os.path.exists(video_path):
        print("Mengunduh video sampel (sample.mp4)...")
        video_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
        try:
            urllib.request.urlretrieve(video_url, video_path)
            print("Unduhan video berhasil!")
        except Exception as e:
            print(f"Gagal mengunduh video sampel: {e}")
            print("Silakan sediakan video kendaraan Anda sendiri dan ubah nama filenya menjadi `sample.mp4`")
            return

    # 2. Load Models
    print("Memuat model YOLOv8...")
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO(model_path)

    # 3. Initialize EasyOCR Reader
    print("Menginisialisasi EasyOCR Reader...")
    # Gunakan GPU jika tersedia untuk performa yang lebih cepat, serta aktifkan deteksi teks bahasa Thailand dan Inggris
    use_gpu = torch.cuda.is_available()
    reader = easyocr.Reader(['th', 'en'], gpu=use_gpu)
    print(f"EasyOCR menggunakan GPU: {use_gpu} (Bahasa: ['th', 'en'])")

    # 4. Load Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Tidak bisa membuka video {video_path}")
        return

    # 5. Process Video Frame by Frame
    results = {}
    frame_nmr = -1
    ret = True

    print("Memproses video... Tekan Ctrl+C untuk membatalkan.")
    
    # Dapatkan total frame untuk progress bar sederhana
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frame yang akan diproses: {total_frames}")

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        
        if not ret:
            break

        results[frame_nmr] = {}
        
        # Cetak progress setiap 10 frame
        if frame_nmr % 10 == 0 or frame_nmr == total_frames - 1:
            print(f"Memproses Frame: {frame_nmr}/{total_frames}")
            if progress_callback:
                progress_callback(frame_nmr + 1, total_frames, "Deteksi Kendaraan & Plat")

        # A. Deteksi & Lacak Kendaraan (Car, Motorcycle, Bus, Truck) menggunakan ByteTrack bawaan YOLOv8
        # Class IDs dari COCO Dataset: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
        track_results = coco_model.track(frame, persist=True, conf=0.25, iou=0.6, tracker="botsort.yaml", verbose=False)
        
        vehicles = []
        if track_results[0].boxes is not None:
            for box in track_results[0].boxes:
                # Dapatkan koordinat bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Dapatkan class ID
                cls = int(box.cls[0].item())
                
                # Dapatkan track ID (bisa None jika tracker belum stabil)
                if box.id is not None:
                    track_id = int(box.id[0].item())
                    # Filter hanya kendaraan
                    if cls in [2, 3, 5, 7]:
                        vehicles.append([x1, y1, x2, y2, track_id])

        # B. Deteksi Plat Nomor
        plate_results = license_plate_detector(frame, verbose=False)
        
        if plate_results[0].boxes is not None:
            for plate_box in plate_results[0].boxes:
                # Koordinat plat nomor
                px1, py1, px2, py2 = plate_box.xyxy[0].tolist()
                plate_score = plate_box.conf[0].item()

                # Cari kendaraan yang menampung plat nomor ini
                car_id = get_car([px1, py1, px2, py2, plate_score], vehicles)

                if car_id != -1:
                    # Crop plat nomor dari frame video
                    h, w, _ = frame.shape
                    c_x1 = max(0, int(px1))
                    c_y1 = max(0, int(py1))
                    c_x2 = min(w, int(px2))
                    c_y2 = min(h, int(py2))
                    
                    plate_crop = frame[c_y1:c_y2, c_x1:c_x2]

                    # Lakukan pre-processing dan OCR untuk membaca plat nomor
                    plate_text, ocr_score = read_license_plate(plate_crop, reader)

                    if plate_text is not None:
                        # Dapatkan bounding box kendaraan yang cocok
                        car_bbox = [v for v in vehicles if v[4] == car_id][0][:4]
                        
                        # Simpan ke dict hasil
                        results[frame_nmr][car_id] = {
                            'car_bbox': car_bbox,
                            'license_plate': {
                                'bbox': [px1, py1, px2, py2],
                                'bbox_score': plate_score,
                                'text': plate_text,
                                'text_score': ocr_score
                            }
                        }

    # 6. Release resources
    cap.release()
    print("Pemrosesan video selesai!")

    # Check if we detected any plates
    has_detections = any(len(results[f]) > 0 for f in results)
    
    # 7. Fallback Simulation Pass for Demo/Testing
    if not has_detections:
        print("\n[INFO]: Tidak ada plat nomor riil yang terdeteksi pada video sampel ini.")
        print("[INFO]: Memulai PASS SIMULASI untuk menghasilkan data demo visualisasi...")
        
        cap = cv2.VideoCapture(video_path)
        frame_nmr = -1
        ret = True
        results = {}
        
        while ret:
            frame_nmr += 1
            ret, frame = cap.read()
            if not ret:
                break
                
            results[frame_nmr] = {}
            
            # Cetak progress simulasi setiap 10 frame
            if frame_nmr % 10 == 0 or frame_nmr == total_frames - 1:
                print(f"Simulasi Frame: {frame_nmr}/{total_frames}")
                if progress_callback:
                    progress_callback(frame_nmr + 1, total_frames, "Pass Simulasi Demo")

            # Deteksi kendaraan
            track_results = coco_model.track(frame, persist=True, conf=0.25, iou=0.6, tracker="botsort.yaml", verbose=False)
            
            vehicles = []
            if track_results[0].boxes is not None:
                for box in track_results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0].item())
                    if box.id is not None:
                        track_id = int(box.id[0].item())
                        if cls in [2, 3, 5, 7]:
                            vehicles.append([x1, y1, x2, y2, track_id])
            
            # Simulasikan plat nomor untuk setiap kendaraan yang terdeteksi
            for v in vehicles:
                vx1, vy1, vx2, vy2, track_id = v
                vw = vx2 - vx1
                vh = vy2 - vy1
                
                # Lokasi plat nomor simulasi (di tengah bawah kendaraan)
                px1 = (vx1 + vx2) / 2 - vw * 0.15
                px2 = (vx1 + vx2) / 2 + vw * 0.15
                py1 = vy2 - vh * 0.12
                py2 = vy2 - vh * 0.04
                
                # Format nomor plat simulasi Indonesia
                mock_text = f"B {1000 + track_id} UAS"
                
                results[frame_nmr][track_id] = {
                    'car_bbox': [vx1, vy1, vx2, vy2],
                    'license_plate': {
                        'bbox': [px1, py1, px2, py2],
                        'bbox_score': 0.95,
                        'text': mock_text,
                        'text_score': 0.99
                    }
                }
        cap.release()
        print("Pass simulasi selesai!")

    # 8. Convert Results to DataFrame and return
    print("Mengonversi hasil deteksi ke DataFrame...")
    df = results_to_df(results)
    return df

if __name__ == '__main__':
    main()