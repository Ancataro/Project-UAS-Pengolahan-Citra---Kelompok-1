import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def visualize_results(video_path, df, output_video_path, progress_callback=None):
    if df is None or df.empty:
        print("Error: DataFrame hasil deteksi kosong. Tidak bisa melakukan visualisasi.")
        return

    print(f"Membaca video asli dari {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Tidak bisa membuka video {video_path}")
        return

    # Ambil spesifikasi video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolusi Video: {width}x{height}, FPS: {fps}, Total Frame: {total_frames}")

    # Cari berkas font TrueType di Windows yang mendukung aksara Thai & Latin
    font_paths = [
        "C:\\Windows\\Fonts\\tahoma.ttf",     # Mendukung banyak bahasa termasuk Thai
        "C:\\Windows\\Fonts\\Leelawad.ttf",   # Font standard Windows untuk Thai
        "C:\\Windows\\Fonts\\Leelawdb.ttf",   # Font Leelawadee Bold
        "C:\\Windows\\Fonts\\arial.ttf",      # Arial Unicode
        "C:\\Windows\\Fonts\\angsau.ttf"      # Angsana New
    ]
    
    font_path = None
    for path in font_paths:
        if os.path.exists(path):
            font_path = path
            break

    if font_path:
        print(f"Menggunakan font sistem: {font_path}")
    else:
        print("Peringatan: Font TrueType tidak ditemukan. Menggunakan font default PIL (beberapa karakter mungkin tidak terbaca).")

    # Inisialisasi ukuran font menggunakan Pillow
    # Kita sesuaikan ukuran berdasarkan resolusi video agar proporsional
    scale_factor = width / 1280.0
    size_label = int(max(14, 18 * scale_factor))
    size_hud_title = int(max(12, 16 * scale_factor))
    size_hud_row = int(max(10, 14 * scale_factor))

    try:
        font_label = ImageFont.truetype(font_path, size_label) if font_path else ImageFont.load_default()
        font_hud_title = ImageFont.truetype(font_path, size_hud_title, encoding="utf-8") if font_path else ImageFont.load_default()
        font_hud_row = ImageFont.truetype(font_path, size_hud_row, encoding="utf-8") if font_path else ImageFont.load_default()
    except Exception as e:
        print(f"Gagal memuat font TrueType ({e}), menggunakan font default.")
        font_label = ImageFont.load_default()
        font_hud_title = ImageFont.load_default()
        font_hud_row = ImageFont.load_default()

    # Inisialisasi video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_nmr = -1
    ret = True
    active_plates = {}
    frames_to_persist = max(15, 5 * fps)  # Minimal 15 frame, default 5 detik

    print("Membuat video visualisasi... Silakan tunggu.")

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        
        if not ret:
            break

        # Filter data untuk frame saat ini
        frame_df = df[df['frame_nmr'] == frame_nmr]

        # 1. TAHAP GAMBAR KOTAK (OpenCV)
        for _, row in frame_df.iterrows():
            car_id = int(row['car_id'])
            
            # Koordinat kendaraan
            car_x1, car_y1, car_x2, car_y2 = (
                int(row['car_x1']), int(row['car_y1']),
                int(row['car_x2']), int(row['car_y2'])
            )

            # Koordinat plat nomor
            plate_x1, plate_y1, plate_x2, plate_y2 = (
                int(row['license_plate_x1']), int(row['license_plate_y1']),
                int(row['license_plate_x2']), int(row['license_plate_y2'])
            )

            # Teks nomor plat
            license_text = str(row['license_number']).strip()
            if pd.isna(row['license_number']) or license_text.lower() in ['nan', '']:
                display_text = f"ID: {car_id} | Mendeteksi..."
            else:
                display_text = f"ID: {car_id} | {license_text}"
                active_plates[car_id] = (license_text, frame_nmr)

            # A. Gambar Bounding Box Kendaraan (Hijau - tebal 3)
            cv2.rectangle(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 3)

            # B. Gambar Bounding Box Plat Nomor (Kuning - tebal 3)
            cv2.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 255, 255), 3)

            # C. Hitung Ukuran Teks untuk Background Label
            try:
                left, top, right, bottom = font_label.getbbox(display_text)
                text_w = right - left
                text_h = bottom - top
            except AttributeError:
                # Fallback untuk Pillow versi lama
                text_w, text_h = font_label.getsize(display_text)

            # Koordinat background label di atas kendaraan
            label_x1 = car_x1
            label_y1 = max(10, car_y1 - text_h - 15)
            label_x2 = car_x1 + text_w + 12
            label_y2 = car_y1
            label_x2 = min(width, label_x2)

            # Gambar label background (Kuning Terang solid)
            cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (0, 255, 255), cv2.FILLED)
            # Gambar outline hitam
            cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (0, 0, 0), 1)

        # 2. TAHAP GAMBAR HUD/OSD (OpenCV)
        # Bersihkan plat lama yang sudah lewat dari durasi tampil (frames_to_persist)
        active_plates = {cid: (txt, f_num) for cid, (txt, f_num) in active_plates.items() if frame_nmr - f_num < frames_to_persist}

        hud_drawn = False
        card_x1, card_y1, card_x2, card_y2 = 0, 0, 0, 0
        if active_plates:
            hud_drawn = True
            card_w = int(max(240, 280 * scale_factor))
            card_h = int(35 + len(active_plates) * (25 * scale_factor) + 10)
            card_x1 = max(0, width - card_w - 20)
            card_y1 = 20
            card_x2 = width - 20
            card_y2 = min(height, card_y1 + card_h)
            
            # Buat overlay hitam transparan
            overlay = frame.copy()
            cv2.rectangle(overlay, (card_x1, card_y1), (card_x2, card_y2), (20, 20, 20), cv2.FILLED)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            # Gambar outline kuning terang
            cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (0, 255, 255), 2)

        # 3. TAHAP GAMBAR TEKS UNICODE (Pillow)
        # Konversi OpenCV (BGR) ke PIL (RGB) untuk menggambar teks Unicode dengan benar
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # A. Tulis Teks di Atas Kendaraan
        for _, row in frame_df.iterrows():
            car_id = int(row['car_id'])
            car_x1 = int(row['car_x1'])
            car_y1 = int(row['car_y1'])

            license_text = str(row['license_number']).strip()
            if pd.isna(row['license_number']) or license_text.lower() in ['nan', '']:
                display_text = f"ID: {car_id} | Mendeteksi..."
            else:
                display_text = f"ID: {car_id} | {license_text}"

            try:
                left, top, right, bottom = font_label.getbbox(display_text)
                text_h = bottom - top
            except AttributeError:
                _, text_h = font_label.getsize(display_text)

            # Posisi teks tepat di atas box kendaraan
            text_pos = (car_x1 + 6, car_y1 - text_h - 10)
            draw.text(text_pos, display_text, font=font_label, fill=(0, 0, 0))

        # B. Tulis Teks di Panel HUD
        if hud_drawn:
            # Header HUD
            draw.text((card_x1 + 10, card_y1 + 8), "PLAT TERDETEKSI (5 DETIK):", font=font_hud_title, fill=(0, 255, 255))
            
            # Baris plat aktif
            sorted_plates = sorted(active_plates.items(), key=lambda x: x[1][1], reverse=True)
            for idx, (cid, (txt, f_num)) in enumerate(sorted_plates):
                remaining_frames = frames_to_persist - (frame_nmr - f_num)
                remaining_sec = max(0.0, remaining_frames / fps)
                row_text = f"ID {cid}: {txt} ({remaining_sec:.1f}s)"
                y_pos = int(card_y1 + 35 + idx * (25 * scale_factor))
                draw.text((card_x1 + 10, y_pos), row_text, font=font_hud_row, fill=(255, 255, 255))

        # Konversi kembali PIL (RGB) ke OpenCV (BGR)
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Tulis frame ke video output
        out.write(frame)

        # Cetak progress setiap 50 frame
        if frame_nmr % 50 == 0 or frame_nmr == total_frames - 1:
            print(f"Menulis Frame Visualisasi: {frame_nmr}/{total_frames}")
            if progress_callback:
                progress_callback(frame_nmr + 1, total_frames, "Membuat Video Visualisasi")

    # Selesai, tutup semua resources
    cap.release()
    out.release()
    print(f"Video visualisasi berhasil disimpan di {output_video_path}")

if __name__ == '__main__':
    # Fallback pengujian langsung dengan membaca file CSV lama jika ada
    import pandas as pd
    try:
        print("Menjalankan visualisasi pengujian dari file CSV lokal...")
        df_test = pd.read_csv('./results_interpolated.csv', encoding='utf-8')
        visualize_results('./sample.mp4', df_test, './output.mp4')
    except Exception as e:
        print(f"Gagal menjalankan visualisasi test: {e}")
        print("Silakan jalankan pipeline utama melalui gui.py terlebih dahulu.")
