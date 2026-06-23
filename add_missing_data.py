import pandas as pd
import numpy as np

def interpolate_missing_data(df):
    if df is None or df.empty:
        print("Data mentah kosong. Tidak ada data untuk diinterpolasi.")
        return df

    print("Melakukan interpolasi data yang hilang...")
    car_ids = df['car_id'].unique()
    interpolated_rows = []

    for car_id in car_ids:
        # Filter data untuk kendaraan tertentu
        car_df = df[df['car_id'] == car_id].sort_values(by='frame_nmr')
        
        min_frame = int(car_df['frame_nmr'].min())
        max_frame = int(car_df['frame_nmr'].max())
        
        # Buat daftar lengkap frame dari min_frame sampai max_frame
        all_frames = list(range(min_frame, max_frame + 1))
        
        # Atur ulang index berdasarkan frame agar frame yang hilang terisi dengan NaN
        car_df = car_df.set_index('frame_nmr').reindex(all_frames)
        
        # Isi kembali car_id
        car_df['car_id'] = car_id
        
        # Daftar kolom koordinat bounding box
        bbox_cols = [
            'car_x1', 'car_y1', 'car_x2', 'car_y2',
            'license_plate_x1', 'license_plate_y1', 'license_plate_x2', 'license_plate_y2'
        ]
        
        # Interpolasi linier koordinat bounding box
        # Menggunakan limit_direction='both' agar data di ujung yang kosong juga terisi (forward/backward fill)
        car_df[bbox_cols] = car_df[bbox_cols].interpolate(method='linear').bfill().ffill()
        
        # Interpolasi score
        score_cols = ['license_plate_score', 'license_number_score']
        car_df[score_cols] = car_df[score_cols].interpolate(method='linear').bfill().ffill()
        
        # Untuk teks nomor plat, cari pembacaan terbaik (skor keyakinan OCR tertinggi)
        valid_plates = car_df.dropna(subset=['license_number'])
        if not valid_plates.empty:
            # Dapatkan baris dengan skor keyakinan OCR tertinggi
            best_plate_row = valid_plates.loc[valid_plates['license_number_score'].idxmax()]
            best_text = best_plate_row['license_number']
            best_ocr_score = best_plate_row['license_number_score']
            best_plate_score = best_plate_row['license_plate_score']
            
            # Terapkan teks nomor plat terbaik ini ke seluruh frame untuk mobil ini
            car_df['license_number'] = best_text
            car_df['license_number_score'] = car_df['license_number_score'].fillna(best_ocr_score)
            car_df['license_plate_score'] = car_df['license_plate_score'].fillna(best_plate_score)
        
        # Reset index agar frame_nmr kembali menjadi kolom biasa
        car_df = car_df.reset_index()
        interpolated_rows.append(car_df)

    # Gabungkan semua data hasil interpolasi
    out_df = pd.concat(interpolated_rows, ignore_index=True)
    
    # Urutkan berdasarkan frame_nmr dan car_id
    out_df = out_df.sort_values(by=['frame_nmr', 'car_id'])
    
    return out_df
