import csv
import re
import cv2
import numpy as np

# Mapping dictionaries for correcting common OCR errors
dict_char_to_int = {
    'O': '0', 'I': '1', 'J': '1', 'L': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8', 'T': '7', 'A': '4'
}
dict_int_to_char = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B', '7': 'T', '4': 'A'
}

def contains_thai(text):
    """Checks if the text contains Thai script characters."""
    return any('\u0e00' <= c <= '\u0e7f' for c in text)

def format_license_plate(text):
    """
    Format license plate string. Supports Indonesian plate format and Thai plates.
    """
    # If the text contains Thai script, clean and return it directly
    if contains_thai(text):
        cleaned = "".join(c for c in text if '\u0e00' <= c <= '\u0e7f' or c.isalnum() or c.isspace())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().upper()
        return cleaned if len(cleaned.replace(" ", "")) >= 3 else None

    # Clean text: keep only alphanumeric characters and convert to uppercase
    text = ''.join(c.upper() for c in text if c.isalnum())
    n = len(text)
    
    # Valid Indonesian plates are typically between 3 and 9 characters long
    if n < 3 or n > 9:
        return None
        
    # Try all valid combinations of prefix, number, suffix lengths
    # Prefix: 1-2 letters
    # Suffix: 1-3 letters
    # Number: 1-4 digits
    for len_prefix in [1, 2]:
        for len_suffix in [1, 2, 3]:
            len_num = n - len_prefix - len_suffix
            if 1 <= len_num <= 4:
                prefix = text[:len_prefix]
                num = text[len_prefix:len_prefix+len_num]
                suffix = text[len_prefix+len_num:]
                
                # Correct the parts using mapping dictionaries
                corrected_prefix = "".join(dict_int_to_char.get(c, c) for c in prefix)
                corrected_num = "".join(dict_char_to_int.get(c, c) for c in num)
                corrected_suffix = "".join(dict_int_to_char.get(c, c) for c in suffix)
                
                # Verify that corrected parts are of the correct type
                if all(c.isalpha() for c in corrected_prefix) and \
                   all(c.isdigit() for c in corrected_num) and \
                   all(c.isalpha() for c in corrected_suffix):
                    
                    # We found a valid format! Return formatted string
                    return f"{corrected_prefix} {corrected_num} {corrected_suffix}"
                    
    # Fallback: find first index that is a digit or digit-like, and last index
    digits = [i for i, c in enumerate(text) if c.isdigit() or c in dict_char_to_int]
    if digits:
        first_d = digits[0]
        last_d = digits[-1]
        
        # Ensure prefix has at least 1 character
        if first_d == 0:
            first_d = 1
        # Ensure suffix has at least 1 character
        if last_d == n - 1:
            last_d = n - 2
            
        if first_d <= last_d:
            prefix = text[:first_d]
            num = text[first_d:last_d+1]
            suffix = text[last_d+1:]
            
            corrected_prefix = "".join(dict_int_to_char.get(c, c) for c in prefix)
            corrected_num = "".join(dict_char_to_int.get(c, c) for c in num)
            corrected_suffix = "".join(dict_int_to_char.get(c, c) for c in suffix)
            
            if all(c.isalpha() for c in corrected_prefix) and \
               all(c.isdigit() for c in corrected_num) and \
               all(c.isalpha() for c in corrected_suffix):
                return f"{corrected_prefix} {corrected_num} {corrected_suffix}"

    return None

def get_car(license_plate_box, vehicle_track_ids):
    """
    Finds the vehicle that contains the license plate.
    license_plate_box: [x1, y1, x2, y2, score]
    vehicle_track_ids: list of [x1, y1, x2, y2, track_id]
    """
    x1, y1, x2, y2, _ = license_plate_box
    
    best_car_id = -1
    max_overlap = 0
    
    for vehicle in vehicle_track_ids:
        vx1, vy1, vx2, vy2, track_id = vehicle
        
        # Calculate intersection rectangle
        ix1 = max(x1, vx1)
        iy1 = max(y1, vy1)
        ix2 = min(x2, vx2)
        iy2 = min(y2, vy2)
        
        if ix1 < ix2 and iy1 < iy2:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            # Pick the vehicle with the largest overlap area
            if intersection_area > max_overlap:
                max_overlap = intersection_area
                best_car_id = track_id
                
    return best_car_id

def read_license_plate(license_plate_crop, reader):
    """
    Crop, pre-process, and OCR the license plate image.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize to double the size (helps OCR read small text)
    gray = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # 3. Apply thresholding (Otsu's thresholding)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Run EasyOCR on thresholded image
    ocr_results = reader.readtext(thresh)
    
    # If OTSU thresholding yielded no text, fallback to grayscale
    if not ocr_results:
        ocr_results = reader.readtext(gray)
        
    if not ocr_results:
        return None, None
        
    # Sort detections by their x-coordinate (left to right)
    ocr_results.sort(key=lambda x: x[0][0][0])
    
    full_text = " ".join([res[1] for res in ocr_results])
    avg_score = np.mean([res[2] for res in ocr_results])
    
    # Try to format Indonesian license plate
    formatted_text = format_license_plate(full_text)
    if formatted_text is not None:
        return formatted_text, avg_score
        
    # Fallback to simple alphanumeric cleaning
    clean_text = "".join(c.upper() for c in full_text if c.isalnum() or c == " ")
    if len(clean_text.replace(" ", "")) >= 3:
        return clean_text.strip(), avg_score
        
    return None, None

def results_to_df(results):
    """
    Converts results dict to pandas DataFrame.
    results format: {frame_nmr: {car_id: {'car_bbox': [...], 'license_plate': {'bbox': [...], 'text': '...', 'bbox_score': ..., 'text_score': ...}}}}
    """
    import pandas as pd
    rows = []
    
    for frame_nmr in sorted(results.keys()):
        for car_id in results[frame_nmr].keys():
            car_bbox = results[frame_nmr][car_id]['car_bbox']
            plate = results[frame_nmr][car_id]['license_plate']
            
            rows.append({
                'frame_nmr': frame_nmr,
                'car_id': car_id,
                'car_x1': car_bbox[0],
                'car_y1': car_bbox[1],
                'car_x2': car_bbox[2],
                'car_y2': car_bbox[3],
                'license_plate_x1': plate['bbox'][0],
                'license_plate_y1': plate['bbox'][1],
                'license_plate_x2': plate['bbox'][2],
                'license_plate_y2': plate['bbox'][3],
                'license_plate_score': plate['bbox_score'],
                'license_number': plate['text'],
                'license_number_score': plate['text_score']
            })
            
    return pd.DataFrame(rows)
