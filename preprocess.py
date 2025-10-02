import cv2
import numpy as np

def remove_border(img, border_percent=0.05):
    """
    Buang border/frame plat nomor
    border_percent: % dari width/height yang dibuang
    """
    h, w = img.shape[:2]
    border_h = int(h * border_percent)
    border_w = int(w * border_percent)
    
    # Crop semua sisi
    cropped = img[border_h:h-border_h, border_w:w-border_w]
    return cropped

def remove_date_section(img, keep_ratio=0.60):
    """
    Buang bagian bawah plat (expired date/tahun-bulan)
    keep_ratio: berapa % dari atas yang diambil (default 60%)
    """
    h = img.shape[0]
    cropped = img[0:int(h*keep_ratio), :]
    return cropped

def check_blur(img, threshold=100.0):
    """
    Check apakah gambar blur menggunakan Laplacian variance
    Return: True = sharp (OK), False = blur (skip)
    threshold: makin tinggi = makin strict
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > threshold

def check_brightness(img, min_brightness=30, max_brightness=220):
    """
    Check apakah gambar terlalu gelap atau terlalu terang
    Return: True = OK, False = skip
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    mean_brightness = np.mean(gray)
    return min_brightness < mean_brightness < max_brightness

def check_image_quality(img, blur_threshold=100.0):
    """
    Comprehensive quality check
    Return: (is_ok, blur_score, brightness)
    """
    is_sharp = check_blur(img, blur_threshold)
    is_bright_ok = check_brightness(img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    
    is_ok = is_sharp and is_bright_ok
    
    return is_ok, blur_score, brightness

def preprocess_plate_v1(img):
    """
    Preprocessing method 1: Adaptive threshold + morphology
    """
    # Resize 2x untuk OCR lebih jelas
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter (buang noise, jaga edge)
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        bilateral, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    
    # Morphological closing
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph

def preprocess_plate_v2(img):
    """
    Preprocessing method 2: CLAHE + Otsu threshold
    """
    # Resize
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Otsu threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def preprocess_plate_v3(img):
    """
    Preprocessing method 3: Unsharp mask + adaptive
    """
    # Resize
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Unsharp masking
    gaussian = cv2.GaussianBlur(gray, (9, 9), 10.0)
    unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        unsharp, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15, 5
    )
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    
    return morph

def preprocess_plate_v0(img):
    """
    Preprocessing method 0: Minimal processing with dilation to connect characters
    """
    # Only remove date section, no border removal
    h = img.shape[0]
    no_date = img[0:int(h*0.65), :]
    
    # Resize 3x for better OCR
    img_large = cv2.resize(no_date, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
    
    # Simple adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    
    # Dilate slightly to make characters thicker and more connected
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    return thresh

def preprocess_plate_raw(img):
    """
    Preprocessing method RAW: Almost no processing - just resize
    Best for detecting separated characters
    """
    # Only remove date section
    h = img.shape[0]
    no_date = img[0:int(h*0.65), :]
    
    # Resize 4x for maximum clarity
    img_large = cv2.resize(no_date, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    # Convert to BGR if needed (PaddleOCR expects BGR)
    if len(img_large.shape) == 2:
        img_large = cv2.cvtColor(img_large, cv2.COLOR_GRAY2BGR)
    
    return img_large

def preprocess_plate(img):
    """
    Main preprocessing pipeline dengan 5 variations (added raw + v0)
    Return: list of preprocessed images untuk voting
    """
    # Version RAW: Almost original (best for separated chars)
    processed_raw = preprocess_plate_raw(img)
    
    # Version 0: Minimal processing (no border removal)
    processed_v0 = preprocess_plate_v0(img)
    
    # Versions 1-3: With border removal
    no_border = remove_border(img, border_percent=0.02)
    no_date = remove_date_section(no_border, keep_ratio=0.70)
    
    processed_images = [
        processed_raw,  # Try raw first
        processed_v0,   # Then minimal
        preprocess_plate_v1(no_date),
        preprocess_plate_v2(no_date),
        preprocess_plate_v3(no_date)
    ]
    
    return processed_images