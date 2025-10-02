from paddleocr import PaddleOCR
from preprocess import preprocess_plate, check_image_quality
from postprocess import post_process_plate, validate_format, confidence_score
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import sys

# Initialize models
print("🔧 Loading models...")
yolo_model = YOLO("plate_detection_bestv1.pt")
# PaddleOCR with adjusted parameters for better small text detection
ocr = PaddleOCR(
    lang='en',
    det_db_thresh=0.2,  # Lower threshold (default 0.3) - detect more text
    det_db_box_thresh=0.4,  # Lower box threshold (default 0.6) - keep smaller boxes
    rec_batch_num=1  # Process one at a time for better accuracy
)
print("✅ Models loaded!\n")

def detect_plates(image):
    """Detect license plates using YOLOv10"""
    results = yolo_model(image, verbose=False)[0]
    
    plates = []
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            plate_img = image[y1:y2, x1:x2]
            
            if plate_img.size > 0:
                plates.append({
                    'image': plate_img,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence
                })
    
    return plates

def ocr_with_paddleocr(processed_img):
    """Run PaddleOCR on processed image"""
    try:
        # Convert grayscale to BGR if needed
        if len(processed_img.shape) == 2:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        
        # PaddleOCR predict returns a list with dict
        result = ocr.predict(processed_img)
        
        if result and isinstance(result, list) and len(result) > 0:
            res_dict = result[0]
            
            # Extract rec_texts and rec_scores
            if 'rec_texts' in res_dict and res_dict['rec_texts']:
                texts = res_dict['rec_texts']
                scores = res_dict.get('rec_scores', [])
                
                # Sort texts by position (left to right) if dt_polys available
                if 'dt_polys' in res_dict and res_dict['dt_polys']:
                    polys = res_dict['dt_polys']
                    # Sort by x-coordinate (leftmost first)
                    text_with_pos = [(texts[i], scores[i], polys[i][0][0]) for i in range(len(texts))]
                    text_with_pos.sort(key=lambda x: x[2])  # Sort by x position
                    texts = [t[0] for t in text_with_pos]
                    scores = [t[1] for t in text_with_pos]
                
                # Combine all text (in order)
                raw_text = ''.join(texts) if isinstance(texts, list) else str(texts)
                avg_confidence = float(np.mean(scores)) if scores and len(scores) > 0 else 0.8
                
                return raw_text, avg_confidence
        
        return None, 0.0
        
    except Exception as e:
        return None, 0.0

def recognize_plate_with_voting(plate_img, num_attempts=3):
    """
    Full pipeline with voting mechanism:
    1. Quality check
    2. Preprocessing (4 variations)
    3. OCR multiple times
    4. Post-processing
    5. Voting for best result
    """
    
    # 1. Quality check
    is_ok, blur_score, brightness = check_image_quality(plate_img, blur_threshold=50.0)
    
    print(f"  📊 Quality: Blur={blur_score:.1f}, Brightness={brightness:.1f}")
    
    if not is_ok:
        print(f"  ⚠️  Image quality too low")
        return None, 0.0, [], []
    
    # 2. Preprocessing
    print(f"  🔄 Preprocessing with 5 methods...")
    processed_images = preprocess_plate(plate_img)
    
    # 3. OCR with voting
    all_results = []
    
    print(f"  🔍 Running OCR with voting...")
    for attempt in range(num_attempts):
        for i, processed in enumerate(processed_images):
            raw_text, ocr_conf = ocr_with_paddleocr(processed)
            
            if raw_text:
                print(f"  🔍 Raw text: {raw_text}")
                cleaned_text = post_process_plate(raw_text)
                
                if cleaned_text:
                    is_valid, area, number, suffix = validate_format(cleaned_text)
                    format_score = confidence_score(cleaned_text)
                    
                    # Combine OCR confidence (0-1) with format score (0-100)
                    # Weight: 70% OCR confidence, 30% format validation
                    combined_confidence = (ocr_conf * 70) + (format_score * 0.3)
                    
                    result = {
                        'raw': raw_text,
                        'cleaned': cleaned_text,
                        'ocr_confidence': ocr_conf,
                        'format_valid': is_valid,
                        'confidence_score': combined_confidence,
                        'method': f"v{i}_attempt{attempt+1}"
                    }
                    all_results.append(result)
    
    if not all_results:
        print(f"  ❌ No valid OCR results")
        return None, 0.0, [], processed_images
    
    # 4. Voting
    cleaned_texts = [r['cleaned'] for r in all_results]
    text_counts = Counter(cleaned_texts)
    most_common = text_counts.most_common(3)
    
    print(f"  🗳️  Voting results:")
    for text, count in most_common:
        print(f"      '{text}': {count} votes")
    
    # 5. Select best
    best_text = most_common[0][0]
    best_results = [r for r in all_results if r['cleaned'] == best_text]
    best_result = max(best_results, key=lambda x: x['confidence_score'])
    
    return best_result['cleaned'], best_result['confidence_score'], all_results, processed_images

def visualize_results(image, plates_data, output_path='result.png'):
    """Create visualization with detection and OCR results"""
    fig = plt.figure(figsize=(20, 12))
    
    # Original image with bbox
    plt.subplot(2, 4, 1)
    img_with_boxes = image.copy()
    for i, data in enumerate(plates_data):
        x1, y1, x2, y2 = data['bbox']
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        if data['plate_text']:
            label = f"Plate {i+1}: {data['plate_text']}"
            cv2.putText(img_with_boxes, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title('License Plate Detection', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Process each plate
    plot_idx = 2
    for i, data in enumerate(plates_data):
        if plot_idx > 8:
            break
        
        # Cropped plate
        plt.subplot(2, 4, plot_idx)
        plt.imshow(cv2.cvtColor(data['plate_img'], cv2.COLOR_BGR2RGB))
        title = f"Plate {i+1} - Original Crop"
        if data['plate_text']:
            title = f"Plate {i+1}\n{data['plate_text']}"
        plt.title(title, fontsize=12, fontweight='bold')
        plt.axis('off')
        plot_idx += 1
        
        # Show preprocessed versions (max 3)
        if data['processed_images']:
            for j, processed in enumerate(data['processed_images'][:3]):
                if plot_idx > 8:
                    break
                
                plt.subplot(2, 4, plot_idx)
                plt.imshow(processed, cmap='gray')
                plt.title(f'Preprocess v{j}', fontsize=11)
                plt.axis('off')
                plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved visualization: {output_path}")
    plt.close()

def process_image(image_path, visualize=True):
    """
    Complete pipeline for single image:
    1. Load image
    2. Detect plates with YOLO
    3. OCR each plate with voting
    4. Visualize results
    """
    print(f"\n{'='*60}")
    print(f"📸 Processing: {image_path}")
    print(f"{'='*60}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load image: {image_path}")
        return None
    
    print(f"📐 Image size: {img.shape[1]}x{img.shape[0]}")
    
    # Detect plates
    print(f"\n🎯 Detecting license plates...")
    plates = detect_plates(img)
    print(f"✅ Found {len(plates)} plate(s)")
    
    if len(plates) == 0:
        print("❌ No license plates detected")
        return None
    
    # Process each plate
    plates_data = []
    for i, plate_data in enumerate(plates):
        print(f"\n{'─'*60}")
        print(f"🔖 Plate #{i+1}")
        print(f"  Detection confidence: {plate_data['confidence']:.3f}")
        
        plate_img = plate_data['image']
        print(f"  Plate size: {plate_img.shape[1]}x{plate_img.shape[0]}")
        
        # OCR with voting
        plate_text, confidence, all_results, processed_images = recognize_plate_with_voting(plate_img)
        
        if plate_text:
            print(f"\n  ✅ RESULT: {plate_text}")
            print(f"  📊 Confidence: {confidence:.1f}/100")
            
            # Validate
            is_valid, area, number, suffix = validate_format(plate_text)
            if is_valid:
                print(f"  ✓ Format: VALID")
                print(f"    Area: {area} | Number: {number} | Suffix: {suffix}")
            else:
                print(f"  ⚠ Format: INVALID")
                # Check if missing area code
                if plate_text and plate_text[0].isdigit():
                    print(f"  ⚠️  WARNING: Missing area code (starts with number)")
                    print(f"      Detected: {plate_text}")
                    print(f"      Possible: D{plate_text}, B{plate_text}, etc.")
        else:
            print(f"\n  ❌ Could not recognize plate")
        
        plates_data.append({
            'bbox': plate_data['bbox'],
            'plate_img': plate_img,
            'plate_text': plate_text,
            'confidence': confidence,
            'processed_images': processed_images,
            'all_results': all_results
        })
    
    # Visualize
    if visualize:
        print(f"\n{'='*60}")
        print("📊 Creating visualization...")
        output_path = 'result.png'  # Always overwrite to result.png
        visualize_results(img, plates_data, output_path)
    
    # Return results
    result = {
        'image_path': image_path,
        'plates': [{'text': p['plate_text'], 'confidence': p['confidence']} 
                   for p in plates_data if p['plate_text']]
    }
    
    return result

def main():
    """Main function"""
    print("\n" + "="*60)
    print("🚗 INDONESIAN LICENSE PLATE RECOGNITION SYSTEM")
    print("🔬 Using: YOLOv10 + PaddleOCR + Voting Mechanism")
    print("="*60)
    
    # Get image paths from command line or use default
    if len(sys.argv) > 1:
        test_images = sys.argv[1:]
    else:
        test_images = ["car.jpg"]  # Default
    
    print(f"\n📋 Processing {len(test_images)} image(s)...")
    
    # Process each image
    all_results = []
    for img_path in test_images:
        try:
            result = process_image(img_path, visualize=True)
            if result:
                all_results.append(result)
        except FileNotFoundError:
            print(f"\n⚠️  Image not found: {img_path}")
        except Exception as e:
            print(f"\n❌ Error processing {img_path}: {e}")
    
    # Summary
    if len(all_results) > 0:
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        
        total_plates = sum(len(r['plates']) for r in all_results)
        print(f"Images processed: {len(all_results)}")
        print(f"Total plates recognized: {total_plates}")
        
        print("\n🎯 Results:")
        for result in all_results:
            img_name = result['image_path']
            for plate in result['plates']:
                print(f"  {img_name}: {plate['text']} ({plate['confidence']:.0f}%)")
    
    print("\n" + "="*60)
    print("✅ Processing complete!")
    print("="*60)

if __name__ == "__main__":
    main()