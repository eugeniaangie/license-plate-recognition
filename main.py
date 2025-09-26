import cv2
import numpy as np
import easyocr
import re
import matplotlib.pyplot as plt
from ultralytics import YOLO

class LicensePlateOCR:
    def __init__(self, model_path="./plate_detection_bestv1.pt"):
        """Initialize the license plate detection and OCR system"""
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        # Character correction mappings (based on common OCR errors)
        self.dict_char_to_int = {
            'O': '0', 'o': '0',
            'I': '1', 'l': '1', '|': '1',
            'Z': '2', 'z': '2',
            'J': '3', 'j': '3',
            'A': '4', 'a': '4',
            'S': '5', 's': '5',
            'G': '6', 'g': '6',
            'T': '7', 't': '7',
            'B': '8', 'b': '8',
            'g': '9', 'q': '9'
        }
        
        self.dict_int_to_char = {
            '0': 'O',
            '1': 'I',
            '2': 'Z',
            '3': 'J',
            '4': 'A',
            '5': 'S',
            '6': 'G',
            '7': 'T',
            '8': 'B'
        }

    def detect_license_plates(self, image):
        """Detect license plates in the image"""
        results = self.model(image)[0]
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence
            })
        
        return detections, results

    def preprocess_license_plate(self, crop):
        """Advanced preprocessing for better OCR results"""
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()
        
        # Resize to improve OCR accuracy
        height, width = gray.shape
        if height < 50:
            scale_factor = 50 / height
            new_width = int(width * scale_factor)
            gray = cv2.resize(gray, (new_width, 50), interpolation=cv2.INTER_CUBIC)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        # Multiple thresholding approaches
        preprocessed_images = []
        
        # 1. Adaptive threshold
        adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        preprocessed_images.append(("Adaptive Threshold", adaptive))
        
        # 2. Otsu's threshold
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("Otsu Threshold", otsu))
        
        # 3. Binary threshold with different values
        for thresh_val in [64, 128, 180]:
            _, binary = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY)
            preprocessed_images.append((f"Binary {thresh_val}", binary))
            
            _, binary_inv = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY_INV)
            preprocessed_images.append((f"Binary INV {thresh_val}", binary_inv))
        
        # 4. Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        preprocessed_images.append(("Morphological", morph))
        
        return preprocessed_images

    def license_complies_format(self, text, format_type="indonesia"):
        """Check if license plate text complies with Indonesian format"""
        if format_type == "indonesia":
            # Indonesian format examples:
            # - D 1078 ALF (1 huruf + 4 angka + 3 huruf) 
            # - B 1234 CD (1 huruf + 4 angka + 2 huruf)
            # - AA 123 B (2 huruf + 3 angka + 1 huruf)
            # - Jakarta: B 1234 ABC, D 5678 EFG
            # - Regional: AA 1234 BB, etc.
            
            text_clean = text.replace(' ', '').upper()
            if len(text_clean) < 4 or len(text_clean) > 10:
                return False
            
            # Indonesian license plate pattern
            # Pattern: 1-3 huruf + 1-4 angka + 1-3 huruf
            pattern = r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{1,3}$'
            
            # Additional validation untuk kode wilayah Indonesia
            valid_prefixes = [
                'A', 'B', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Z',
                'AA', 'AB', 'AD', 'AE', 'AG', 'BA', 'BB', 'BD', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM', 'BN',
                'BP', 'BR', 'BS', 'BT', 'CC', 'CD', 'CE', 'CG', 'DA', 'DB', 'DD', 'DE', 'DG', 'DH', 'DK',
                'DL', 'DM', 'DN', 'DP', 'DR', 'DS', 'DT', 'EA', 'EB', 'ED', 'EE', 'EF', 'EG', 'EH', 'EK'
            ]
            
            if re.match(pattern, text_clean):
                # Check if starts with valid Indonesian region code
                for prefix in valid_prefixes:
                    if text_clean.startswith(prefix):
                        return True
                # If no specific prefix match, still accept if follows general pattern
                return True
            
            return False
        
        return False

    def format_license(self, text):
        """Format and correct OCR errors in Indonesian license plate text"""
        text = text.upper().replace(' ', '').replace('"', '').replace("'", '')
        
        # Remove common OCR artifacts
        artifacts = ['(', ')', '[', ']', '{', '}', '-', '_', '/', '\\', '|', '.', ',']
        for artifact in artifacts:
            text = text.replace(artifact, '')
        
        # Apply character corrections based on Indonesian plate structure
        corrected = ''
        for i, char in enumerate(text):
            # Indonesian plates: huruf di awal dan akhir, angka di tengah
            
            # Deteksi posisi berdasarkan pola Indonesia
            is_in_region_code = i < 3 and (i == 0 or text[i-1].isalpha())
            is_in_number_part = False
            is_in_suffix_code = False
            
            # Cari dimana angka mulai dan berakhir
            number_start = -1
            number_end = -1
            for j, c in enumerate(text):
                if c.isdigit():
                    if number_start == -1:
                        number_start = j
                    number_end = j + 1
            
            if number_start != -1 and number_end != -1:
                is_in_number_part = number_start <= i < number_end
                is_in_suffix_code = i >= number_end
            
            # Apply corrections based on position
            if is_in_region_code or is_in_suffix_code:
                # Posisi huruf: angka -> huruf
                if char in self.dict_int_to_char:
                    corrected += self.dict_int_to_char[char]
                elif char in self.dict_char_to_int.keys() and char.isalpha():
                    corrected += char  # Keep original letter
                else:
                    corrected += char
            elif is_in_number_part:
                # Posisi angka: huruf -> angka
                if char in self.dict_char_to_int:
                    corrected += self.dict_char_to_int[char]
                else:
                    corrected += char
            else:
                corrected += char
        
        # Format dengan spasi sesuai standar Indonesia: X XXXX XXX atau XX XXXX XX
        if len(corrected) >= 4:
            # Cari akhir kode wilayah (huruf awal)
            region_code_end = 0
            for i, char in enumerate(corrected):
                if char.isalpha():
                    region_code_end = i + 1
                else:
                    break
            
            # Cari akhir nomor (angka)
            number_end = region_code_end
            for i in range(region_code_end, len(corrected)):
                if corrected[i].isdigit():
                    number_end = i + 1
                else:
                    break
            
            if region_code_end > 0 and number_end > region_code_end:
                # Format: [KODE WILAYAH] [NOMOR] [KODE AKHIRAN]
                formatted = corrected[:region_code_end] + ' ' + corrected[region_code_end:number_end]
                if number_end < len(corrected):
                    formatted += ' ' + corrected[number_end:]
                return formatted
        
        return corrected

    def read_license_plate(self, crop_images):
        """Read license plate text from preprocessed images"""
        best_result = None
        best_score = 0
        all_results = []
        
        for method_name, processed_img in crop_images:
            try:
                detections = self.reader.readtext(processed_img)
                
                for detection in detections:
                    bbox, text, confidence = detection
                    
                    # Clean and format text
                    formatted_text = self.format_license(text)
                    
                    result = {
                        'method': method_name,
                        'raw_text': text,
                        'formatted_text': formatted_text,
                        'confidence': confidence,
                        'complies_format': self.license_complies_format(formatted_text)
                    }
                    
                    all_results.append(result)
                    
                    # Prioritize results that comply with format and have good confidence
                    score = confidence
                    if result['complies_format']:
                        score *= 1.5  # Boost score for valid format
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        
            except Exception as e:
                print(f"Error processing {method_name}: {e}")
        
        return best_result, all_results

    def visualize_results(self, image, detections, results_data):
        """Visualize detection and OCR results"""
        # Create figure for visualization
        plt.figure(figsize=(20, 12))
        
        # Show original image with detections
        plt.subplot(2, 4, 1)
        img_with_boxes = image.copy()
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f"Plate {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title('License Plate Detection')
        plt.axis('off')
        
        # Show cropped plates and preprocessing results
        plot_idx = 2
        for i, (detection, result_data) in enumerate(zip(detections, results_data)):
            if plot_idx > 8:
                break
                
            x1, y1, x2, y2 = detection['bbox']
            crop = image[y1:y2, x1:x2]
            
            # Original crop
            plt.subplot(2, 4, plot_idx)
            plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            plt.title(f'Plate {i+1} Original')
            plt.axis('off')
            plot_idx += 1
            
            if plot_idx <= 8 and result_data['best_result']:
                # Best preprocessing result
                best_method = result_data['best_result']['method']
                preprocessed_images = result_data['preprocessed_images']
                
                for method_name, processed_img in preprocessed_images:
                    if method_name == best_method:
                        plt.subplot(2, 4, plot_idx)
                        plt.imshow(processed_img, cmap='gray')
                        plt.title(f'Best: {best_method}')
                        plt.axis('off')
                        plot_idx += 1
                        break
        
        plt.tight_layout()
        plt.show()

    def process_image(self, image_path, visualize=True):
        """Complete pipeline for license plate detection and OCR"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        print(f"Processing image: {image_path}")
        print(f"Image shape: {image.shape}")
        
        # Detect license plates
        detections, yolo_results = self.detect_license_plates(image)
        print(f"Found {len(detections)} license plate(s)")
        
        results_data = []
        
        for i, detection in enumerate(detections):
            print(f"\n--- Processing Plate {i+1} ---")
            x1, y1, x2, y2 = detection['bbox']
            crop = image[y1:y2, x1:x2]
            
            # Preprocess the crop
            preprocessed_images = self.preprocess_license_plate(crop)
            
            # Perform OCR
            best_result, all_results = self.read_license_plate(preprocessed_images)
            
            result_data = {
                'detection': detection,
                'crop': crop,
                'preprocessed_images': preprocessed_images,
                'best_result': best_result,
                'all_results': all_results
            }
            results_data.append(result_data)
            
            # Print results
            if best_result:
                print(f"Best Result:")
                print(f"  Method: {best_result['method']}")
                print(f"  Raw Text: '{best_result['raw_text']}'")
                print(f"  Formatted: '{best_result['formatted_text']}'")
                print(f"  Confidence: {best_result['confidence']:.3f}")
                print(f"  Valid Format: {best_result['complies_format']}")
            else:
                print("No valid text detected")
            
            print(f"\nAll OCR attempts for Plate {i+1}:")
            for result in all_results:
                print(f"  {result['method']}: '{result['formatted_text']}' (conf: {result['confidence']:.3f}, valid: {result['complies_format']})")
        
        # Visualize results
        if visualize:
            self.visualize_results(image, detections, results_data)
        
        return results_data

# Usage example
if __name__ == "__main__":
    # Initialize the system
    ocr_system = LicensePlateOCR("./plate_detection_bestv1.pt")
    
    # Process an image
    try:
        results = ocr_system.process_image("car.jpg", visualize=True)
        
        # Print final summary
        print("\n=== FINAL RESULTS ===")
        for i, result in enumerate(results):
            print(f"License Plate {i+1}:")
            if result['best_result']:
                print(f"  Text: {result['best_result']['formatted_text']}")
                print(f"  Confidence: {result['best_result']['confidence']:.3f}")
                print(f"  Valid: {result['best_result']['complies_format']}")
            else:
                print("  No valid text detected")
            print()
            
    except Exception as e:
        print(f"Error: {e}")