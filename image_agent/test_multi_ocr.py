import easyocr
import pytesseract
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import sys
from pathlib import Path

def preprocess_image_v1(image):
    """Basic preprocessing - invert and enhance."""
    # Convert to grayscale
    gray = ImageOps.grayscale(image)
    # Invert colors (white text on dark background)
    inverted = ImageOps.invert(gray)
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(inverted)
    enhanced = enhancer.enhance(2.0)
    return enhanced

def preprocess_image_v2(image):
    """Advanced preprocessing with multiple steps."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Extract green channel (since text is on green background)
    r, g, b = image.split()
    
    # Use green channel and invert
    green_channel = ImageOps.invert(g)
    
    # Apply threshold to make text more distinct
    threshold = 200
    green_channel = green_channel.point(lambda x: 255 if x > threshold else 0)
    
    # Apply slight blur to reduce noise
    smoothed = green_channel.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(smoothed)
    enhanced = enhancer.enhance(2.0)
    
    return enhanced

def test_multi_ocr(image_path: str):
    """Test multiple OCR approaches with different preprocessing."""
    print(f"Testing multiple OCR approaches on: {image_path}")
    
    # Load image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Directory for saving processed images
    output_dir = Path(image_path).parent
    
    # Test EasyOCR with different preprocessing
    print("\n=== EasyOCR Tests ===")
    reader = easyocr.Reader(['ja', 'en'], verbose=True)
    
    # Original image
    print("\nTesting original image:")
    results = reader.readtext(np.array(image))
    for bbox, text, conf in results:
        print(f"Text: {text}")
        print(f"Confidence: {conf:.2%}")
        print("-" * 50)
    
    # Preprocessing v1
    processed_v1 = preprocess_image_v1(image)
    processed_v1.save(output_dir / "preprocessed_v1.jpg")
    print("\nTesting with preprocessing v1:")
    results_v1 = reader.readtext(np.array(processed_v1))
    for bbox, text, conf in results_v1:
        print(f"Text: {text}")
        print(f"Confidence: {conf:.2%}")
        print("-" * 50)
    
    # Preprocessing v2
    processed_v2 = preprocess_image_v2(image)
    processed_v2.save(output_dir / "preprocessed_v2.jpg")
    print("\nTesting with preprocessing v2:")
    results_v2 = reader.readtext(np.array(processed_v2))
    for bbox, text, conf in results_v2:
        print(f"Text: {text}")
        print(f"Confidence: {conf:.2%}")
        print("-" * 50)
    
    # Test Tesseract (if available)
    try:
        print("\n=== Tesseract Tests ===")
        # Configure Tesseract for Japanese
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Original image
        print("\nTesting original image:")
        text = pytesseract.image_to_string(image, lang='jpn')
        print(text)
        print("-" * 50)
        
        # Preprocessing v1
        print("\nTesting with preprocessing v1:")
        text_v1 = pytesseract.image_to_string(processed_v1, lang='jpn')
        print(text_v1)
        print("-" * 50)
        
        # Preprocessing v2
        print("\nTesting with preprocessing v2:")
        text_v2 = pytesseract.image_to_string(processed_v2, lang='jpn')
        print(text_v2)
        print("-" * 50)
        
    except Exception as e:
        print(f"Tesseract test failed: {str(e)}")
    
    print("\nPreprocessed images saved as 'preprocessed_v1.jpg' and 'preprocessed_v2.jpg'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_multi_ocr.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_multi_ocr(image_path) 