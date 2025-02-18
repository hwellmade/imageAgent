import easyocr
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import sys
from pathlib import Path

def preprocess_image(image):
    """Apply various preprocessing techniques to improve OCR."""
    # Convert to grayscale
    gray = ImageOps.grayscale(image)
    
    # Invert colors (since we have white text on dark background)
    inverted = ImageOps.invert(gray)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(inverted)
    enhanced = enhancer.enhance(2.0)
    
    # Enhance sharpness
    sharpener = ImageEnhance.Sharpness(enhanced)
    sharpened = sharpener.enhance(2.0)
    
    return sharpened

def test_ocr_enhanced(image_path: str):
    """Test OCR with enhanced preprocessing."""
    print(f"Testing enhanced OCR on image: {image_path}")
    
    # Initialize reader
    print("Initializing EasyOCR reader...")
    reader = easyocr.Reader(['ja', 'en'], verbose=True)
    
    # Load original image
    print("Loading image...")
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Test original image
    print("\nTesting with original image:")
    results = reader.readtext(np.array(image))
    print("\nDetected text (original):")
    for bbox, text, conf in results:
        print(f"Text: {text}")
        print(f"Confidence: {conf:.2%}")
        print("-" * 50)
    
    # Test with preprocessing
    print("\nTesting with preprocessing:")
    processed_img = preprocess_image(image)
    results_processed = reader.readtext(np.array(processed_img))
    print("\nDetected text (preprocessed):")
    for bbox, text, conf in results_processed:
        print(f"Text: {text}")
        print(f"Confidence: {conf:.2%}")
        print("-" * 50)
    
    # Save preprocessed image for inspection
    output_path = Path(image_path).parent / "preprocessed.jpg"
    processed_img.save(output_path)
    print(f"\nPreprocessed image saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_ocr_enhanced.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_ocr_enhanced(image_path) 