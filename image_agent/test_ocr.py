import easyocr
from PIL import Image
import numpy as np
import sys
from pathlib import Path

def test_ocr(image_path: str):
    """Test OCR on a single image and print detailed results."""
    print(f"Testing OCR on image: {image_path}")
    
    # Initialize reader with debug info
    print("Initializing EasyOCR reader...")
    reader = easyocr.Reader(['ja', 'en'], verbose=True)
    
    # Load and process image
    print("Loading image...")
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image)
    
    # Perform OCR with different configurations
    print("\nTesting with default settings:")
    results = reader.readtext(img_array)
    print("\nDetected text (default):")
    for bbox, text, conf in results:
        print(f"Text: {text}")
        print(f"Confidence: {conf:.2%}")
        print(f"Bounding Box: {bbox}")
        print("-" * 50)
    
    # Try with different parameters
    print("\nTesting with paragraph=True:")
    results_paragraph = reader.readtext(img_array, paragraph=True)
    print("\nDetected text (paragraph mode):")
    for bbox, text, conf in results_paragraph:
        print(f"Text: {text}")
        print(f"Confidence: {conf:.2%}")
        print("-" * 50)
    
    # Try with different contrast/preprocessing
    print("\nTesting with contrast adjustment:")
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(image)
    enhanced_img = enhancer.enhance(1.5)  # Increase contrast
    results_enhanced = reader.readtext(np.array(enhanced_img))
    print("\nDetected text (enhanced contrast):")
    for bbox, text, conf in results_enhanced:
        print(f"Text: {text}")
        print(f"Confidence: {conf:.2%}")
        print("-" * 50)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_ocr.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_ocr(image_path) 