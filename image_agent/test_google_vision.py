from google.cloud import vision
import sys
from pathlib import Path

def test_google_vision(image_path: str):
    """Test Google Cloud Vision API on a single image."""
    print(f"Testing Google Cloud Vision API on image: {image_path}")
    
    # Create a client
    client = vision.ImageAnnotatorClient()
    
    # Read the image file
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    # Perform text detection
    print("\nPerforming text detection...")
    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    # Print full text
    if texts:
        print("\nFull text:")
        print(texts[0].description)
        print("\nDetailed text blocks:")
        for text in texts[1:]:  # Skip the first one as it's the full text
            print(f"Text: {text.description}")
            vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
            print(f"Bounding Box: {vertices}")
            print("-" * 50)
    else:
        print("No text detected")
    
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_google_vision.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_google_vision(image_path) 