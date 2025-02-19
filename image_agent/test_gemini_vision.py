import os
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_prompt(target_lang: str = "en") -> str:
    """Create a test prompt for vision analysis."""
    return f"""Analyze this image and provide the following information in JSON format:
    1. Describe what text you see in the image
    2. Translate the text to {target_lang}
    3. Provide the locations of the text

    Format the response as valid JSON with this structure:
    {{
        "original_text": "text found in image",
        "translated_text": "translation in {target_lang}",
        "text_locations": [
            {{
                "text": "specific text segment",
                "position": "description of where this text appears"
            }}
        ]
    }}
    """

def test_vision_api(image_path: str, target_lang: str = "en"):
    """Test the Gemini Vision API with a single image."""
    try:
        # Get API key from environment
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        # Configure the API
        genai.configure(api_key=api_key)
        logger.info("Configured Gemini API")

        # Initialize the model
        model = genai.GenerativeModel('gemini-2.0-flash-lite-preview-02-05')
        logger.info("Initialized Gemini Pro Vision model")

        # Create test prompt
        prompt = create_test_prompt(target_lang)
        logger.info(f"Created prompt for target language: {target_lang}")

        # Load and process image
        logger.info(f"Loading image from: {image_path}")
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info("Converted image to RGB mode")

        # Generate content
        logger.info("Sending request to Gemini API...")
        response = model.generate_content([prompt, image])
        response.resolve()  # Wait for completion

        # Print response
        logger.info("\n=== API Response ===")
        print(response.text)
        logger.info("=== End Response ===")

        return response.text

    except Exception as e:
        logger.error(f"Error in vision API test: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python test_gemini_vision.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    test_vision_api(image_path) 