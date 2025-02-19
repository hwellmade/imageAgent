import os
import google.generativeai as genai
from typing import Dict, Optional, Any, List
from enum import Enum
import logging
from pathlib import Path
import json
from PIL import Image
import base64
import io

logger = logging.getLogger(__name__)

class SupportedLanguages(str, Enum):
    """Supported languages with their codes and display names."""
    CHINESE_TRADITIONAL = "zh-Hant"
    ENGLISH = "en"
    JAPANESE = "ja"
    KOREAN = "ko"

    @property
    def display_name(self) -> str:
        """Get the display name for the language."""
        display_names = {
            "zh-Hant": "Traditional Chinese",
            "en": "English",
            "ja": "Japanese",
            "ko": "Korean"
        }
        return display_names[self.value]

def create_vision_prompt(target_lang_code: str, target_lang_name: str) -> str:
    """Create a dynamic prompt for vision analysis."""
    return f"""Analyze this image and extract text information with the following requirements:

1. Text Extraction and Layout:
- Extract all visible text while preserving spatial relationships
- For each text block/paragraph and line, provide coordinates as relative percentages:
  * Use format [x1%, y1%, x2%, y2%] where:
    - (0%, 0%) is the top-left corner of the image
    - (100%, 100%) is the bottom-right corner of the image
    - Each coordinate pair (x%, y%) represents a point relative to image dimensions
  * Example: [10%, 20%, 90%, 30%] means:
    - Left edge is 10% from left border
    - Top edge is 20% from top border
    - Right edge is 90% from left border
    - Bottom edge is 30% from top border
- Indicate whether text is vertical or horizontal
- Preserve text content with original line breaks

2. Translation:
- Translate all text to {target_lang_name}
- Keep product names, numbers, and measurements in their original form
- Maintain formatting symbols (●, ・, ※, etc.)
- Use {target_lang_code} for proper character encoding

3. Required Output Format (JSON):
{{
  "original_language": "string",
  "target_language": "{target_lang_code}",
  "metadata": {{
    "image_orientation": "vertical|horizontal",
    "total_paragraphs": number,
    "total_lines": number
  }},
  "paragraphs": [
    {{
      "id": number,
      "coordinates": [x1%, y1%, x2%, y2%],  // Relative percentages from top-left
      "orientation": "vertical|horizontal",
      "lines": [
        {{
          "coordinates": [x1%, y1%, x2%, y2%],  // Relative percentages from top-left
          "original_text": "string",
          "translated_text": "string"
        }}
      ]
    }}
  ]
}}

Important: 
- All coordinates must be expressed as percentages relative to image dimensions
- Use top-left (0%, 0%) as coordinate origin
- Ensure coordinates are precise for accurate overlay placement
- Translate to {target_lang_name} ({target_lang_code})
- Preserve all formatting and special characters
"""

class VisionService:
    """Service for handling image analysis using Gemini Vision API."""
    
    def __init__(self):
        self._initialized = False
        self.supported_languages = {lang.value: lang.display_name for lang in SupportedLanguages}
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            
            # Configure the API
            genai.configure(api_key=api_key)
            # Initialize with the working model
            self.model = genai.GenerativeModel('gemini-2.0-flash-lite-preview-02-05')
            self._initialized = True
            logger.info("Vision service initialized successfully with Gemini Flash-Lite")
        except Exception as e:
            logger.error(f"Vision service initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize vision service: {str(e)}")
    
    def validate_language(self, lang_code: str) -> Optional[str]:
        """Validate and return language name if supported."""
        return self.supported_languages.get(lang_code)
    
    def _prepare_image(self, image_path: str) -> Dict:
        """Prepare image for API request."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Convert to base64
                img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                
                return {
                    "mime_type": "image/jpeg",
                    "data": img_base64
                }
        except Exception as e:
            logger.error(f"Error preparing image: {str(e)}")
            raise
    
    async def analyze_image(
        self,
        image_path: str,
        target_lang_code: str = "en"
    ) -> Dict[str, Any]:
        """Analyze image and extract text with translation."""
        if not self._initialized:
            raise RuntimeError("Vision service is not properly initialized")
        
        try:
            # Validate language
            target_lang_name = self.validate_language(target_lang_code)
            if not target_lang_name:
                raise ValueError(f"Unsupported language code: {target_lang_code}")
            
            # Create prompt
            prompt = create_vision_prompt(target_lang_code, target_lang_name)
            
            # Load and process image
            logger.info(f"Loading image from: {image_path}")
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.info("Converted image to RGB mode")
            
            # Generate content
            logger.info("Sending request to Gemini API...")
            response = self.model.generate_content([prompt, image])
            response.resolve()  # Wait for completion
            
            # Parse response
            try:
                result = json.loads(response.text)
                logger.info("Successfully parsed API response")
                return result
            except json.JSONDecodeError:
                logger.error("Failed to parse API response as JSON")
                raise ValueError("Invalid response format from API")
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            raise

# Create singleton instance
vision_service = VisionService() 