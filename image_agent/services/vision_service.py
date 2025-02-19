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

1. Text Structure:
- Extract all visible text while preserving reading order
- Group text into logical paragraphs
- Preserve line breaks within paragraphs
- Indicate text orientation (vertical/horizontal)

2. Translation:
- Translate all text to {target_lang_name}
- Keep product names, numbers, and measurements in their original form
- Maintain formatting symbols (●, ・, ※, etc.)
- Use {target_lang_code} for proper character encoding

3. Required Output Format (STRICT JSON):
- Response must be valid JSON with proper comma placement
- No trailing commas
- All property names and string values must be in double quotes
- Commas must be between elements, not after them
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
      "orientation": "vertical|horizontal",
      "lines": [
        {{
          "original_text": "string",
          "translated_text": "string",
          "text_orientation": "vertical|horizontal"
        }}
      ]
    }}
  ]
}}

Important: 
- Preserve all text content exactly as shown
- Maintain all formatting and special characters
- Translate to {target_lang_name} ({target_lang_code})
- Focus on accurate text extraction and translation
- Ensure JSON output follows strict formatting rules with proper comma placement
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
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean and normalize JSON response from LLM output."""
        try:
            # Remove markdown code block markers
            response_text = response_text.strip()
            if response_text.startswith("```"):
                # Remove opening ```json or ``` line
                first_newline = response_text.find('\n')
                if first_newline != -1:
                    response_text = response_text[first_newline + 1:]
                # Remove closing ```
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
            
            # Strip any remaining whitespace
            response_text = response_text.strip()
            
            def normalize_json_structure(text):
                """Normalize JSON by parsing and reconstructing problematic parts."""
                # First pass: fix basic syntax issues
                text = text.replace('",\n          ,', '",')
                text = text.replace('",\n        ,', '",')
                text = text.replace('",\n      ,', '",')
                
                try:
                    # Try to parse as is first
                    return json.loads(text)
                except json.JSONDecodeError as e:
                    # If that fails, try more aggressive normalization
                    lines = text.split('\n')
                    normalized_lines = []
                    in_object = False
                    previous_line = ""
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Handle property names and values
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip().strip('"')
                            value = value.strip()
                            
                            # Ensure key is properly quoted
                            key = f'"{key}"'
                            
                            # Handle string values
                            if value.startswith('"'):
                                if not value.endswith('"') or value.endswith('",'):
                                    value = value.rstrip(',') + '"'
                            
                            # Reconstruct the line
                            line = f'{key}: {value}'
                            
                            # Add comma if needed
                            if not line.endswith('}') and not line.endswith(']') and not line.endswith(','):
                                line += ','
                                
                        # Handle object/array boundaries
                        if '{' in line:
                            in_object = True
                        if '}' in line:
                            in_object = False
                            # Remove trailing comma before closing brace
                            if previous_line.endswith(','):
                                normalized_lines[-1] = normalized_lines[-1].rstrip(',')
                        
                        normalized_lines.append(line)
                        previous_line = line
                    
                    # Join lines and try to parse again
                    normalized_json = '\n'.join(normalized_lines)
                    try:
                        return json.loads(normalized_json)
                    except json.JSONDecodeError:
                        # If still failing, try one more time with even more aggressive cleaning
                        normalized_json = normalized_json.replace(',,', ',')
                        normalized_json = normalized_json.replace(',}', '}')
                        normalized_json = normalized_json.replace(',]', ']')
                        return json.loads(normalized_json)
            
            # Normalize and validate JSON structure
            normalized_data = normalize_json_structure(response_text)
            
            # Convert back to formatted string
            return json.dumps(normalized_data, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error cleaning JSON response: {str(e)}", exc_info=True)
            raise

    def _validate_json_response(self, result: Dict[str, Any]) -> bool:
        """Validate the JSON response has all required fields."""
        required_fields = [
            'original_language',
            'target_language',
            'metadata',
            'paragraphs'
        ]
        
        metadata_fields = [
            'image_orientation',
            'total_paragraphs',
            'total_lines'
        ]
        
        # Check top-level fields
        for field in required_fields:
            if field not in result:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Check metadata fields
        for field in metadata_fields:
            if field not in result['metadata']:
                logger.error(f"Missing metadata field: {field}")
                return False
        
        # Check paragraphs structure
        for i, para in enumerate(result['paragraphs']):
            if 'id' not in para or 'lines' not in para:
                logger.error(f"Invalid paragraph structure at index {i}")
                return False
            
            # Check lines structure
            for j, line in enumerate(para['lines']):
                if 'original_text' not in line or 'translated_text' not in line:
                    logger.error(f"Invalid line structure at paragraph {i}, line {j}")
                    return False
        
        return True

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
            
            # Clean and parse response
            try:
                # Get raw response text
                response_text = response.text.strip()
                logger.debug(f"Raw response: {response_text}")
                
                # Clean JSON response
                json_text = self._clean_json_response(response_text)
                logger.debug(f"Cleaned JSON text: {json_text}")
                
                try:
                    result = json.loads(json_text)
                    logger.info("Successfully parsed API response")
                    
                    # Validate response structure
                    if not self._validate_json_response(result):
                        raise ValueError("Invalid response structure")
                    
                    return result
                    
                except json.JSONDecodeError as je:
                    logger.error(f"JSON parsing error: {str(je)}")
                    logger.error(f"Position: char {je.pos}")
                    logger.error(f"Line number: {je.lineno}")
                    logger.error(f"Column number: {je.colno}")
                    logger.error(f"Attempted to parse: {json_text}")
                    raise ValueError(f"Invalid JSON format in API response: {str(je)}")
                    
            except Exception as parse_error:
                logger.error(f"Error parsing response: {str(parse_error)}")
                logger.error(f"Raw response was: {response_text}")
                raise ValueError(f"Failed to parse API response: {str(parse_error)}")
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}", exc_info=True)
            raise

# Create singleton instance
vision_service = VisionService() 