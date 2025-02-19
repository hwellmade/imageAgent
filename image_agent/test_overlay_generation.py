import os
from pathlib import Path
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import logging
import json
from typing import Dict, List, Any
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_test_prompt(target_lang: str = "en") -> str:
    """Create a test prompt for vision analysis with coordinate requirements."""
    return f"""**Analyze this image and provide structured OCR output with precise geometric localization**

    ### **Core Requirements:**
    1. **Text Detection & Layout Preservation**  
    - Detect text blocks at **paragraph → line → word hierarchy** (visually accurate grouping)  
    - Preserve original reading order (top-to-bottom for vertical text, left-to-right for horizontal)

    2. **Coordinate System Rules**  
    - **Precision:** Coordinates must wrap **exact visible text boundaries** (no padding/guesswork)  
    - **Percentage Calculation:**  
        ```{{python}}
        x% = (pixel_x / image_width) * 100  
        y% = (pixel_y / image_height) * 100  
        ```
    - **Example:** If text spans from (120px, 80px) to (480px, 160px) in a 1600x1200 image:  
        `[7.5%, 6.67%, 30.0%, 13.33%]`  
    - **Multi-line Threshold:** Lines belong to the same paragraph if vertical gap ≤ 1.5× font height

    3. **Visual Context Handling**  
    - **Orientation Detection:**  
        - **Vertical Text:** Characters flow top→bottom, lines progress right→left  
        - **Horizontal Text:** Standard left→right, top→bottom  

    ### **Output Format (JSON):**
    {{
        "original_language": "string",
        "target_language": "{target_lang}",
        "metadata": {{
            "image_orientation": "vertical|horizontal",
            "dimensions": {{"width_px": number, "height_px": number}},
            "total_paragraphs": number,
            "total_lines": number
        }},
        "paragraphs": [
            {{
                "id": number,
                "coordinates": [x1%, y1%, x2%, y2%],  // Where:
                // - (0,0) is the top-left corner of the image
                // - (x1,y1) is the top-left corner of the box
                // - (x2,y2) is the bottom-right corner of the box
                // - All values are percentages of image width/height
                "orientation": "vertical|horizontal",
                "lines": [
                    {{
                        "coordinates": [x1%, y1%, x2%, y2%],  // Same coordinate system as above
                        "original_text": "string",
                        "translated_text": "string",
                        "text_orientation": "vertical|horizontal"
                    }}
                ]
            }}
        ]
    }}

    Important:
    - Coordinates are relative percentages of image dimensions
    - (0,0) is the top-left corner of the image
    - For each box, provide:
      * (x1,y1): top-left corner coordinates
      * (x2,y2): bottom-right corner coordinates
    - All coordinates should be between 0 and 100
    """

def draw_text_overlay(
    image_path: str,
    analysis_result: Dict[str, Any],
    output_path: str,
    use_translated_text: bool = False
) -> None:
    """Draw text overlay on the image with debug visualization."""
    try:
        # Open the image without any rotation
        with Image.open(image_path) as image:
            # Print original image dimensions
            image_width, image_height = image.size
            print(f"Original image dimensions: {image_width}x{image_height}")
            
            # Create overlay
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            def percentage_to_pixels(coords: List[float], is_vertical: bool = True) -> List[int]:
                """Convert percentage coordinates to pixel coordinates.
                
                Args:
                    coords: List of [x1, y1, x2, y2] coordinates in percentages
                    is_vertical: Whether the image is in vertical orientation
                
                Returns:
                    List of [x1, y1, x2, y2] coordinates in pixels
                """
                x1, y1, x2, y2 = coords
                
                # Print debug info
                print(f"\nCoordinate Conversion Debug:")
                print(f"  Input percentages: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")
                print(f"  Image dimensions: {image_width}x{image_height}")
                print(f"  Image orientation: {'vertical' if is_vertical else 'horizontal'}")
                
                # Only flip coordinates for vertical images
                if is_vertical:
                    print("  Applying vertical image adjustments...")
                    # Flip y coordinates for vertical images
                    y1 = 100 - y1
                    y2 = 100 - y2
                    print(f"  After vertical flip: y1={y1:.2f}, y2={y2:.2f}")
                # calculate the di==dimension ratio between LLM v.s. actual image
                x_scaling = analysis_result['metadata']['dimensions']['width_px'] / image_width
                y_scaling = analysis_result['metadata']['dimensions']['height_px'] / image_height
                # Calculate pixel coordinates
                pixel_x1 = int(x1 * image_width / 100 / x_scaling)
                pixel_y1 = int(y1 * image_height / 100 / y_scaling)
                pixel_x2 = int(x2 * image_width / 100 / x_scaling)
                pixel_y2 = int(y2 * image_height / 100 / y_scaling)
                
                pixel_coords = [pixel_x1, pixel_y1, pixel_x2, pixel_y2]
                
                # Print conversion results
                print(f"  Output pixels: ({pixel_x1}, {pixel_y1}) to ({pixel_x2}, {pixel_y2})")
                print(f"  Box dimensions: {pixel_x2-pixel_x1}x{pixel_y2-pixel_y1} pixels")
                
                return pixel_coords
            
            # First draw white boxes for lines
            for paragraph in analysis_result['paragraphs']:
                for j, line in enumerate(paragraph['lines']):
                    # Convert line coordinates to pixels
                    is_vertical = analysis_result['metadata']['image_orientation'] == 'vertical'
                    line_coords = percentage_to_pixels(line['coordinates'], is_vertical)
                    text = line['translated_text'] if use_translated_text else line['original_text']
                    
                    # Draw white box for line
                    draw.rectangle(line_coords, outline=(255, 255, 255, 255), width=2)
                    
                    # Log coordinates for debugging
                    print(f"Line {j} - Original coords (percentages): {line['coordinates']}")
                    print(f"Line {j} - Pixel coords: {line_coords}")
                    print(f"Line {j} - Text: {text}")
            
            # Then draw red boxes for paragraphs (so they're on top)
            for i, paragraph in enumerate(analysis_result['paragraphs']):
                # Convert paragraph coordinates to pixels
                is_vertical = analysis_result['metadata']['image_orientation'] == 'vertical'
                para_coords = percentage_to_pixels(paragraph['coordinates'], is_vertical)
                
                # Draw red box for paragraph
                draw.rectangle(para_coords, outline=(255, 0, 0, 255), width=4)
                print(f"Paragraph {i} - Original coords (percentages): {paragraph['coordinates']}")
                print(f"Paragraph {i} - Pixel coords: {para_coords}")
            
            # Save debug visualization
            debug_path = str(Path(output_path).parent / f"debug_{Path(output_path).name}")
            result_debug = Image.alpha_composite(image.convert('RGBA'), overlay)
            result_debug.convert('RGB').save(debug_path, quality=95)
            print(f"Saved debug visualization to: {debug_path}")
            
            # Now draw the actual text overlay
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            for paragraph in analysis_result['paragraphs']:
                for line in paragraph['lines']:
                    line_coords = percentage_to_pixels(line['coordinates'], analysis_result['metadata']['image_orientation'] == 'vertical')
                    text = line['translated_text'] if use_translated_text else line['original_text']
                    
                    x1, y1, x2, y2 = line_coords
                    line_width = x2 - x1
                    line_height = y2 - y1
                    
                    # Draw semi-transparent background for text
                    draw.rectangle(line_coords, fill=(0, 0, 0, 128))
                    
                    # Calculate font size
                    font_size = int(min(line_height * 0.8, line_width / (len(text) * 0.7)))
                    font_size = max(12, min(32, font_size))
                    
                    try:
                        font = ImageFont.truetype("msgothic.ttc", font_size)
                    except:
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                        except:
                            font = ImageFont.load_default()
                    
                    # Draw text centered in its box
                    draw.text(
                        (x1 + line_width/2, y1 + line_height/2),
                        text,
                        font=font,
                        fill=(255, 255, 255, 255),
                        anchor="mm"
                    )
            
            # Composite and save final result
            result = Image.alpha_composite(image.convert('RGBA'), overlay)
            result.convert('RGB').save(output_path, quality=95)
            
            print(f"Saved {'translated' if use_translated_text else 'original'} overlay to: {output_path}")
            
    except Exception as e:
        print(f"Error in drawing overlay: {str(e)}")
        raise

def test_overlay_generation(image_path: str, target_lang: str = "en"):
    """Test LLM response and overlay generation."""
    try:
        # Ensure input image exists
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        
        # Create output directory if it doesn't exist
        output_dir = image_path.parent / "output"
        output_dir.mkdir(exist_ok=True)
        print(f"Using output directory: {output_dir}")

        # Get API key from environment
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        # Configure the API
        genai.configure(api_key=api_key)
        print("Configured Gemini API")

        # Initialize the model
        model = genai.GenerativeModel('gemini-2.0-flash-lite-preview-02-05')
        print("Initialized Gemini model")

        # Create test prompt
        prompt = create_test_prompt(target_lang)
        print(f"Created prompt for target language: {target_lang}")

        # Load and process image
        print(f"Loading image from: {image_path}")
        image = Image.open(image_path)
        
        # Check EXIF orientation
        try:
            exif = image._getexif()
            orientation = exif.get(274) if exif else None  # 274 is the orientation tag
            print(f"EXIF Orientation tag: {orientation}")
            # Orientation values:
            # 1: Normal (no rotation)
            # 3: 180 degree rotation
            # 6: 90 degree rotation (vertical, needs to be rotated clockwise)
            # 8: 270 degree rotation (vertical, needs to be rotated counter-clockwise)
        except:
            print("No EXIF orientation found")
        
        # Log original image dimensions
        original_width, original_height = image.size
        print(f"Original image dimensions (raw): {original_width}x{original_height}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("Converted image to RGB mode")

        # Generate content
        print("Sending request to Gemini API...")
        response = model.generate_content([prompt, image])
        response.resolve()

        # Parse response
        try:
            # Clean up the response text by removing markdown code block markers
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            result = json.loads(response_text.strip())
            print("Successfully parsed LLM response")
            print(f"LLM reports image orientation as: {result['metadata']['image_orientation']}")
            
            # Save raw response for inspection
            response_path = output_dir / f"{image_path.stem}_llm_response.json"
            with open(response_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Saved raw LLM response to: {response_path}")
            
            # Generate overlay images
            original_overlay_path = output_dir / f"{image_path.stem}_original_overlay.jpg"
            translated_overlay_path = output_dir / f"{image_path.stem}_translated_overlay.jpg"
            
            print(f"Generating original overlay at: {original_overlay_path}")
            # Generate original text overlay
            draw_text_overlay(str(image_path), result, str(original_overlay_path), use_translated_text=False)
            
            print(f"Generating translated overlay at: {translated_overlay_path}")
            # Generate translated text overlay
            draw_text_overlay(str(image_path), result, str(translated_overlay_path), use_translated_text=True)
            
            print("Test completed successfully")
            print(f"Original overlay saved to: {original_overlay_path}")
            print(f"Translated overlay saved to: {translated_overlay_path}")
            return result
            
        except json.JSONDecodeError:
            print("Failed to parse LLM response as JSON")
            print("Raw response:", response.text)
            raise

    except Exception as e:
        print(f"Error in test: {str(e)}")
        raise

if __name__ == "__main__":
    # Use the specified test image
    image_path = "frontend/test_images/test2.jpg"
    if not Path(image_path).exists():
        print(f"Error: Test image not found at {image_path}")
        exit(1)
    
    test_overlay_generation(image_path) 