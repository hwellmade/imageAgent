from google.cloud import vision
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io
import math

def calculate_text_angle(points: list) -> float:
    """Calculate the angle of text based on bounding box points."""
    # Get the points
    x1, y1 = points[0]  # bottom-left
    x2, y2 = points[1]  # bottom-right
    x3, y3 = points[3]  # top-left
    
    # Calculate width and height of the bounding box
    width = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    height = math.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    
    # Determine if text is more horizontal or vertical based on aspect ratio
    is_vertical = height > width
    
    if is_vertical:
        # For vertical text, use the left edge (points[0] to points[3])
        angle = math.degrees(math.atan2(y3 - y1, x3 - x1))
    else:
        # For horizontal text, use the bottom edge (points[0] to points[1])
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    
    # Normalize angle to be between -90 and 90 degrees
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180
    
    # For vertical text, we need to add 90 degrees first
    if is_vertical:
        angle += 90
        # Normalize again after adding 90
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180
    
    # Now flip the angle 180 degrees if it's in the wrong direction
    if (is_vertical and angle < 0) or (not is_vertical and angle > 0):
        angle += 180
    else:
        angle -= 180
    
    # Final normalization
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180
        
    return angle

def calculate_font_size(bbox_width: float, bbox_height: float, text_length: int, is_vertical: bool) -> int:
    """
    Calculate appropriate font size based on bounding box dimensions and text length.
    
    Args:
        bbox_width: Width of the bounding box
        bbox_height: Height of the bounding box
        text_length: Length of the text to display
        is_vertical: Whether the text is vertical
        
    Returns:
        Calculated font size
    """
    # Use the smaller dimension for size calculation
    if is_vertical:
        available_space = bbox_width
        text_space_needed = bbox_height / max(1, text_length)
    else:
        available_space = bbox_height
        text_space_needed = bbox_width / max(1, text_length)
    
    # Base size on available space
    base_size = min(available_space * 0.9, text_space_needed * 1.2)
    
    # Clamp the font size between reasonable limits
    return int(max(12, min(40, base_size)))

def draw_text_overlay(image_path: str, texts: list, output_path: str | None = None) -> None:
    """Draw detected text and bounding boxes on the image."""
    # Open the image
    image = Image.open(image_path)
    
    # Create a drawing context with RGBA mode for transparency
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Try to load a font that supports Japanese text
    try:
        # For Windows - we'll set the size later
        base_font_name = "msgothic.ttc"
        test_font = ImageFont.truetype(base_font_name, 12)  # Just for testing if font exists
    except:
        try:
            # For Unix-like systems
            base_font_name = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            test_font = ImageFont.truetype(base_font_name, 12)
        except:
            base_font_name = None
    
    # Skip the first text annotation as it contains the entire text
    for text in texts[1:]:
        # Get vertices of the bounding polygon
        vertices = text.bounding_poly.vertices
        points = [(vertex.x, vertex.y) for vertex in vertices]
        
        # Calculate width and height of the bounding box
        bbox_width = math.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
        bbox_height = math.sqrt((points[3][0] - points[0][0])**2 + (points[3][1] - points[0][1])**2)
        is_vertical = bbox_height > bbox_width
        
        # Calculate appropriate font size
        font_size = calculate_font_size(bbox_width, bbox_height, len(text.description), is_vertical)
        
        # Create font with calculated size
        try:
            font = ImageFont.truetype(base_font_name, font_size) if base_font_name else ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Calculate text angle
        angle = calculate_text_angle(points)
        # breakpoint()
        # Draw semi-transparent background (70% opacity)
        draw.polygon(points, fill=(0, 0, 0, 179))
        
        # Draw bounding box
        draw.polygon(points, outline='red')
        
        # Calculate center position
        center_x = sum(point[0] for point in points) / len(points)
        center_y = sum(point[1] for point in points) / len(points)
        
        # Create a temporary image for rotated text
        # Make the temporary image larger to accommodate rotated text
        max_dim = max(bbox_width, bbox_height) * 2
        txt = Image.new('RGBA', (int(max_dim), int(max_dim)), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt)
        
        # Draw text centered in temporary image
        txt_draw.text(
            (txt.width/2, txt.height/2),
            text.description,
            font=font,
            fill='white',
            anchor="mm"
        )
        
        # Rotate text
        txt = txt.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        
        # Calculate the offset to center the rotated text
        paste_x = int(center_x - txt.width/2)
        paste_y = int(center_y - txt.height/2)
        
        # Paste rotated text onto overlay
        overlay.paste(txt, (paste_x, paste_y), txt)
    
    # Composite the overlay with the original image
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    
    # Save the result
    if output_path is None:
        output_path = Path(image_path).parent / f"{Path(image_path).stem}_overlay.jpg"
    
    # Convert back to RGB for JPEG save
    result = result.convert('RGB')
    result.save(output_path, quality=95)
    print(f"Saved overlay visualization to: {output_path}")

def test_google_vision_overlay(image_path: str) -> None:
    """Test Google Cloud Vision API with visualization overlay."""
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
    
    if not texts:
        print("No text detected in the image.")
        return
    
    # Print full text
    print("\nFull text:")
    print(texts[0].description)
    
    # Draw overlay visualization
    draw_text_overlay(image_path, texts)
    
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_google_vision_overlay.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_google_vision_overlay(image_path) 