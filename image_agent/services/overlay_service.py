import logging
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from pathlib import Path

logger = logging.getLogger(__name__)

class OverlayService:
    """Service for drawing text overlays on images."""
    
    def __init__(self):
        self._font_cache = {}
        logger.info("Overlay Service initialized")
    
    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get font with caching for better performance."""
        if size in self._font_cache:
            return self._font_cache[size]
            
        try:
            font = ImageFont.truetype("msgothic.ttc", size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
            except:
                font = ImageFont.load_default()
                logger.warning("Using default font - proper fonts not found")
        
        self._font_cache[size] = font
        return font
    
    def _calculate_font_size(self, box_width: float, box_height: float, text_length: int) -> int:
        """Calculate optimal font size for text box."""
        MIN_FONT_SIZE = 12
        MAX_FONT_SIZE = 32
        
        # Calculate base size from box dimensions
        width_based_size = box_width / (text_length * 0.7)  # Allow for character spacing
        height_based_size = box_height * 0.8  # Leave margin
        
        # Use the smaller of the two sizes
        base_size = min(width_based_size, height_based_size)
        
        # Clamp to reasonable limits
        return int(max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, base_size)))
    
    async def draw_text_overlay(
        self,
        image_path: str,
        text_blocks: List[Dict[str, Any]],
        output_path: str,
        use_translated_text: bool = False
    ) -> None:
        """Draw text overlay on image.
        
        Args:
            image_path: Path to original image
            text_blocks: List of blocks containing:
                - coordinates: List[List[int]] (from OCR)
                - original_text: str (matched from LLM)
                - translated_text: str (from LLM, optional)
            output_path: Where to save the overlay image
            use_translated_text: Whether to use translated text
        """
        try:
            logger.info(f"Drawing {'translated' if use_translated_text else 'original'} overlay")
            
            # Open and prepare image
            with Image.open(image_path) as image:
                if image.mode != 'RGBA':
                    image = image.convert('RGBA')
                
                # Create overlay
                overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                
                # Process each text block
                for block in text_blocks:
                    # Get text content
                    text = block['translated_text'] if use_translated_text and 'translated_text' in block else block.get('original_text', block.get('text', ''))
                    
                    # Get coordinates (support both field names)
                    coords = block.get('coordinates', block.get('bounding_box'))
                    if not coords or len(coords) != 4:
                        logger.warning(f"Invalid coordinates for text: {text[:20]}...")
                        continue
                    
                    # Calculate box dimensions
                    width = max(p[0] for p in coords) - min(p[0] for p in coords)
                    height = max(p[1] for p in coords) - min(p[1] for p in coords)
                    
                    # Draw semi-transparent background
                    draw.polygon(coords, fill=(0, 0, 0, 128))
                    
                    # Calculate font size
                    font_size = self._calculate_font_size(width, height, len(text))
                    font = self._get_font(font_size)
                    
                    # Calculate text position (center of box)
                    center_x = sum(p[0] for p in coords) / len(coords)
                    center_y = sum(p[1] for p in coords) / len(coords)
                    
                    # Draw text
                    draw.text(
                        (center_x, center_y),
                        text,
                        font=font,
                        fill=(255, 255, 255, 255),
                        anchor="mm"
                    )
                
                # Composite overlay with original image
                result = Image.alpha_composite(image, overlay)
                
                # Save result
                result.convert('RGB').save(output_path, 'JPEG', quality=95)
                logger.info(f"Saved overlay to: {output_path}")
                
        except Exception as e:
            logger.error(f"Error drawing overlay: {str(e)}", exc_info=True)
            raise

# Create singleton instance
overlay_service = OverlayService() 