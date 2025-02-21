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
    
    def _detect_text_language(self, text: str) -> str:
        """Detect the primary language of the text based on character ranges."""
        japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF')  # Hiragana and Katakana
        chinese_chars = sum(1 for c in text if '\u4E00' <= c <= '\u9FFF')  # CJK Unified Ideographs
        latin_chars = sum(1 for c in text if '\u0000' <= c <= '\u007F')  # Basic Latin

        # If there are any Japanese-specific characters, prioritize Japanese
        if japanese_chars > 0:
            return 'ja'
        # If there are Chinese characters and no Japanese-specific ones, it's likely Chinese
        elif chinese_chars > 0:
            return 'zh'
        # Default to English for Latin characters or any other case
        else:
            return 'en'

    def _get_font(self, size: int, text: str = "") -> ImageFont.FreeTypeFont:
        """Get language-appropriate font with caching."""
        lang = self._detect_text_language(text)
        cache_key = f"{size}_{lang}"
        
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]
            
        # Windows system font paths
        system_font_path = "C:/Windows/Fonts/"
        
        # Language-specific font paths with full system paths
        font_paths = {
            'en': [
                system_font_path + "msgothic.ttf",    
                system_font_path + "arial.ttf",   
                system_font_path + "calibri.ttf"  
            ],
            'ja': [
                system_font_path + "meiryo.ttc",   # Meiryo
                system_font_path + "msgothic.ttc", # MS Gothic
                system_font_path + "yugothm.ttc"   # Yu Gothic Medium
            ],
            'zh': [
                system_font_path + "msjh.ttc",     # Microsoft JhengHei
                system_font_path + "msyh.ttc",     # Microsoft YaHei
                system_font_path + "simsun.ttc"    # SimSun
            ]
        }
        
        # Try fonts for detected language first
        font = None
        primary_paths = font_paths.get(lang, font_paths['en'])
        
        for font_path in primary_paths:
            try:
                font = ImageFont.truetype(font_path, size)
                logger.info(f"Successfully loaded {lang} font: {font_path}")
                break
            except Exception as e:
                logger.warning(f"Failed to load font {font_path}: {str(e)}")
                continue
        
        if font is None:
            logger.warning("Using default font - no suitable fonts found")
            font = ImageFont.load_default()
        
        self._font_cache[cache_key] = font
        return font

    def _calculate_font_size(self, box_width: float, box_height: float, text_length: int, is_vertical: bool = False) -> int:
        """Calculate optimal font size for text box."""
        MIN_FONT_SIZE = 14  # Increased minimum size
        MAX_FONT_SIZE = 32  # Keep maximum size
        
        if text_length <= 0 or box_width <= 0 or box_height <= 0:
            return MIN_FONT_SIZE
        
        if is_vertical:
            # For vertical text (Tategaki)
            char_height = box_height / (text_length * 1.2)  # Reduced spacing multiplier
            char_width = box_width * 0.8  # Increased width ratio
            base_size = min(char_width, char_height)
        else:
            # For horizontal text (Yokogaki)
            width_based_size = box_width / (text_length * 0.7)  # Reduced spacing multiplier
            height_based_size = box_height * 0.7  # Increased height ratio
            base_size = min(width_based_size, height_based_size)
        
        # Less aggressive size reduction for longer text
        if text_length > 20:
            base_size *= 0.85  # Reduced from 0.8
        elif text_length > 10:
            base_size *= 0.95  # Reduced from 0.9
        
        return int(max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, base_size)))
    
    def _calculate_true_angle(self, coords: List[Tuple[float, float]]) -> float:
        """Calculate actual text angle from coordinates."""
        try:
            # Get the points (assuming coords are in order: TL, TR, BR, BL)
            x1, y1 = coords[0]  # top-left
            x2, y2 = coords[1]  # top-right
            x3, y3 = coords[2]  # bottom-right
            x4, y4 = coords[3]  # bottom-left
            
            # Calculate width and height
            width = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            height = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)
            
            # Determine if text is more horizontal or vertical based on aspect ratio
            is_vertical = height > width * 1.2
            
            if is_vertical:
                # For vertical text, use the right edge (points[1] to points[2])
                angle = math.degrees(math.atan2(y3 - y2, x3 - x2))
            else:
                # For horizontal text, use the top edge (points[0] to points[1])
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
            
            logger.info(f"Angle calculation:")
            logger.info(f"  Is vertical: {is_vertical}")
            logger.info(f"  Box dimensions - width: {width:.2f}, height: {height:.2f}")
            logger.info(f"  Final angle: {angle:.2f}°")
            
            return angle
            
        except Exception as e:
            logger.warning(f"Error calculating true angle: {str(e)}")
            return 0.0

    def _should_render_vertical(self, coords: List[Tuple[float, float]], text: str, block_info: Dict = None) -> Dict[str, Any]:
        """Determine text rendering style including rotation and vertical/horizontal mode."""
        # Calculate basic dimensions
        width = max(p[0] for p in coords) - min(p[0] for p in coords)
        height = max(p[1] for p in coords) - min(p[1] for p in coords)
        aspect_ratio = height / width if width > 0 else float('inf')
        
        logger.info(f"\nText Direction Analysis for text: '{text[:20]}...'")
        logger.info(f"Initial dimensions - width: {width:.2f}, height: {height:.2f}, ratio: {height/width:.2f}")
        logger.info(f"Coordinates: {coords}")
        
        # Calculate true angle from coordinates
        true_angle = self._calculate_true_angle(coords)
        normalized_angle = abs(true_angle)
        
        logger.info(f"Angle analysis:")
        logger.info(f"  True angle: {true_angle:.2f}°")
        logger.info(f"  Normalized angle: {normalized_angle:.2f}°")
        
        # Check if text is vertical based on dimensions and angle
        is_tall = aspect_ratio > 2.0  # More strict threshold for vertical text
        is_near_vertical = normalized_angle > 75 and normalized_angle < 105
        
        # Check layout information from OCR
        layout_vertical = False
        text_flow = None
        if block_info and 'layout' in block_info:
            layout = block_info['layout']
            # Check for explicit text direction in OCR results
            if 'text_direction' in layout:
                text_flow = layout['text_direction']
                layout_vertical = text_flow in ['UP_TO_DOWN', 'DOWN_TO_UP']
            # Check line-level vertical flag
            elif 'line' in layout and 'is_vertical' in layout['line']:
                layout_vertical = layout['line']['is_vertical']
            # Check block-level writing direction
            elif 'block' in layout and 'writing_direction' in layout['block']:
                layout_vertical = layout['block']['writing_direction'].get('is_vertical', False)
            
            logger.info(f"  Layout info - vertical: {layout_vertical}, text_flow: {text_flow}")
        
        # Determine if this is rotated horizontal text
        # A text block is considered rotated horizontal if:
        # 1. It's near vertical angle (around 90 degrees)
        # 2. It's not explicitly marked as vertical by OCR
        # 3. Either:
        #    a) The text flow indicates horizontal direction, or
        #    b) The aspect ratio isn't extreme enough to suggest vertical writing
        is_rotated_horizontal = (
            is_near_vertical and 
            not layout_vertical and
            (text_flow in ['LEFT_TO_RIGHT', 'RIGHT_TO_LEFT'] or aspect_ratio < 3.0)
        )
        
        # Final decision on rotation and direction
        # Trust OCR's vertical detection, otherwise use our geometric analysis
        is_vertical = layout_vertical or (is_tall and not is_rotated_horizontal)
        should_rotate = abs(true_angle) > 5
        
        # Determine text flow based on OCR info or geometric analysis
        if not text_flow:
            text_flow = 'TOP_TO_BOTTOM' if is_vertical else 'LEFT_TO_RIGHT'
        
        # Create rendering style
        render_style = {
            'is_vertical': is_vertical,
            'rotation_angle': true_angle,
            'should_rotate': should_rotate,
            'text_flow': text_flow,
            'is_rotated_horizontal': is_rotated_horizontal
        }
        
        logger.info(f"Final rendering decision:")
        logger.info(f"  Is vertical: {is_vertical}")
        logger.info(f"  Should rotate: {should_rotate}")
        logger.info(f"  Rotation angle: {true_angle:.2f}°")
        logger.info(f"  Text flow: {text_flow}")
        logger.info(f"  Is rotated horizontal: {is_rotated_horizontal}")
        
        return render_style
    
    async def draw_text_overlay(
        self,
        image_path: str,
        text_blocks: List[Dict[str, Any]],
        output_path: str,
        use_translated_text: bool = False
    ) -> None:
        """Draw text overlay on image."""
        try:
            logger.info(f"Drawing {'translated' if use_translated_text else 'original'} overlay")
            
            with Image.open(image_path) as image:
                if image.mode != 'RGBA':
                    image = image.convert('RGBA')
                
                # Create overlays
                shading_overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                text_overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                shading_draw = ImageDraw.Draw(shading_overlay)
                text_draw = ImageDraw.Draw(text_overlay)
                
                # First pass: Draw all shading with reduced opacity
                for block in text_blocks:
                    coords = block.get('coordinates', block.get('bounding_box'))
                    if not coords or len(coords) != 4:
                        continue
                    shading_draw.polygon(coords, fill=(0, 0, 0, 195))  # Reduced opacity for better contrast
                
                # Second pass: Draw all text
                for block in text_blocks:
                    text = block['translated_text'] if use_translated_text and 'translated_text' in block else block.get('original_text', block.get('text', ''))
                    coords = block.get('coordinates', block.get('bounding_box'))
                    
                    if not text or not coords or len(coords) != 4:
                        continue

                    # Calculate dimensions and center point
                    width = max(p[0] for p in coords) - min(p[0] for p in coords)
                    height = max(p[1] for p in coords) - min(p[1] for p in coords)
                    min_x = min(p[0] for p in coords)
                    min_y = min(p[1] for p in coords)
                    center_x = min_x + width / 2
                    center_y = min_y + height / 2

                    # Calculate rotation angle
                    dx = coords[1][0] - coords[0][0]
                    dy = coords[1][1] - coords[0][1]
                    angle = math.degrees(math.atan2(dy, dx))

                    # Calculate font size and get language-appropriate font
                    is_vertical = height > width * 1.2
                    font_size = self._calculate_font_size(width, height, len(text), is_vertical)
                    font = self._get_font(font_size, text)

                    # Get text size
                    bbox = text_draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Create temporary image with enough space for text
                    temp_size = max(text_width, text_height) * 2  # Double the size to ensure enough space
                    temp_img = Image.new('RGBA', (temp_size, temp_size), (0, 0, 0, 0))
                    temp_draw = ImageDraw.Draw(temp_img)

                    # Draw text centered in temporary image
                    temp_draw.text(
                        (temp_size/2, temp_size/2),
                        text,
                        font=font,
                        fill=(255, 255, 255, 255),
                        anchor="mm"
                    )

                    # Rotate text
                    if abs(angle) > 5:
                        temp_img = temp_img.rotate(
                            -angle,  # Negative angle to counter the original rotation
                            expand=True,
                            resample=Image.Resampling.BICUBIC
                        )

                    # Calculate paste position to center the text
                    paste_x = int(center_x - temp_img.width/2)
                    paste_y = int(center_y - temp_img.height/2)

                    # Paste rotated text onto text overlay
                    text_overlay.paste(temp_img, (paste_x, paste_y), temp_img)
                
                # Composite the overlays in order: image -> shading -> text
                result = Image.alpha_composite(image, shading_overlay)
                result = Image.alpha_composite(result, text_overlay)
                
                # Convert to RGB before saving as JPEG
                if result.mode == 'RGBA':
                    result = result.convert('RGB')
                
                # Save the result
                result.save(output_path)
                logger.info(f"Saved overlay result to: {output_path}")
                
        except Exception as e:
            logger.error(f"Error drawing text overlay: {str(e)}")
            raise

# Create singleton instance
overlay_service = OverlayService() 