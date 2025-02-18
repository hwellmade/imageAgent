import sys
from typing import List, Dict, Any, Optional
from google.cloud import vision
import io
from PIL import Image, ImageDraw, ImageFont, ImageOps
import math
from pathlib import Path
import os
import logging
from datetime import datetime
import asyncio
from functools import lru_cache
import hashlib

# Configure service-specific logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    '%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
handler.flush()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Force immediate flush
handler.terminator = '\n'
handler.flush()

class OCRService:
    """Service for performing OCR on images using Google Cloud Vision."""
    
    def __init__(self):
        logger.info("[OCR Service] Initializing OCR Service")
        try:
            self._client = vision.ImageAnnotatorClient()
            # Test the client connection
            self._initialized = True
            logger.info("[OCR Service] Successfully initialized Google Vision client")
        except Exception as e:
            logger.error(f"[OCR Service] Failed to initialize Google Vision client: {str(e)}")
            self._initialized = False
            self._client = None
        
        # Get base directory
        self._base_dir = Path(__file__).parent.parent.parent
        self._upload_dir = self._base_dir / "uploads"
        # Ensure temp directory exists
        self._temp_dir = self._upload_dir / "temp"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[OCR Service] Directory setup complete")
        
        # Cache for font objects
        self._font_cache = {}
        # Cache for processed images
        self._image_cache = {}
        # Cache for OCR results
        self._ocr_cache = {}
        
        # Maximum cache size (adjust based on memory constraints)
        self._max_cache_size = 100

    def _reinitialize_client(self):
        """Reinitialize the Google Vision client if needed."""
        try:
            self._client = vision.ImageAnnotatorClient()
            self._initialized = True
            logger.info("[OCR Service] Successfully reinitialized Google Vision client")
            return True
        except Exception as e:
            logger.error(f"[OCR Service] Failed to reinitialize Google Vision client: {str(e)}")
            self._initialized = False
            self._client = None
            return False

    def _get_temp_path(self, filename: str) -> Path:
        """Get a temporary file path in the uploads directory."""
        return self._temp_dir / f"temp_{Path(filename).stem}.jpg"

    def _calculate_text_angle(self, points: list) -> float:
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

    def _calculate_font_size(self, bbox_width: float, bbox_height: float, text_length: int, is_vertical: bool) -> int:
        """Calculate appropriate font size based on bounding box dimensions and text length."""
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

    def _draw_text_overlay(self, image_path: str, texts: list, output_path: str | None = None) -> str:
        """Draw detected text and bounding boxes on the image."""
        start_time = datetime.now()
        logger.info(f"[OCR Service] Starting overlay drawing at {start_time.strftime('%H:%M:%S.%f')}")
        
        # Open the image and get its original orientation
        image_start = datetime.now()
        logger.info(f"[OCR Service] Starting image load at {image_start.strftime('%H:%M:%S.%f')}")
        image = Image.open(image_path)
        
        # Get original EXIF orientation
        try:
            exif = image._getexif()
            orientation = exif.get(274) if exif else None  # 274 is the orientation tag
            logger.info(f"[OCR Service] Original image orientation: {orientation}")
        except:
            orientation = None
            logger.info("[OCR Service] No EXIF orientation found")
        
        original_size = image.size
        logger.info(f"[OCR Service] Image loaded at {datetime.now().strftime('%H:%M:%S.%f')}, size: {original_size}")
        
        # Create overlay
        overlay = Image.new('RGBA', original_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Load font once and cache it
        font_start = datetime.now()
        logger.info(f"[OCR Service] Starting font loading at {font_start.strftime('%H:%M:%S.%f')}")
        font_cache = {}  # Cache fonts by size
        try:
            base_font_name = "msgothic.ttc"
            # Test with minimum size to verify font
            ImageFont.truetype(base_font_name, 12)
        except:
            try:
                base_font_name = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                ImageFont.truetype(base_font_name, 12)
            except:
                base_font_name = None
                logger.warning(f"[OCR Service] Failed to load custom font at {datetime.now().strftime('%H:%M:%S.%f')}, using default")
        font_end = datetime.now()
        logger.info(f"[OCR Service] Font loading completed at {font_end.strftime('%H:%M:%S.%f')}")
        logger.info(f"[OCR Service] Font loading duration: {(font_end - font_start).total_seconds():.2f}s")
        
        # Process each text block
        text_start = datetime.now()
        logger.info(f"[OCR Service] Starting text processing at {text_start.strftime('%H:%M:%S.%f')}")
        
        # Pre-calculate all text parameters first
        param_start = datetime.now()
        text_params = []
        for text in texts[1:]:
            vertices = text.bounding_poly.vertices
            points = [(vertex.x, vertex.y) for vertex in vertices]
            
            bbox_width = math.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
            bbox_height = math.sqrt((points[3][0] - points[0][0])**2 + (points[3][1] - points[0][1])**2)
            is_vertical = bbox_height > bbox_width
            
            font_size = self._calculate_font_size(bbox_width, bbox_height, len(text.description), is_vertical)
            
            # Get or create font
            if font_size not in font_cache:
                try:
                    font_cache[font_size] = ImageFont.truetype(base_font_name, font_size) if base_font_name else ImageFont.load_default()
                except:
                    font_cache[font_size] = ImageFont.load_default()
            
            angle = self._calculate_text_angle(points)
            center_x = sum(point[0] for point in points) / len(points)
            center_y = sum(point[1] for point in points) / len(points)
            
            text_params.append({
                'points': points,
                'font': font_cache[font_size],
                'angle': angle,
                'center': (center_x, center_y),
                'text': text.description,
                'bbox_size': (bbox_width, bbox_height)
            })
        logger.info(f"[OCR Service] Parameter calculation completed at {datetime.now().strftime('%H:%M:%S.%f')}")
        
        # Draw all backgrounds first
        bg_start = datetime.now()
        logger.info(f"[OCR Service] Starting background drawing at {bg_start.strftime('%H:%M:%S.%f')}")
        for params in text_params:
            draw.polygon(params['points'], fill=(0, 0, 0, 179))
            draw.polygon(params['points'], outline='red')
        logger.info(f"[OCR Service] Background drawing completed at {datetime.now().strftime('%H:%M:%S.%f')}")
        
        # Then draw all text
        text_draw_start = datetime.now()
        logger.info(f"[OCR Service] Starting text drawing at {text_draw_start.strftime('%H:%M:%S.%f')}")
        for params in text_params:
            bbox_width, bbox_height = params['bbox_size']
            max_dim = max(bbox_width, bbox_height) * 2
            txt = Image.new('RGBA', (int(max_dim), int(max_dim)), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt)
            
            txt_draw.text(
                (txt.width/2, txt.height/2),
                params['text'],
                font=params['font'],
                fill='white',
                anchor="mm"
            )
            
            txt = txt.rotate(params['angle'], expand=True, resample=Image.Resampling.BICUBIC)
            
            center_x, center_y = params['center']
            paste_x = int(center_x - txt.width/2)
            paste_y = int(center_y - txt.height/2)
            
            overlay.paste(txt, (paste_x, paste_y), txt)
        
        text_draw_end = datetime.now()
        logger.info(f"[OCR Service] Text drawing completed at {text_draw_end.strftime('%H:%M:%S.%f')}")
        logger.info(f"[OCR Service] Text drawing duration: {(text_draw_end - text_draw_start).total_seconds():.2f}s")
        
        # After drawing is complete, composite the images
        result = Image.alpha_composite(image.convert('RGBA'), overlay)
        
        # Rotate the result back to original orientation if needed
        if orientation:
            rotation_map = {
                3: 180,
                6: 270,
                8: 90
            }
            if orientation in rotation_map:
                result = result.rotate(rotation_map[orientation], expand=True)
                logger.info(f"[OCR Service] Rotated result by {rotation_map[orientation]} degrees")
        
        # Convert back to RGB for saving
        result = result.convert('RGB')
        result.save(output_path, quality=95)
        
        end_time = datetime.now()
        logger.info(f"[OCR Service] Overlay drawing completed at {end_time.strftime('%H:%M:%S.%f')}")
        logger.info(f"[OCR Service] Total overlay drawing duration: {(end_time - start_time).total_seconds():.2f}s")
        return output_path

    def _get_image_hash(self, image_content: bytes) -> str:
        """Generate a hash for image content."""
        return hashlib.md5(image_content).hexdigest()
        
    def _optimize_image(self, image: Image.Image, max_dimension: int = 1920) -> Image.Image:
        """Optimize image size and quality for OCR."""
        # Resize if needed
        width, height = image.size
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int((height * max_dimension) / width)
            else:
                new_height = max_dimension
                new_width = int((width * max_dimension) / height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Enhance image for better OCR
        image = ImageOps.autocontrast(image)
        
        return image
        
    @lru_cache(maxsize=100)
    def _get_font(self, font_name: str, size: int) -> ImageFont.FreeTypeFont:
        """Cached font loading."""
        try:
            return ImageFont.truetype(font_name, size)
        except:
            return ImageFont.load_default()
            
    async def detect_text(self, 
                         file: Any,
                         source_lang: str = 'auto',
                         saved_file_path: str | None = None) -> tuple[list[dict], str | None]:
        """Detect text in the image using Google Cloud Vision with caching."""
        start_time = datetime.now()
        logger.info(f"[OCR Service] Starting text detection at {start_time.strftime('%H:%M:%S.%f')}")
        
        try:
            # Read and hash image content
            read_start = datetime.now()
            if hasattr(file, 'read'):
                contents = await file.read()
            else:
                with open(saved_file_path, 'rb') as f:
                    contents = f.read()
            
            image_hash = self._get_image_hash(contents)
            
            # Check cache for existing results
            if image_hash in self._ocr_cache:
                logger.info("[OCR Service] Using cached OCR results")
                return self._ocr_cache[image_hash]
            
            # Optimize image before OCR
            image = Image.open(io.BytesIO(contents))
            optimized_image = self._optimize_image(image)
            optimized_contents = io.BytesIO()
            optimized_image.save(optimized_contents, format='JPEG', quality=85)
            optimized_contents = optimized_contents.getvalue()
            
            # Perform OCR with optimized image
            vision_image = vision.Image(content=optimized_contents)
            response = self._client.text_detection(image=vision_image)
            texts = response.text_annotations
            
            if not texts:
                return [], None
                
            # Generate overlay with optimized image
            if saved_file_path:
                # Save optimized image
                optimized_image.save(saved_file_path, quality=85)
                output_path = str(Path(saved_file_path).parent / f"{Path(saved_file_path).stem}_overlay.jpg")
            else:
                temp_path = self._get_temp_path("temp_image.jpg")
                optimized_image.save(temp_path, quality=85)
                output_path = str(Path(temp_path).parent / f"{Path(temp_path).stem}_overlay.jpg")
            
            # Draw overlay
            self._draw_text_overlay(saved_file_path or temp_path, texts, output_path)
            
            # Format and cache results
            detected_texts = []
            for text in texts[1:]:
                vertices = text.bounding_poly.vertices
                bbox = [[vertex.x, vertex.y] for vertex in vertices]
                detected_texts.append({
                    'text': text.description.strip(),
                    'confidence': getattr(text, 'confidence', 0.99),
                    'bounding_box': bbox,
                    'language': source_lang
                })
            
            relative_output_path = str(Path(output_path).relative_to(self._base_dir)).replace("\\", "/")
            if not relative_output_path.startswith('uploads/'):
                relative_output_path = f"uploads/{relative_output_path}"
                
            # Cache results
            result = (detected_texts, relative_output_path)
            self._ocr_cache[image_hash] = result
            
            # Manage cache size
            if len(self._ocr_cache) > self._max_cache_size:
                # Remove oldest entry
                self._ocr_cache.pop(next(iter(self._ocr_cache)))
                
            return result
            
        except Exception as e:
            logger.error(f"[OCR Service] Error: {str(e)}")
            raise

# Create a singleton instance
ocr_service = OCRService() 