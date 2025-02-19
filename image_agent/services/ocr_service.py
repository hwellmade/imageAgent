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
        is_vertical = height > width * 1.2
        
        if is_vertical:
            # For vertical text, use the left edge (points[0] to points[3])
            angle = math.degrees(math.atan2(y3 - y1, x3 - x1))
            # Add 90 degrees for vertical text
            angle += 90
        else:
            # For horizontal text, use the bottom edge (points[0] to points[1])
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        # Normalize angle to be between -90 and 90 degrees
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180
            
        # Reverse the angle to match the correct rotation direction
        angle = -angle
            
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

    def _calculate_uniform_font_size(self, texts: list, image_height: int) -> int:
        """Calculate a uniform font size based on the average text block height."""
        # Get heights of text blocks
        heights = []
        for text in texts:
            vertices = text.bounding_poly.vertices
            points = [(vertex.x, vertex.y) for vertex in vertices]
            height = max(p[1] for p in points) - min(p[1] for p in points)
            heights.append(height)
        
        if not heights:
            return 10  # Default minimum font size
        
        # Use median height to avoid outliers
        median_height = sorted(heights)[len(heights)//2]
        
        # Scale font size relative to image height (further reduced scaling factors)
        relative_size = median_height / image_height
        base_font_size = int(relative_size * 12)  # Reduced from 20 to 12
        breakpoint()
        # Clamp to reasonable limits
        return max(6, min(10, base_font_size))  # Reduced from (8, 16) to (6, 10)

    def _calculate_block_angle(self, points: list) -> float:
        """Calculate the angle of a block based on its bounding box points."""
        # Get the points
        x1, y1 = points[0]  # bottom-left
        x2, y2 = points[1]  # bottom-right
        x3, y3 = points[2]  # top-right
        x4, y4 = points[3]  # top-left
        
        # Calculate width and height
        width = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        height = math.sqrt((x4 - x1)**2 + (y4 - y1)**2)
        
        # Determine if text is vertical based on aspect ratio
        is_vertical = height > width * 1.2
        
        if is_vertical:
            # For vertical text, use the left edge (bottom to top)
            angle = math.degrees(math.atan2(y4 - y1, x4 - x1))
            # Add 90 degrees for vertical text
            angle += 90
        else:
            # For horizontal text, use the bottom edge (left to right)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
        # Normalize angle to be between -180 and 180 degrees
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
            
        # For vertical text, we need to adjust the angle
        if is_vertical:
            if angle > 0:
                angle -= 180
            
        return angle

    def _calculate_line_font_size(self, line_bbox: list, text: str, is_vertical: bool) -> int:
        """Calculate appropriate font size for a line to fit within its bounding box."""
        # Calculate actual box dimensions
        width = math.sqrt((line_bbox[1][0] - line_bbox[0][0])**2 + 
                         (line_bbox[1][1] - line_bbox[0][1])**2)
        height = math.sqrt((line_bbox[3][0] - line_bbox[0][0])**2 + 
                          (line_bbox[3][1] - line_bbox[0][1])**2)
        
        # Use a more conservative space calculation
        if is_vertical:
            # For vertical text, use width as the constraint
            available_space = width * 0.7  # Use 70% of width
            text_space_needed = height / (len(text) + 1)  # Add 1 for spacing
        else:
            # For horizontal text, use height as the constraint
            available_space = height * 0.7  # Use 70% of height
            # Estimate characters per line based on width/height ratio
            chars_per_line = max(1, min(len(text), width / height * 1.5))
            text_space_needed = width / chars_per_line
        
        # Calculate base size
        base_size = min(available_space, text_space_needed)
        
        # More conservative font size limits
        return int(max(10, min(32, base_size)))

    def _draw_text_overlay(self, image_path: str, texts: list, output_path: str | None = None) -> str:
        """Draw detected text and bounding boxes on the image."""
        start_time = datetime.now()
        logger.info(f"[OCR Service] Starting overlay drawing at {start_time.strftime('%H:%M:%S.%f')}")
        
        # Open the image and get its original orientation
        image_start = datetime.now()
        logger.info(f"[OCR Service] Starting image load at {image_start.strftime('%H:%M:%S.%f')}")
        image = Image.open(image_path)
        
        # Calculate uniform font size based on image size
        uniform_font_size = self._calculate_uniform_font_size(texts[1:], image.height)
        logger.info(f"[OCR Service] Using uniform font size: {uniform_font_size}")
        
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
        
        # Load font once for all text
        font_start = datetime.now()
        logger.info(f"[OCR Service] Starting font loading at {font_start.strftime('%H:%M:%S.%f')}")
        try:
            base_font_name = "msgothic.ttc"
            font = ImageFont.truetype(base_font_name, uniform_font_size)
        except:
            try:
                base_font_name = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                font = ImageFont.truetype(base_font_name, uniform_font_size)
            except:
                font = ImageFont.load_default()
                logger.warning(f"[OCR Service] Failed to load custom font at {datetime.now().strftime('%H:%M:%S.%f')}, using default")
        
        # Process each text block
        for text in texts[1:]:  # Skip the first text as it contains all text
            vertices = text.bounding_box.vertices
            points = [(vertex.x, vertex.y) for vertex in vertices]
            
            # Draw semi-transparent background
            draw.polygon(points, fill=(0, 0, 0, 179))
            draw.polygon(points, outline='red')
            
            # Calculate center position
            center_x = sum(point[0] for point in points) / len(points)
            center_y = sum(point[1] for point in points) / len(points)
            
            # Calculate text angle
            angle = self._calculate_text_angle(points)
            
            # Create temporary image for rotated text
            bbox_width = max(p[0] for p in points) - min(p[0] for p in points)
            bbox_height = max(p[1] for p in points) - min(p[1] for p in points)
            max_dim = max(bbox_width, bbox_height) * 2
            
            txt = Image.new('RGBA', (int(max_dim), int(max_dim)), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt)
            
            # Preserve original text including symbols and punctuation
            txt_draw.text(
                (txt.width/2, txt.height/2),
                text.description,
                font=font,
                fill='white',
                anchor="mm"
            )
            # breakpoint()
            # Rotate text
            if angle < 0:
                txt = txt.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC)
            else:
                txt = txt.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
            
            # Calculate paste position
            paste_x = int(center_x - txt.width/2)
            paste_y = int(center_y - txt.height/2)
            
            # Paste rotated text
            overlay.paste(txt, (paste_x, paste_y), txt)
        
        # Composite the overlay with the original image
        result = Image.alpha_composite(image.convert('RGBA'), overlay)
        
        # Handle orientation
        if orientation:
            rotation_map = {3: 180, 6: 270, 8: 90}
            if orientation in rotation_map:
                result = result.rotate(rotation_map[orientation], expand=True)
        
        # Save the result
        if output_path is None:
            output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_overlay.jpg")
        
        result = result.convert('RGB')
        result.save(output_path, quality=95)
        
        logger.info(f"[OCR Service] Overlay drawing completed in {(datetime.now() - start_time).total_seconds():.2f}s")
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
        """Detect text in the image using Google Cloud Vision with hierarchical text detection."""
        start_time = datetime.now()
        logger.info(f"[OCR Service] Starting text detection at {start_time.strftime('%H:%M:%S.%f')}")
        
        image_content = None
        temp_file = None
        
        try:
            # Read and hash image content
            read_start = datetime.now()
            if hasattr(file, 'read'):
                image_content = await file.read()
            else:
                with open(saved_file_path, 'rb') as f:
                    image_content = f.read()
            
            image_hash = self._get_image_hash(image_content)
            
            # Check cache for existing results
            if image_hash in self._ocr_cache:
                logger.info("[OCR Service] Using cached OCR results")
                return self._ocr_cache[image_hash]
            
            # Create a temporary file for the optimized image
            temp_file = self._get_temp_path("temp_image.jpg")
            
            # Optimize image before OCR
            with Image.open(io.BytesIO(image_content)) as image:
                optimized_image = self._optimize_image(image)
                # Save optimized image to temporary file
                optimized_image.save(temp_file, format='JPEG', quality=85)
                
                # Read the optimized image for OCR
                with open(temp_file, 'rb') as f:
                    optimized_contents = f.read()
            
            # Perform OCR with optimized image using document text detection
            vision_image = vision.Image(content=optimized_contents)
            response = self._client.document_text_detection(
                image=vision_image,
                image_context={"language_hints": [source_lang] if source_lang != 'auto' else []}
            )
            document = response.full_text_annotation
            
            if not document.text:
                return [], None
                
            # Extract text with hierarchical structure
            detected_texts = []
            
            # Process each page (usually just one)
            for page in document.pages:
                # Process each block
                for block in page.blocks:
                    # Get block bounding box
                    if block.bounding_box:
                        block_bbox = [[vertex.x, vertex.y] 
                                    for vertex in block.bounding_box.vertices]
                        
                        # Store lines for this block
                        block_lines = []
                        current_line = {
                            'words': [],
                            'bbox': None,
                            'text': '',
                            'confidence': 0.0
                        }
                        
                        # Process each paragraph
                        for paragraph in block.paragraphs:
                            last_word_y = None
                            
                            # Process each word
                            for word in paragraph.words:
                                if word.bounding_box:
                                    word_bbox = [[vertex.x, vertex.y] 
                                               for vertex in word.bounding_box.vertices]
                                    word_text = ''.join([symbol.text for symbol in word.symbols])
                                    word_confidence = word.confidence
                                    
                                    # Calculate word's center y-coordinate
                                    word_y = sum(p[1] for p in word_bbox) / len(word_bbox)
                                    
                                    # If this is a new line (based on y-position difference)
                                    if last_word_y is not None and abs(word_y - last_word_y) > (word_bbox[3][1] - word_bbox[0][1]) * 0.5:
                                        # Save current line
                                        if current_line['words']:
                                            block_lines.append(current_line)
                                            current_line = {
                                                'words': [],
                                                'bbox': None,
                                                'text': '',
                                                'confidence': 0.0
                                            }
                                    
                                    # Add word to current line
                                    current_line['words'].append({
                                        'text': word_text,
                                        'bbox': word_bbox,
                                        'confidence': word_confidence
                                    })
                                    last_word_y = word_y
                        
                        # Add last line if not empty
                        if current_line['words']:
                            block_lines.append(current_line)
                        
                        # Process each line in the block
                        for line in block_lines:
                            # Calculate line bounding box
                            if line['words']:
                                all_points = [p for word in line['words'] for p in word['bbox']]
                                min_x = min(p[0] for p in all_points)
                                max_x = max(p[0] for p in all_points)
                                min_y = min(p[1] for p in all_points)
                                max_y = max(p[1] for p in all_points)
                                line['bbox'] = [
                                    [min_x, max_y],  # bottom-left
                                    [max_x, max_y],  # bottom-right
                                    [max_x, min_y],  # top-right
                                    [min_x, min_y]   # top-left
                                ]
                                line['text'] = ' '.join(word['text'] for word in line['words'])
                                line['confidence'] = sum(word['confidence'] for word in line['words']) / len(line['words'])
                                
                                # Add line to detected texts
                                detected_texts.append({
                                    'text': line['text'],
                                    'confidence': line['confidence'],
                                    'bounding_box': line['bbox'],
                                    'block_bbox': block_bbox,  # Store the parent block's bounding box
                                    'language': source_lang,
                                    'block_type': block.block_type.name,
                                    'is_line': True
                                })
            
            # Find and print the longest text block
            if detected_texts:
                longest_text = max(detected_texts, key=lambda x: len(x['text']))
                logger.info("\n=== Longest Text Block Information ===")
                logger.info(f"Text content: {longest_text['text']}")
                logger.info(f"Text length: {len(longest_text['text'])} characters")
                logger.info(f"Bounding box coordinates: {longest_text['bounding_box']}")
                logger.info(f"Block bounding box coordinates: {longest_text['block_bbox']}")
                logger.info(f"Confidence: {longest_text['confidence']:.2%}")
                logger.info("=====================================\n")
            
            # Sort lines by vertical position and then horizontal position
            detected_texts.sort(key=lambda x: (
                min(p[1] for p in x['bounding_box']),  # y coordinate
                min(p[0] for p in x['bounding_box'])   # x coordinate
            ))
            
            # Generate overlay with optimized image
            if saved_file_path:
                output_path = str(Path(saved_file_path).parent / f"{Path(saved_file_path).stem}_overlay.jpg")
                # Copy optimized image to saved_file_path
                with open(temp_file, 'rb') as src, open(saved_file_path, 'wb') as dst:
                    dst.write(src.read())
            else:
                output_path = str(Path(temp_file).parent / f"{Path(temp_file).stem}_overlay.jpg")
            
            # Draw overlay using line-level text detection
            self._draw_text_overlay_by_line(temp_file, detected_texts, output_path)
            
            relative_output_path = str(Path(output_path).relative_to(self._base_dir)).replace("\\", "/")
            if not relative_output_path.startswith('uploads/'):
                relative_output_path = f"uploads/{relative_output_path}"
                
            # Cache results
            result = (detected_texts, relative_output_path)
            self._ocr_cache[image_hash] = result
            
            # Manage cache size
            if len(self._ocr_cache) > self._max_cache_size:
                self._ocr_cache.pop(next(iter(self._ocr_cache)))
                
            return result
            
        except Exception as e:
            logger.error(f"[OCR Service] Error: {str(e)}")
            raise
        finally:
            # Clean up temporary file
            if temp_file and Path(temp_file).exists():
                try:
                    Path(temp_file).unlink()
                except Exception as e:
                    logger.warning(f"[OCR Service] Failed to clean up temporary file: {str(e)}")

    def _draw_text_overlay_by_line(self, image_path: str, texts: list, output_path: str) -> None:
        """Draw detected text by lines while respecting block boundaries."""
        try:
            # Open the image
            with Image.open(image_path) as image:
                # Create overlay
                overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                
                # Try to load Japanese font
                try:
                    font = ImageFont.truetype("msgothic.ttc", 1)  # Load with size 1 first
                except:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 1)
                    except:
                        font = ImageFont.load_default()
                
                # Group lines by their parent block
                block_lines = {}
                for text_item in texts:
                    # Convert coordinates to float
                    text_item['block_bbox'] = [[float(x), float(y)] for x, y in text_item['block_bbox']]
                    text_item['bounding_box'] = [[float(x), float(y)] for x, y in text_item['bounding_box']]
                    
                    block_key = str(text_item['block_bbox'])  # Use block bbox as key
                    if block_key not in block_lines:
                        block_lines[block_key] = []
                    block_lines[block_key].append(text_item)
                
                # Process each block
                for block_key, lines in block_lines.items():
                    # Draw block background
                    block_bbox = lines[0]['block_bbox']  # All lines in this group share the same block_bbox
                    draw.polygon([(x, y) for x, y in block_bbox], fill=(0, 0, 0, 160))
                    draw.polygon([(x, y) for x, y in block_bbox], outline=(65, 105, 225), width=1)
                    
                    # Calculate block dimensions and angle
                    block_angle = self._calculate_block_angle(block_bbox)
                    block_width = math.sqrt((block_bbox[1][0] - block_bbox[0][0])**2 + 
                                          (block_bbox[1][1] - block_bbox[0][1])**2)
                    block_height = math.sqrt((block_bbox[3][0] - block_bbox[0][0])**2 + 
                                           (block_bbox[3][1] - block_bbox[0][1])**2)
                    
                    # Determine if block is vertical
                    is_vertical = block_height > block_width * 1.2
                    
                    # Process each line in the block
                    for line in lines:
                        # Get line bounding box
                        line_bbox = line['bounding_box']
                        text = line['text']
                        
                        # Calculate font size based on line dimensions
                        font_size = self._calculate_line_font_size(line_bbox, text, is_vertical)
                        
                        try:
                            current_font = ImageFont.truetype(font.path, font_size)
                        except:
                            current_font = ImageFont.load_default()
                        
                        # Calculate line center
                        line_center_x = sum(p[0] for p in line_bbox) / 4
                        line_center_y = sum(p[1] for p in line_bbox) / 4
                        
                        # Get text dimensions
                        bbox = draw.textbbox((0, 0), text, font=current_font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        
                        # Calculate temporary image size (more precise)
                        if is_vertical:
                            temp_width = int(text_height * 1.2)  # More space for vertical text
                            temp_height = int(text_width * 1.2)
                        else:
                            temp_width = int(text_width * 1.2)
                            temp_height = int(text_height * 1.2)
                        
                        # Create temporary image for rotated text
                        txt = Image.new('RGBA', (temp_width, temp_height), (0, 0, 0, 0))
                        txt_draw = ImageDraw.Draw(txt)
                        
                        # Draw text centered in temporary image
                        txt_draw.text(
                            (temp_width/2, temp_height/2),
                            text,
                            font=current_font,
                            fill=(255, 255, 255, 255),
                            anchor="mm"
                        )
                        
                        # Rotate text using block angle
                        if block_angle < 0 :
                            txt = txt.rotate(-block_angle, expand=True, resample=Image.Resampling.BICUBIC)
                        else:
                            txt = txt.rotate(block_angle, expand=True, resample=Image.Resampling.BICUBIC)
                        
                        # Calculate paste position (adjusted to better fit within bounds)
                        paste_x = int(line_center_x - txt.width/2)
                        paste_y = int(line_center_y - txt.height/2)
                        
                        # Draw rotated text background
                        bg_txt = Image.new('RGBA', txt.size, (0, 0, 0, 200))
                        overlay.paste(bg_txt, (paste_x, paste_y), txt)
                        
                        # Draw rotated text
                        overlay.paste(txt, (paste_x, paste_y), txt)
                
                # Composite the overlay with the original image
                result = Image.alpha_composite(image.convert('RGBA'), overlay)
                
                # Save the result
                result = result.convert('RGB')
                result.save(output_path, quality=95)
                
                logger.info(f"[OCR Service] Saved line-level overlay visualization to: {output_path}")
        except Exception as e:
            logger.error(f"[OCR Service] Error in drawing overlay: {str(e)}")
            raise

# Create a singleton instance
ocr_service = OCRService() 