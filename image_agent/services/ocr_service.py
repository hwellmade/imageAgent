import sys
from typing import List, Dict, Any, Tuple, Optional
from google.cloud import vision
import io
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance, ImageFilter
import math
from pathlib import Path
import os
import logging
from datetime import datetime
import asyncio
from functools import lru_cache
import hashlib
import json
import numpy as np

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
    """Service for performing OCR using Google Cloud Vision."""
    
    def __init__(self):
        logger.info("[OCR Service] Initializing OCR Service")
        try:
            self._client = vision.ImageAnnotatorClient()
            self._initialized = True
            logger.info("[OCR Service] Successfully initialized Google Vision client")
        except Exception as e:
            logger.error(f"[OCR Service] Failed to initialize Google Vision client: {str(e)}")
            self._initialized = False
            self._client = None
            
        # Cache for OCR results
        self._ocr_cache = {}
        
    def _reinitialize_client(self) -> bool:
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
    
    def _get_image_hash(self, image_content: bytes) -> str:
        """Generate a hash for image content."""
        return hashlib.md5(image_content).hexdigest()
    
    def _validate_text_block(self, text_block: vision.TextAnnotation) -> bool:
        """Validate a text block meets minimum requirements."""
        # Check if block has text content
        if not hasattr(text_block, 'description') or not text_block.description.strip():
            return False
            
        # Check if block has valid coordinates
        if not hasattr(text_block, 'bounding_poly') or not text_block.bounding_poly.vertices:
            return False
        
        return True
    
    def _group_blocks_into_lines(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group text blocks into lines based on vertical position and overlap."""
        if not blocks:
            return []
            
        # Sort blocks by vertical position (y-coordinate)
        sorted_blocks = sorted(blocks, key=lambda b: sum(p[1] for p in b['bounding_box']) / len(b['bounding_box']))
        
        lines = []
        current_line = [sorted_blocks[0]]
        current_y = sum(p[1] for p in sorted_blocks[0]['bounding_box']) / len(sorted_blocks[0]['bounding_box'])
        
        # Threshold for considering blocks part of the same line (adjust if needed)
        LINE_HEIGHT_THRESHOLD = 20  # pixels
        
        for block in sorted_blocks[1:]:
            block_y = sum(p[1] for p in block['bounding_box']) / len(block['bounding_box'])
            
            # If block is close enough vertically, add to current line
            if abs(block_y - current_y) <= LINE_HEIGHT_THRESHOLD:
                current_line.append(block)
            else:
                # Sort blocks in current line by x-coordinate
                current_line.sort(key=lambda b: sum(p[0] for p in b['bounding_box']) / len(b['bounding_box']))
                
                # Create line entry
                line_text = ' '.join(block['text'] for block in current_line)
                
                # Calculate line bounding box
                all_points = [p for block in current_line for p in block['bounding_box']]
                min_x = min(p[0] for p in all_points)
                min_y = min(p[1] for p in all_points)
                max_x = max(p[0] for p in all_points)
                max_y = max(p[1] for p in all_points)
                
                line = {
                    'text': line_text,
                    'original_text': line_text,
                    'bounding_box': [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)],
                    'coordinates': [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)],
                    'block_type': 'LINE',
                    'original_blocks': current_line
                }
                
                # Log line information
                logger.debug(f"[OCR Service] Created line: '{line_text}'")
                logger.debug(f"[OCR Service] Line contains {len(current_line)} blocks")
                
                lines.append(line)
                
                # Start new line
                current_line = [block]
                current_y = block_y
        
        # Add last line with same logic
        if current_line:
            current_line.sort(key=lambda b: sum(p[0] for p in b['bounding_box']) / len(b['bounding_box']))
            line_text = ' '.join(block['text'] for block in current_line)
            
            all_points = [p for block in current_line for p in block['bounding_box']]
            min_x = min(p[0] for p in all_points)
            min_y = min(p[1] for p in all_points)
            max_x = max(p[0] for p in all_points)
            max_y = max(p[1] for p in all_points)
            
            line = {
                'text': line_text,
                'original_text': line_text,
                'bounding_box': [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)],
                'coordinates': [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)],
                'block_type': 'LINE',
                'original_blocks': current_line
            }
            
            logger.debug(f"[OCR Service] Created final line: '{line_text}'")
            logger.debug(f"[OCR Service] Line contains {len(current_line)} blocks")
            
            lines.append(line)
        
        return lines

    def _preprocess_image(self, image_content: bytes) -> bytes:
        """Preprocess image to improve OCR quality with simple but effective enhancements."""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_content))
            
            # Store original format
            original_format = image.format or 'PNG'
            
            # Convert to RGB if needed
            if image.mode not in ['L', 'RGB']:
                image = image.convert('RGB')
            
            # Resize if image is too large (helps with processing speed and memory)
            max_dimension = 2000
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"[OCR Service] Resized image to {new_size}")
            
            # Convert to grayscale
            gray = ImageOps.grayscale(image)
            
            # Invert colors (since we have white text on dark background)
            inverted = ImageOps.invert(gray)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(inverted)
            enhanced = enhancer.enhance(2.0)
            
            # Enhance sharpness
            sharpener = ImageEnhance.Sharpness(enhanced)
            final = sharpener.enhance(2.0)
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            final.save(output_buffer, format=original_format, quality=95)
            processed_content = output_buffer.getvalue()
            
            # Save debug images if needed
            if logger.getEffectiveLevel() <= logging.DEBUG:
                debug_dir = Path("debug_images")
                debug_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                stages = {
                    "1_original": image,
                    "2_grayscale": gray,
                    "3_inverted": inverted,
                    "4_enhanced_contrast": enhanced,
                    "5_final_sharpened": final
                }
                
                for stage_name, stage_image in stages.items():
                    stage_image.save(debug_dir / f"{stage_name}_{timestamp}.png")
            
            logger.info("[OCR Service] Successfully preprocessed image with simplified pipeline")
            return processed_content
            
        except Exception as e:
            logger.error(f"[OCR Service] Image preprocessing failed: {str(e)}", exc_info=True)
            return image_content  # Return original content if preprocessing fails

    def _process_text_blocks(self, response: vision.TextAnnotation) -> List[Dict[str, Any]]:
        """Process text blocks from Google Vision response to get line-level information."""
        if not response.full_text_annotation:
            return []
            
        processed_lines = []
        current_line_text = []
        current_line_vertices = []
        
        # Process each page
        for page in response.full_text_annotation.pages:
            # Process each block
            for block in page.blocks:
                # Process each paragraph
                for paragraph in block.paragraphs:
                    # Process each word
                    for word in paragraph.words:
                        # Get word text
                        word_text = ''.join(symbol.text for symbol in word.symbols)
                        current_line_text.append(word_text)
                        
                        # Get word vertices
                        word_vertices = [(vertex.x, vertex.y) for vertex in word.bounding_box.vertices]
                        current_line_vertices.extend(word_vertices)
                        
                        # Check if this word ends a line
                        # The last symbol of each word contains break information
                        last_symbol = word.symbols[-1]
                        is_line_break = (hasattr(last_symbol, 'property') and 
                                       hasattr(last_symbol.property, 'detected_break') and
                                       last_symbol.property.detected_break.type in [
                                           vision.TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE,
                                           vision.TextAnnotation.DetectedBreak.BreakType.LINE_BREAK
                                       ])
                        
                        # If we have a line break or this is the last word, create a line entry
                        if is_line_break or (word == paragraph.words[-1] and current_line_text):
                            # Calculate line bounding box from accumulated vertices
                            min_x = min(x for x, _ in current_line_vertices)
                            min_y = min(y for _, y in current_line_vertices)
                            max_x = max(x for x, _ in current_line_vertices)
                            max_y = max(y for _, y in current_line_vertices)
                            
                            line_box = [
                                (min_x, min_y),  # top-left
                                (max_x, min_y),  # top-right
                                (max_x, max_y),  # bottom-right
                                (min_x, max_y)   # bottom-left
                            ]
                            
                            # Create line entry
                            line = {
                                'text': ' '.join(current_line_text),
                                'original_text': ' '.join(current_line_text),
                                'bounding_box': line_box,
                                'coordinates': line_box,
                                'block_type': 'LINE'
                            }
                            
                            # Log line details
                            logger.debug(f"\nProcessed line:")
                            logger.debug(f"Text: {line['text']}")
                            logger.debug(f"Bounding Box:")
                            logger.debug(f"  Top-left: ({line_box[0][0]}, {line_box[0][1]})")
                            logger.debug(f"  Top-right: ({line_box[1][0]}, {line_box[1][1]})")
                            logger.debug(f"  Bottom-right: ({line_box[2][0]}, {line_box[2][1]})")
                            logger.debug(f"  Bottom-left: ({line_box[3][0]}, {line_box[3][1]})")
                            
                            processed_lines.append(line)
                            
                            # Reset accumulators
                            current_line_text = []
                            current_line_vertices = []
        
        return processed_lines

    async def detect_text(
        self,
        file: Any,
        source_lang: str = 'auto',
        saved_file_path: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Detect text in an image using Google Cloud Vision API."""
        try:
            if not self._initialized and not self._reinitialize_client():
                raise RuntimeError("OCR service is not properly initialized")
            
            logger.info(f"[OCR Service] Starting text detection at {datetime.now().strftime('%H:%M:%S.%f')}")
            
            # Determine base output directory
            if saved_file_path:
                base_dir = Path(saved_file_path).parent
            else:
                base_dir = Path.cwd()
            
            # Create output directory structure
            output_dir = base_dir / "output" / "ocr_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get image content and filename
            if saved_file_path:
                logger.info(f"[OCR Service] Reading from saved file: {saved_file_path}")
                with open(saved_file_path, 'rb') as image_file:
                    content = image_file.read()
                filename_stem = Path(saved_file_path).stem
            else:
                logger.info("[OCR Service] Reading from uploaded file")
                content = await file.read()
                filename_stem = f"uploaded_image_{timestamp}"
            
            # Create results directory for this run
            run_dir = output_dir / f"{filename_stem}_{timestamp}"
            run_dir.mkdir(exist_ok=True)
            
            # Save input image copy
            input_image_path = run_dir / "input_image.jpg"
            with open(input_image_path, 'wb') as f:
                f.write(content)
            
            # Preprocess image
            processed_content = self._preprocess_image(content)
            
            # Save preprocessed image
            preprocessed_image_path = run_dir / "preprocessed_image.jpg"
            with open(preprocessed_image_path, 'wb') as f:
                f.write(processed_content)
            
            # Create image object
            image = vision.Image(content=processed_content)
            
            # Perform text detection
            logger.info("[OCR Service] Sending request to Google Vision API")
            response = self._client.text_detection(image=image)
            
            # Process text blocks using the new method
            processed_lines = self._process_text_blocks(response)
            
            # Log the full text first
            if response.text_annotations:
                logger.info("\n=== OCR Full Text Overview ===")
                logger.info(response.text_annotations[0].description)
                logger.info("=" * 50)
            
            # Log processed lines
            if processed_lines:
                logger.info("\n=== OCR Detected Lines ===")
                for i, line in enumerate(processed_lines, 1):
                    logger.info(f"Line {i}: {line['text']}")
                    if logger.getEffectiveLevel() <= logging.DEBUG:
                        logger.debug(f"  Block Type: {line['block_type']}")
                        logger.debug(f"  Bounding Box: {line['bounding_box']}")
            
            # Cache results
            self._ocr_cache[self._get_image_hash(processed_content)] = processed_lines
            
            logger.info(f"\n[OCR Service] Successfully processed {len(processed_lines)} lines from {len(processed_lines)} blocks")
            
            return processed_lines, None
            
        except Exception as e:
            logger.error(f"Error in text detection: {str(e)}", exc_info=True)
            return [], str(e)

# Create singleton instance
ocr_service = OCRService() 