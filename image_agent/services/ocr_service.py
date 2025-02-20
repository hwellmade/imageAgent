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

    def _calculate_line_angle(self, vertices: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Calculate the true angle of a text line from its collected vertices."""
        try:
            if not vertices:
                return {
                    'angle': 0,
                    'confidence': 0.0
                }

            # Fit a line to the points using least squares
            x_coords = [x for x, _ in vertices]
            y_coords = [y for _, y in vertices]
            
            if len(x_coords) < 2:  # Need at least 2 points
                return {
                    'angle': 0,
                    'confidence': 0.0
                }

            # Calculate the angle using linear regression
            x_mean = sum(x_coords) / len(x_coords)
            y_mean = sum(y_coords) / len(y_coords)
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_coords, y_coords))
            denominator = sum((x - x_mean) ** 2 for x in x_coords)
            
            if denominator == 0:  # Vertical line
                angle = 90
                confidence = 1.0
            else:
                slope = numerator / denominator
                angle = math.degrees(math.atan(slope))
                
                # Calculate confidence based on how well the points fit the line
                residuals = [(y - (slope * (x - x_mean) + y_mean)) ** 2 for x, y in zip(x_coords, y_coords)]
                mse = sum(residuals) / len(residuals)
                confidence = 1.0 / (1.0 + mse)  # Convert MSE to confidence score between 0 and 1
            
            # Normalize angle to -90 to 90 range
            while angle > 90:
                angle -= 180
            while angle < -90:
                angle += 180
                
            return {
                'angle': round(angle, 2),
                'confidence': round(confidence, 2)
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate line angle: {str(e)}")
            return {
                'angle': 0,
                'confidence': 0.0
            }

    def _process_text_blocks(self, response: vision.TextAnnotation) -> List[Dict[str, Any]]:
        """Process text blocks from Google Vision response to get line-level information."""
        if not response.full_text_annotation:
            return []
            
        processed_lines = []
        current_line_text = []
        current_line_vertices = []
        page_info = {}
        
        # Process each page
        for page in response.full_text_annotation.pages:
            # Get page-level orientation with more detailed logging
            orientation_value = None
            
            # Try multiple possible locations for orientation information
            if hasattr(page, 'property') and hasattr(page.property, 'detectedOrientation'):
                # First check property.detectedOrientation
                orientation_value = {
                    'type': 'DETECTED',
                    'angle': page.property.detectedOrientation.angle if hasattr(page.property.detectedOrientation, 'angle') else 0,
                    'confidence': page.property.detectedOrientation.confidence if hasattr(page.property.detectedOrientation, 'confidence') else 0.0
                }
                logger.info(f"[OCR Service] Found orientation in page.property.detectedOrientation")
            elif hasattr(page, 'layout') and hasattr(page.layout, 'orientation'):
                # Then check layout.orientation
                try:
                    orientation_type = vision.TextAnnotation.Page.Layout.Orientation.Name(page.layout.orientation)
                    orientation_value = {
                        'type': orientation_type,
                        'angle': page.layout.orientation.angle if hasattr(page.layout.orientation, 'angle') else 0,
                        'confidence': page.layout.confidence if hasattr(page.layout, 'confidence') else 0.0
                    }
                    logger.info(f"[OCR Service] Found orientation in page.layout.orientation")
                except ValueError as e:
                    logger.warning(f"Could not get orientation type from layout: {e}")
            
            # If no explicit orientation found, calculate from page dimensions
            if not orientation_value:
                # Use page dimensions to infer basic orientation
                orientation_value = {
                    'type': 'INFERRED',
                    'angle': 0,  # No rotation detected
                    'confidence': 0.8 if page.height > page.width * 1.2 else 0.0  # High confidence if clearly vertical
                }
                logger.info("[OCR Service] Using inferred orientation from page dimensions")
            
            logger.info(f"[OCR Service] Final orientation value: {orientation_value}")
            
            page_info = {
                'width': page.width,
                'height': page.height,
                'language_codes': [lang.language_code for lang in page.property.detected_languages] if hasattr(page, 'property') else [],
                'orientation': orientation_value,
                'confidence': page.confidence if hasattr(page, 'confidence') else None
            }
            
            # Process each block
            for block in page.blocks:
                # Initialize block layout with default values
                block_layout = {
                    'text_direction': 'UNKNOWN',
                    'orientation': {
                        'type': 'UNKNOWN',
                        'angle': 0
                    },
                    'writing_direction': {
                        'is_vertical': False,
                        'text_flow': 'UNKNOWN'
                    }
                }
                
                # Extract block layout information if available
                if hasattr(block, 'layout'):
                    layout = block.layout
                    
                    # Get text direction
                    if hasattr(layout, 'text_direction'):
                        try:
                            text_direction = vision.TextAnnotation.DetectedBreak.TextDirection.Name(layout.text_direction)
                            block_layout['text_direction'] = text_direction
                            block_layout['writing_direction']['text_flow'] = text_direction
                        except ValueError:
                            block_layout['text_direction'] = str(layout.text_direction)
                            block_layout['writing_direction']['text_flow'] = str(layout.text_direction)
                    
                    # Get orientation
                    if hasattr(layout, 'orientation'):
                        try:
                            orientation = vision.TextAnnotation.Page.Layout.Orientation.Name(layout.orientation)
                            block_layout['orientation'] = {
                                'type': orientation,
                                'angle': layout.orientation.angle if hasattr(layout.orientation, 'angle') else 0
                            }
                        except ValueError:
                            block_layout['orientation'] = {
                                'type': str(layout.orientation),
                                'angle': 0
                            }
                
                # Process each paragraph
                for paragraph in block.paragraphs:
                    # Initialize paragraph layout with default values
                    para_layout = {
                        'text_direction': block_layout['text_direction'],
                        'orientation': block_layout['orientation'].copy(),
                        'writing_direction': block_layout['writing_direction'].copy()
                    }
                    
                    # Update with paragraph-specific layout if available
                    if hasattr(paragraph, 'layout'):
                        layout = paragraph.layout
                        if hasattr(layout, 'text_direction'):
                            try:
                                text_direction = vision.TextAnnotation.DetectedBreak.TextDirection.Name(layout.text_direction)
                                para_layout['text_direction'] = text_direction
                                para_layout['writing_direction']['text_flow'] = text_direction
                            except ValueError:
                                para_layout['text_direction'] = str(layout.text_direction)
                                para_layout['writing_direction']['text_flow'] = str(layout.text_direction)
                    
                    # Process each word
                    for word in paragraph.words:
                        word_text = ''.join(symbol.text for symbol in word.symbols)
                        word_vertices = [(vertex.x, vertex.y) for vertex in word.bounding_box.vertices]
                        
                        # Calculate word dimensions
                        x_coords = [v[0] for v in word_vertices]
                        y_coords = [v[1] for v in word_vertices]
                        width = max(x_coords) - min(x_coords)
                        height = max(y_coords) - min(y_coords)
                        
                        # Determine if text is vertical
                        is_vertical = height > width * 1.2
                        
                        # Update block and paragraph layout writing direction
                        if is_vertical:
                            block_layout['writing_direction']['is_vertical'] = True
                            para_layout['writing_direction']['is_vertical'] = True
                        
                        current_line_text.append(word_text)
                        current_line_vertices.extend(word_vertices)
                        
                        # Check for line break
                        last_symbol = word.symbols[-1]
                        is_line_break = (hasattr(last_symbol, 'property') and 
                                       hasattr(last_symbol.property, 'detected_break') and
                                       last_symbol.property.detected_break.type in [
                                           vision.TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE,
                                           vision.TextAnnotation.DetectedBreak.BreakType.LINE_BREAK
                                       ])
                        
                        if is_line_break or (word == paragraph.words[-1] and current_line_text):
                            # Calculate line bounding box
                            min_x = min(x for x, _ in current_line_vertices)
                            min_y = min(y for _, y in current_line_vertices)
                            max_x = max(x for x, _ in current_line_vertices)
                            max_y = max(y for _, y in current_line_vertices)
                            
                            line_box = [
                                (min_x, min_y),
                                (max_x, min_y),
                                (max_x, max_y),
                                (min_x, max_y)
                            ]
                            
                            # Calculate line dimensions and orientation
                            line_width = max_x - min_x
                            line_height = max_y - min_y
                            line_is_vertical = line_height > line_width * 1.2
                            
                            # Calculate true line angle from collected vertices
                            angle_info = self._calculate_line_angle(current_line_vertices)
                            
                            # Create line entry with enhanced layout information
                            line = {
                                'text': ' '.join(current_line_text),
                                'original_text': ' '.join(current_line_text),
                                'bounding_box': line_box,
                                'coordinates': line_box,
                                'block_type': 'LINE',
                                'layout': {
                                    'block': block_layout,
                                    'paragraph': para_layout,
                                    'line': {
                                        'dimensions': {
                                            'width': line_width,
                                            'height': line_height
                                        },
                                        'is_vertical': line_is_vertical,
                                        'aspect_ratio': line_height / line_width if line_width > 0 else float('inf'),
                                        'angle': angle_info['angle'],
                                        'angle_confidence': angle_info['confidence']
                                    }
                                }
                            }
                            
                            processed_lines.append(line)
                            current_line_text = []
                            current_line_vertices = []
        
        return processed_lines, page_info

    async def detect_text(
        self,
        file: Any,
        source_lang: str = 'auto',
        saved_file_path: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[vision.TextAnnotation]]:
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
            
            # Save raw OCR results
            raw_ocr_path = run_dir / "raw_ocr_results.json"
            raw_ocr_data = {
                "full_text": response.text_annotations[0].description if response.text_annotations else None,
                "text_annotations": [
                    {
                        "description": annotation.description,
                        "bounding_poly": [[vertex.x, vertex.y] for vertex in annotation.bounding_poly.vertices] if annotation.bounding_poly else None,
                        "confidence": annotation.confidence if hasattr(annotation, 'confidence') else None,
                        "locale": annotation.locale if hasattr(annotation, 'locale') else None
                    }
                    for annotation in response.text_annotations[1:] if response.text_annotations
                ] if response.text_annotations else [],
                "pages": [
                    {
                        "width": page.width,
                        "height": page.height,
                        "blocks": [
                            {
                                "text": "".join(symbol.text for word in paragraph.words for symbol in word.symbols),
                                "bounding_box": [[vertex.x, vertex.y] for vertex in block.bounding_box.vertices] if block.bounding_box else None,
                                "confidence": block.confidence if hasattr(block, 'confidence') else None,
                                "orientation": str(block.layout.orientation) if hasattr(block, 'layout') and hasattr(block.layout, 'orientation') else None,
                                "text_direction": str(block.layout.text_direction) if hasattr(block, 'layout') and hasattr(block.layout, 'text_direction') else None
                            }
                            for block in page.blocks
                            for paragraph in block.paragraphs
                        ]
                    }
                    for page in response.full_text_annotation.pages
                ] if response.full_text_annotation else []
            }
            with open(raw_ocr_path, 'w', encoding='utf-8') as f:
                json.dump(raw_ocr_data, f, ensure_ascii=False, indent=2)
            logger.info(f"[OCR Service] Saved raw OCR results to: {raw_ocr_path}")
            
            # Process text blocks using the new method
            processed_lines, page_info = self._process_text_blocks(response)
            
            # Log the full text first
            if response.text_annotations:
                logger.info("\n=== OCR Full Text Overview ===")
                logger.info(response.text_annotations[0].description)
                logger.info("=" * 50)
            
            # Log processed lines and page info
            if processed_lines:
                logger.info("\n=== OCR Detected Lines ===")
                for i, line in enumerate(processed_lines, 1):
                    logger.info(f"Line {i}: {line['text']}")
                    if logger.getEffectiveLevel() <= logging.DEBUG:
                        logger.debug(f"  Block Type: {line['block_type']}")
                        logger.debug(f"  Bounding Box: {line['bounding_box']}")
            
            logger.info("\n=== Page Information ===")
            logger.info(f"Page Info: {page_info}")
            
            # Cache results with page info
            self._ocr_cache[self._get_image_hash(processed_content)] = (processed_lines, page_info)
            
            logger.info(f"\n[OCR Service] Successfully processed {len(processed_lines)} lines")
            
            return processed_lines, page_info, response
            
        except Exception as e:
            logger.error(f"Error in text detection: {str(e)}", exc_info=True)
            return [], None, None

# Create singleton instance
ocr_service = OCRService() 