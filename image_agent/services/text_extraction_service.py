from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
import logging
from pathlib import Path
import json
import asyncio
from .ocr_service import ocr_service
from .vision_service import vision_service
from .overlay_service import overlay_service
import re
import unicodedata
import math

logger = logging.getLogger(__name__)

class TextExtractionService:
    """Service to coordinate OCR, LLM analysis, and result matching."""
    
    def __init__(self):
        self._initialized = True
        self._overlay_service = overlay_service
        logger.info("Text Extraction Service initialized")

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove non-printable characters and normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Convert to lowercase and remove punctuation
        text = ''.join(
            c.lower() for c in text
            if c.isalnum() or c.isspace()
        )
        
        return text.strip()

    def _calculate_text_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """Calculate detailed similarity metrics between two texts."""
        # Normalize texts
        t1 = self._normalize_text(text1)
        t2 = self._normalize_text(text2)
        
        # If either text is empty after normalization, return zero metrics
        if not t1 or not t2:
            return {
                "overall_similarity": 0.0,
                "char_similarity": 0.0,
                "word_similarity": 0.0,
                "length_similarity": 0.0
            }
        
        # Calculate character-based similarity
        char_similarity = SequenceMatcher(None, t1, t2).ratio()
        
        # Calculate word-based similarity
        words1 = set(t1.split())
        words2 = set(t2.split())
        if not words1 or not words2:
            word_similarity = 0.0
        else:
            common_words = words1.intersection(words2)
            word_similarity = 2 * len(common_words) / (len(words1) + len(words2))
        
        # Calculate length similarity
        length_similarity = min(len(t1), len(t2)) / max(len(t1), len(t2))
        
        # Calculate overall similarity with weighted components
        overall_similarity = (0.4 * char_similarity + 
                            0.4 * word_similarity + 
                            0.2 * length_similarity)
        
        return {
            "overall_similarity": overall_similarity,
            "char_similarity": char_similarity,
            "word_similarity": word_similarity,
            "length_similarity": length_similarity
        }

    def _match_results(
        self,
        ocr_results: List[Dict[str, Any]],
        llm_results: Dict[str, Any],
        similarity_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """Match OCR coordinates with LLM text analysis."""
        matched_results = {
            'original_language': llm_results['original_language'],
            'target_language': llm_results['target_language'],
            'metadata': {
                **llm_results['metadata'],
                'matching_metrics': {
                    'total_matches': 0,
                    'successful_matches': 0,
                    'average_similarity': 0.0,
                    'similarity_distribution': {
                        'high': 0,    # similarity > 0.8
                        'medium': 0,  # 0.5 <= similarity <= 0.8
                        'low': 0      # similarity < 0.5
                    }
                },
                'text_orientation': {
                    'global_orientation': None,
                    'text_direction_stats': {
                        'vertical_count': 0,
                        'horizontal_count': 0
                    }
                }
            },
            'paragraphs': []
        }
        
        total_matches = 0
        total_similarity = 0.0
        
        # Create a list of all OCR text blocks with their coordinates
        ocr_blocks = []
        for line in ocr_results:
            # Handle both individual blocks and grouped lines
            if 'original_blocks' in line:
                for block in line['original_blocks']:
                    ocr_blocks.append({
                        'text': block['text'],
                        'bounding_box': block['bounding_box']
                    })
            else:
                ocr_blocks.append({
                    'text': line['text'],
                    'bounding_box': line['bounding_box']
                })
        
        # Process each paragraph from LLM results
        for paragraph in llm_results['paragraphs']:
            matched_paragraph = {
                'id': paragraph['id'],
                'lines': [],
                'layout': {
                    'orientation': 'unknown',
                    'angle': 0,
                    'is_vertical': False,
                    'text_direction': '',
                    'statistics': {
                        'total_lines': 0,
                        'vertical_lines': 0,
                        'horizontal_lines': 0
                    }
                }
            }
            
            # Match each line in the paragraph
            for llm_line in paragraph['lines']:
                best_match = None
                best_similarity_metrics = None
                best_match_group = None
                
                llm_text = llm_line['original_text']
                
                # Try matching with individual OCR blocks
                for ocr_block in ocr_blocks:
                    similarity_metrics = self._calculate_text_similarity(
                        llm_text,
                        ocr_block['text']
                    )
                    
                    if (similarity_metrics['overall_similarity'] > similarity_threshold and
                        (best_similarity_metrics is None or 
                         similarity_metrics['overall_similarity'] > best_similarity_metrics['overall_similarity'])):
                        best_similarity_metrics = similarity_metrics
                        best_match = ocr_block
                        best_match_group = [ocr_block]
                
                # If no good match found, try matching with combined adjacent blocks
                if not best_match:
                    for i in range(len(ocr_blocks)):
                        combined_text = ocr_blocks[i]['text']
                        combined_blocks = [ocr_blocks[i]]
                        
                        for j in range(i + 1, min(i + 3, len(ocr_blocks))):
                            combined_text += ' ' + ocr_blocks[j]['text']
                            combined_blocks.append(ocr_blocks[j])
                            
                            similarity_metrics = self._calculate_text_similarity(
                                llm_text,
                                combined_text
                            )
                            
                            if (similarity_metrics['overall_similarity'] > similarity_threshold and
                                (best_similarity_metrics is None or 
                                 similarity_metrics['overall_similarity'] > best_similarity_metrics['overall_similarity'])):
                                best_similarity_metrics = similarity_metrics
                                best_match = {
                                    'text': combined_text,
                                    'bounding_box': self._combine_bounding_boxes([b['bounding_box'] for b in combined_blocks])
                                }
                                best_match_group = combined_blocks
                
                # Create matched line entry
                total_matches += 1
                
                if best_match:
                    # Calculate rotation for the matched block
                    rotation_info = self._calculate_rotation(best_match['bounding_box'])
                    
                    matched_line = {
                        'original_text': llm_line['original_text'],
                        'translated_text': llm_line['translated_text'],
                        'ocr_text': best_match['text'],
                        'coordinates': best_match['bounding_box'],
                        'similarity_metrics': best_similarity_metrics,
                        'match_details': {
                            'num_matched_blocks': len(best_match_group),
                            'matched_text_length': len(best_match['text'])
                        },
                        'layout': {
                            'orientation': rotation_info['orientation'] if rotation_info else 'unknown',
                            'angle': rotation_info['angle'] if rotation_info else 0,
                            'is_vertical': rotation_info['is_vertical'] if rotation_info else False,
                            'dimensions': {
                                'width': rotation_info['width'] if rotation_info else 0,
                                'height': rotation_info['height'] if rotation_info else 0
                            }
                        }
                    }
                    
                    # Update paragraph statistics
                    matched_paragraph['layout']['statistics']['total_lines'] += 1
                    if matched_line['layout']['is_vertical']:
                        matched_paragraph['layout']['statistics']['vertical_lines'] += 1
                        matched_results['metadata']['text_orientation']['text_direction_stats']['vertical_count'] += 1
                    else:
                        matched_paragraph['layout']['statistics']['horizontal_lines'] += 1
                        matched_results['metadata']['text_orientation']['text_direction_stats']['horizontal_count'] += 1
                    
                    # Update similarity statistics
                    similarity = best_similarity_metrics['overall_similarity']
                    total_similarity += similarity
                    
                    if similarity > 0.8:
                        matched_results['metadata']['matching_metrics']['similarity_distribution']['high'] += 1
                    elif similarity >= 0.5:
                        matched_results['metadata']['matching_metrics']['similarity_distribution']['medium'] += 1
                    else:
                        matched_results['metadata']['matching_metrics']['similarity_distribution']['low'] += 1
                        
                    matched_results['metadata']['matching_metrics']['successful_matches'] += 1
                    
                else:
                    matched_line = {
                        'original_text': llm_line['original_text'],
                        'translated_text': llm_line['translated_text'],
                        'ocr_text': None,
                        'coordinates': None,
                        'similarity_metrics': {
                            'overall_similarity': 0.0,
                            'char_similarity': 0.0,
                            'word_similarity': 0.0,
                            'length_similarity': 0.0
                        },
                        'match_details': {
                            'num_matched_blocks': 0,
                            'matched_text_length': 0
                        },
                        'layout': {
                            'orientation': 'unknown',
                            'angle': 0,
                            'is_vertical': False,
                            'dimensions': {
                                'width': 0,
                                'height': 0
                            }
                        }
                    }
                
                matched_paragraph['lines'].append(matched_line)
            
            # Determine paragraph orientation based on majority
            vertical_lines = matched_paragraph['layout']['statistics']['vertical_lines']
            horizontal_lines = matched_paragraph['layout']['statistics']['horizontal_lines']
            if vertical_lines + horizontal_lines > 0:
                matched_paragraph['layout']['orientation'] = 'vertical' if vertical_lines > horizontal_lines else 'horizontal'
                matched_paragraph['layout']['is_vertical'] = vertical_lines > horizontal_lines
                
                # Calculate average angle for the paragraph
                valid_angles = [line['layout']['angle'] for line in matched_paragraph['lines'] 
                              if line['layout']['orientation'] != 'unknown']
                if valid_angles:
                    matched_paragraph['layout']['angle'] = sum(valid_angles) / len(valid_angles)
            
            matched_results['paragraphs'].append(matched_paragraph)
        
        # Update final matching metrics
        matched_results['metadata']['matching_metrics']['total_matches'] = total_matches
        matched_results['metadata']['matching_metrics']['average_similarity'] = (
            total_similarity / total_matches if total_matches > 0 else 0.0
        )
        
        # Determine global orientation
        vertical_count = matched_results['metadata']['text_orientation']['text_direction_stats']['vertical_count']
        horizontal_count = matched_results['metadata']['text_orientation']['text_direction_stats']['horizontal_count']
        if vertical_count + horizontal_count > 0:
            matched_results['metadata']['text_orientation']['global_orientation'] = {
                'primary_direction': 'vertical' if vertical_count > horizontal_count else 'horizontal',
                'confidence': abs(vertical_count - horizontal_count) / (vertical_count + horizontal_count)
            }
        
        return matched_results

    def _should_group_lines(self, line1: Dict[str, Any], line2: Dict[str, Any]) -> bool:
        """Determine if two lines should be grouped together."""
        # Get bounding boxes
        box1 = line1['bounding_box']
        box2 = line2['bounding_box']
        
        # Calculate vertical distance between boxes
        box1_bottom = max(p[1] for p in box1)
        box2_top = min(p[1] for p in box2)
        vertical_distance = abs(box2_top - box1_bottom)
        
        # Calculate average line height
        box1_height = max(p[1] for p in box1) - min(p[1] for p in box1)
        box2_height = max(p[1] for p in box2) - min(p[1] for p in box2)
        avg_height = (box1_height + box2_height) / 2
        
        # Lines should be grouped if they're close enough vertically
        return vertical_distance <= avg_height * 0.5

    def _combine_bounding_boxes(self, boxes: List[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
        """Combine multiple bounding boxes into one encompassing box."""
        if not boxes:
            return []
            
        # Find min/max coordinates
        all_points = [p for box in boxes for p in box]
        min_x = min(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_x = max(p[0] for p in all_points)
        max_y = max(p[1] for p in all_points)
        
        # Return combined box coordinates
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

    def _calculate_rotation(self, vertices):
        if not vertices or len(vertices) < 4:
            return None
        
        # Calculate width and height using all vertices
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        # Calculate angle from first two vertices (top edge)
        dx = vertices[1][0] - vertices[0][0]
        dy = vertices[1][1] - vertices[0][1]
        angle = math.degrees(math.atan2(dy, dx))
        
        # Normalize angle to -90 to 90 range
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180
        
        # Determine if text is vertical based on dimensions and angle
        # For Japanese text, we consider it vertical if:
        # 1. Height is significantly larger than width (typical for vertical text)
        # 2. The angle is close to vertical (near 90 or -90 degrees)
        aspect_ratio = height / width if width > 0 else float('inf')
        is_vertical = (
            aspect_ratio > 1.5 or  # Height is significantly larger than width
            abs(abs(angle) - 90) < 15  # Angle is within 15 degrees of vertical
        )
        
        # For vertical text, adjust angle to be relative to vertical axis
        if is_vertical and abs(angle) < 45:
            angle = 90 - angle if angle >= 0 else -90 - angle
        
        return {
            'orientation': 'vertical' if is_vertical else 'horizontal',
            'angle': round(angle, 2),
            'is_vertical': is_vertical,
            'width': width,
            'height': height,
            'aspect_ratio': round(aspect_ratio, 2)
        }

    def _process_line(self, line):
        vertices = line.get('boundingBox', {}).get('vertices', [])
        rotation_info = self._calculate_rotation(vertices) if vertices else None
        
        # Get text direction from line layout
        text_direction = line.get('layout', {}).get('textDirection', '')
        
        return {
            'text': line.get('text', ''),
            'coordinates': vertices,
            'layout': {
                'orientation': rotation_info.get('orientation') if rotation_info else 'unknown',
                'angle': rotation_info.get('angle') if rotation_info else 0,
                'is_vertical': rotation_info.get('is_vertical') if rotation_info else False,
                'text_direction': text_direction,
                'dimensions': {
                    'width': rotation_info.get('width') if rotation_info else 0,
                    'height': rotation_info.get('height') if rotation_info else 0
                }
            }
        }

    def _process_paragraph(self, paragraph):
        processed_lines = []
        total_vertical = 0
        total_horizontal = 0
        
        # Process each line and collect rotation statistics
        for line in paragraph.get('lines', []):
            processed_line = self._process_line(line)
            processed_lines.append(processed_line)
            
            if processed_line['layout']['orientation'] == 'vertical':
                total_vertical += 1
            elif processed_line['layout']['orientation'] == 'horizontal':
                total_horizontal += 1
        
        # Determine paragraph orientation based on majority of lines
        is_vertical = total_vertical > total_horizontal
        
        # Calculate average angle from lines with valid rotation
        valid_angles = [line['layout']['angle'] for line in processed_lines 
                       if line['layout']['orientation'] != 'unknown']
        avg_angle = sum(valid_angles) / len(valid_angles) if valid_angles else 0
        
        return {
            'lines': processed_lines,
            'layout': {
                'orientation': 'vertical' if is_vertical else 'horizontal',
                'angle': round(avg_angle, 2),
                'is_vertical': is_vertical,
                'text_direction': paragraph.get('layout', {}).get('textDirection', ''),
                'statistics': {
                    'total_lines': len(processed_lines),
                    'vertical_lines': total_vertical,
                    'horizontal_lines': total_horizontal
                }
            }
        }

    def _combine_results(self, ocr_results, page_info=None):
        combined = {
            'original_language': self.source_lang,
            'target_language': self.target_lang,
            'metadata': {
                'page_info': page_info or {},
                'text_properties': [],
                'text_orientation': {
                    'global_orientation': None,
                    'text_direction_stats': {
                        'vertical_count': 0,
                        'horizontal_count': 0
                    }
                }
            },
            'paragraphs': []
        }
        
        # Process paragraphs with enhanced rotation information
        for paragraph in ocr_results.get('paragraphs', []):
            processed_paragraph = self._process_paragraph(paragraph)
            
            # Update text direction statistics
            if processed_paragraph['layout']['is_vertical']:
                combined['metadata']['text_orientation']['text_direction_stats']['vertical_count'] += 1
            else:
                combined['metadata']['text_orientation']['text_direction_stats']['horizontal_count'] += 1
            
            # Create enhanced paragraph entry with detailed layout info
            enhanced_paragraph = {
                'lines': [],
                'layout': {
                    'orientation': processed_paragraph['layout']['orientation'],
                    'angle': processed_paragraph['layout']['angle'],
                    'is_vertical': processed_paragraph['layout']['is_vertical'],
                    'text_direction': processed_paragraph['layout']['text_direction'],
                    'writing_style': {
                        'mode': 'vertical' if processed_paragraph['layout']['is_vertical'] else 'horizontal',
                        'text_flow': processed_paragraph['layout'].get('text_direction', 'UNKNOWN'),
                        'statistics': processed_paragraph['layout']['statistics']
                    }
                }
            }

            # Process each line with enhanced layout information
            for line in processed_paragraph['lines']:
                enhanced_line = {
                    'original_text': line.get('original_text', ''),
                    'translated_text': line.get('translated_text', ''),
                    'ocr_text': line.get('text', ''),
                    'coordinates': line.get('coordinates', line.get('bounding_box', [])),
                    'layout': {
                        'block': line.get('layout', {}).get('block', {}),
                        'paragraph': line.get('layout', {}).get('paragraph', {}),
                        'line': line.get('layout', {}).get('line', {}),
                        'writing_direction': {
                            'is_vertical': line.get('layout', {}).get('line', {}).get('is_vertical', False),
                            'text_flow': line.get('layout', {}).get('block', {}).get('text_direction', 'UNKNOWN'),
                            'dimensions': line.get('layout', {}).get('line', {}).get('dimensions', {})
                        }
                    }
                }
                enhanced_paragraph['lines'].append(enhanced_line)
            
            combined['paragraphs'].append(enhanced_paragraph)
        
        # Calculate global orientation
        vertical_count = combined['metadata']['text_orientation']['text_direction_stats']['vertical_count']
        horizontal_count = combined['metadata']['text_orientation']['text_direction_stats']['horizontal_count']
        
        if vertical_count + horizontal_count > 0:
            is_primarily_vertical = vertical_count > horizontal_count
            confidence = abs(vertical_count - horizontal_count) / (vertical_count + horizontal_count)
            
            combined['metadata']['text_orientation']['global_orientation'] = {
                'primary_direction': 'vertical' if is_primarily_vertical else 'horizontal',
                'confidence': confidence,
                'text_flow': 'UP_TO_DOWN' if is_primarily_vertical else 'LEFT_TO_RIGHT',
                'statistics': {
                    'vertical_blocks': vertical_count,
                    'horizontal_blocks': horizontal_count,
                    'total_blocks': vertical_count + horizontal_count
                }
            }
        
        return combined

    async def process_image(
        self,
        image_path: str,
        target_lang: str = "en",
        debug: bool = False
    ) -> Dict[str, Any]:
        """Process image through OCR and LLM, match results."""
        try:
            # 1. Run OCR and LLM in parallel
            logger.info("[Text Extraction] Starting parallel OCR and LLM processing...")
            ocr_task = ocr_service.detect_text(
                file=None,
                source_lang='ja',
                saved_file_path=str(Path(image_path).absolute())
            )
            llm_task = vision_service.analyze_image(
                image_path=image_path,
                target_lang_code=target_lang
            )
            
            # Wait for both tasks to complete
            ocr_results, llm_results = await asyncio.gather(
                ocr_task,
                llm_task,
                return_exceptions=True
            )
            
            # Unpack OCR results
            if isinstance(ocr_results, tuple):
                if len(ocr_results) == 3:
                    ocr_lines, page_info, raw_response = ocr_results
                else:
                    ocr_lines, page_info = ocr_results
                    raw_response = None
            else:
                ocr_lines, page_info, raw_response = ocr_results, None, None
            
            # Log results for debugging
            logger.info("\n=== OCR Results (Line-Grouped) ===")
            if ocr_lines:
                logger.info(f"Number of OCR lines: {len(ocr_lines)}")
                for i, line in enumerate(ocr_lines, 1):
                    logger.info(f"\nLine {i}:")
                    logger.info(f"Text: {line['text']}")
                    logger.info(f"Block Type: {line['block_type']}")
                    logger.info(f"Bounding Box: {line['bounding_box']}")
                    if 'original_blocks' in line:
                        logger.info("Original blocks in this line:")
                        for j, block in enumerate(line['original_blocks'], 1):
                            logger.info(f"  - Block {j}: {block['text']}")
                    logger.info("-" * 40)
            else:
                logger.error("No OCR results found")
            
            # Log LLM results
            logger.info("\n=== LLM Results ===")
            if isinstance(llm_results, Exception):
                logger.error(f"LLM Error: {str(llm_results)}")
            else:
                logger.info(f"LLM Response: {json.dumps(llm_results, ensure_ascii=False, indent=2)}")
            
            # Check for exceptions
            if isinstance(ocr_lines, tuple):
                ocr_lines, error = ocr_lines
                if error:
                    raise RuntimeError(f"OCR failed: {error}")
            if isinstance(llm_results, Exception):
                raise RuntimeError(f"LLM analysis failed: {str(llm_results)}")
            
            if not ocr_lines:
                raise ValueError("No text detected by OCR")
            if not llm_results or 'paragraphs' not in llm_results:
                raise ValueError("Invalid or empty response from LLM")
            
            # 2. Match results
            logger.info("[Text Extraction] Matching OCR and LLM results...")
            matched_results = self._match_results(ocr_lines, llm_results)
            
            # Add page orientation information from Google OCR
            if page_info:
                matched_results['metadata']['page_info'] = {
                    'width': page_info.get('width'),
                    'height': page_info.get('height'),
                    'orientation': {
                        'type': page_info.get('orientation', {}).get('type'),
                        'angle': page_info.get('orientation', {}).get('angle', 0),
                        'confidence': page_info.get('orientation', {}).get('confidence', 0.0)
                    } if page_info.get('orientation') else None,
                    'confidence': page_info.get('confidence'),
                    'detected_languages': page_info.get('language_codes', [])
                }
                
                # Add writing direction information to each paragraph based on OCR analysis
                for paragraph in matched_results['paragraphs']:
                    # Find all lines in this paragraph that have OCR matches
                    matched_lines = [line for line in paragraph['lines'] if line['ocr_text'] is not None]
                    if matched_lines:
                        # Get the most common writing style from matched lines
                        writing_styles = []
                        for line in matched_lines:
                            if 'writing_style' in line:
                                writing_styles.append(line['writing_style'])
                        
                        if writing_styles:
                            # Use the most common writing style for the paragraph
                            paragraph['writing_style'] = writing_styles[0]
                            
                            # Add writing direction and rotation to paragraph metadata
                            block_direction = writing_styles[0].get('block_direction', {})
                            rotation_info = writing_styles[0].get('rotation', {})
                            paragraph['text_properties'] = {
                                'direction': {
                                    'mode': block_direction.get('mode', 'horizontal'),
                                    'text_direction': block_direction.get('text_direction'),
                                    'confidence': page_info.get('confidence', 0.0)
                                },
                                'rotation': {
                                    'angle': rotation_info.get('angle', 0),
                                    'is_rotated': rotation_info.get('is_rotated', False),
                                    'rotation_confidence': rotation_info.get('rotation_confidence', 0.0)
                                },
                                'line_analysis': writing_styles[0].get('line_analysis', {}),
                                'text_properties': writing_styles[0].get('text_properties', {})
                            }
                            
                            # Add the same information to each line in the paragraph
                            for line in paragraph['lines']:
                                if line.get('ocr_text') is not None:
                                    line['text_properties'] = paragraph['text_properties'].copy()
                                    # If line has its own writing style, use that for more accurate per-line information
                                    if 'writing_style' in line:
                                        line_block_direction = line['writing_style'].get('block_direction', {})
                                        line_rotation_info = line['writing_style'].get('rotation', {})
                                        line['text_properties'].update({
                                            'direction': {
                                                'mode': line_block_direction.get('mode', 'horizontal'),
                                                'text_direction': line_block_direction.get('text_direction'),
                                                'confidence': page_info.get('confidence', 0.0)
                                            },
                                            'rotation': {
                                                'angle': line_rotation_info.get('angle', 0),
                                                'is_rotated': line_rotation_info.get('is_rotated', False),
                                                'rotation_confidence': line_rotation_info.get('rotation_confidence', 0.0)
                                            },
                                            'line_analysis': line['writing_style'].get('line_analysis', {}),
                                            'text_properties': line['writing_style'].get('text_properties', {})
                                        })
            
            # Remove any LLM-based orientation information
            if 'image_orientation' in matched_results['metadata']:
                del matched_results['metadata']['image_orientation']
            
            # Remove orientation from paragraphs if it exists (from LLM)
            for paragraph in matched_results['paragraphs']:
                if 'orientation' in paragraph:
                    del paragraph['orientation']
            
            # Log matched results
            logger.info("\n=== Matched Results ===")
            logger.info("Matching OCR lines with LLM paragraphs:")
            for para_idx, para in enumerate(matched_results['paragraphs'], 1):
                logger.info(f"\nParagraph {para_idx}:")
                for line_idx, line in enumerate(para['lines'], 1):
                    logger.info(f"Line {line_idx}:")
                    logger.info(f"  Original: {line['original_text']}")
                    logger.info(f"  Translated: {line['translated_text']}")
                    logger.info(f"  OCR Text: {line['ocr_text']}")
                    if line['similarity_metrics']:
                        logger.info(f"  Match Similarity: {line['similarity_metrics']['overall_similarity']:.2%}")
                    logger.info("-" * 30)
            
            # 3. Generate overlay images
            logger.info("[Text Extraction] Generating overlay images...")
            output_dir = Path(image_path).parent / "output"
            output_dir.mkdir(exist_ok=True)
            
            original_overlay_path = output_dir / f"{Path(image_path).stem}_original.jpg"
            translated_overlay_path = output_dir / f"{Path(image_path).stem}_translated.jpg"
            
            # Generate overlays
            await self._overlay_service.draw_text_overlay(
                image_path=image_path,
                text_blocks=[
                    line for para in matched_results['paragraphs']
                    for line in para['lines']
                    if line['coordinates'] is not None
                ],
                output_path=str(original_overlay_path),
                use_translated_text=False
            )
            
            await self._overlay_service.draw_text_overlay(
                image_path=image_path,
                text_blocks=[
                    line for para in matched_results['paragraphs']
                    for line in para['lines']
                    if line['coordinates'] is not None
                ],
                output_path=str(translated_overlay_path),
                use_translated_text=True
            )
            
            # 4. Save debug information if requested
            if debug:
                debug_info = {
                    "ocr_results": {
                        "line_grouped": ocr_lines,
                        "raw_response": {
                            "full_text": raw_response.text_annotations[0].description if raw_response and raw_response.text_annotations else None,
                            "text_annotations": [
                                {
                                    "description": annotation.description,
                                    "bounding_poly": [[vertex.x, vertex.y] for vertex in annotation.bounding_poly.vertices] if annotation.bounding_poly else None,
                                    "confidence": annotation.confidence if hasattr(annotation, 'confidence') else None
                                }
                                for annotation in raw_response.text_annotations[1:] if raw_response and raw_response.text_annotations
                            ] if raw_response and raw_response.text_annotations else [],
                        },
                        "statistics": {
                            "total_lines": len(ocr_lines)
                        }
                    },
                    "llm_results": llm_results,
                    "matched_results": matched_results,
                    "matching_statistics": {
                        "total_matched_lines": sum(1 for para in matched_results['paragraphs'] for line in para['lines'] if line['coordinates'] is not None),
                        "average_match_similarity": sum(line['similarity_metrics']['overall_similarity'] for para in matched_results['paragraphs'] for line in para['lines']) / sum(len(para['lines']) for para in matched_results['paragraphs'])
                    }
                }
                debug_path = output_dir / f"{Path(image_path).stem}_debug.json"
                with open(debug_path, 'w', encoding='utf-8') as f:
                    json.dump(debug_info, f, ensure_ascii=False, indent=2)
                logger.info(f"[Text Extraction] Debug information saved to: {debug_path}")
            
            return matched_results
            
        except Exception as e:
            logger.error(f"[Text Extraction] Error in text extraction: {str(e)}", exc_info=True)
            raise

# Create singleton instance
text_extraction_service = TextExtractionService() 