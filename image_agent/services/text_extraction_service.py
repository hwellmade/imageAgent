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
                'orientation': paragraph.get('orientation', 'horizontal'),
                'lines': []
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
                    matched_line = {
                        'original_text': llm_line['original_text'],
                        'translated_text': llm_line['translated_text'],
                        'ocr_text': best_match['text'],
                        'coordinates': best_match['bounding_box'],
                        'similarity_metrics': best_similarity_metrics,
                        'match_details': {
                            'num_matched_blocks': len(best_match_group),
                            'matched_text_length': len(best_match['text'])
                        }
                    }
                    
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
                        }
                    }
                
                matched_paragraph['lines'].append(matched_line)
            
            matched_results['paragraphs'].append(matched_paragraph)
        
        # Update final matching metrics
        matched_results['metadata']['matching_metrics']['total_matches'] = total_matches
        matched_results['metadata']['matching_metrics']['average_similarity'] = (
            total_similarity / total_matches if total_matches > 0 else 0.0
        )
        
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
            
            # Log results for debugging
            logger.info("\n=== OCR Results (Line-Grouped) ===")
            if isinstance(ocr_results, tuple):
                ocr_results, error = ocr_results
                if error:
                    logger.error(f"OCR Error: {error}")
            if ocr_results:
                logger.info(f"Number of OCR lines: {len(ocr_results)}")
                for i, line in enumerate(ocr_results, 1):
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
            if isinstance(ocr_results, tuple):
                ocr_results, error = ocr_results
                if error:
                    raise RuntimeError(f"OCR failed: {error}")
            if isinstance(llm_results, Exception):
                raise RuntimeError(f"LLM analysis failed: {str(llm_results)}")
            
            if not ocr_results:
                raise ValueError("No text detected by OCR")
            if not llm_results or 'paragraphs' not in llm_results:
                raise ValueError("Invalid or empty response from LLM")
            
            # 2. Match results
            logger.info("[Text Extraction] Matching OCR and LLM results...")
            matched_results = self._match_results(ocr_results, llm_results)
            
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
                        "line_grouped": ocr_results,
                        "statistics": {
                            "total_lines": len(ocr_results)
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