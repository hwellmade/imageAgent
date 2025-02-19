from typing import List, Dict, Any, Tuple
import asyncio
from pathlib import Path
import json
from datetime import datetime
import logging
import re
from statistics import mean, stdev
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from .services.translation_service import translation_service
from .services.ocr_service import ocr_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextBlock:
    text: str
    bbox: List[List[float]]
    center: Tuple[float, float]
    area: float
    width: float
    height: float
    density: float  # Characters per unit area
    line_height: float  # Estimated line height

class TranslationTest:
    def __init__(self):
        self.results: Dict[str, Any] = {}
        
    def _split_into_sentences(self, texts: List[Dict[str, Any]]) -> List[str]:
        """Split texts into sentences based on block and paragraph structure."""
        sentences = []
        
        # Sort blocks by vertical position
        sorted_texts = sorted(texts, key=lambda t: (
            sum(p[1] for p in t['bounding_box']) / len(t['bounding_box'])  # y-coordinate
        ))
        
        current_sentence = ""
        
        for block in sorted_texts:
            block_text = block['text'].strip()
            block_type = block.get('block_type', 'TEXT')
            
            # Skip non-text blocks
            if block_type != 'TEXT':
                continue
            
            # Split block text into potential sentences
            parts = re.split(r'([。．.！!？?])', block_text)
            
            for i in range(0, len(parts), 2):
                part = parts[i].strip()
                delimiter = parts[i + 1] if i + 1 < len(parts) else ""
                
                if not part and not current_sentence:
                    continue
                
                # Handle special cases
                is_product_info = bool(re.search(r'[※][0-9０-９]', part))
                is_list_item = bool(re.match(r'[●・]', part))
                
                if current_sentence:
                    if is_product_info or is_list_item:
                        # Start new sentence for product info or list items
                        if current_sentence.strip():
                            sentences.append(current_sentence.strip())
                        current_sentence = part + delimiter
                    else:
                        # Continue current sentence
                        current_sentence += " " + part + delimiter
                else:
                    current_sentence = part + delimiter
                
                # End sentence if we have a delimiter
                if delimiter and not is_product_info:
                    if current_sentence.strip():
                        sentences.append(current_sentence.strip())
                    current_sentence = ""
        
        # Add any remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return [s for s in sentences if len(s.strip()) > 1]
    
    def _calculate_text_block_features(self, text_dict: Dict[str, Any]) -> TextBlock:
        """Calculate comprehensive features for a text block."""
        bbox = text_dict['bounding_box']
        
        # Calculate center
        center_x = sum(p[0] for p in bbox) / len(bbox)
        center_y = sum(p[1] for p in bbox) / len(bbox)
        
        # Calculate dimensions
        width = max(p[0] for p in bbox) - min(p[0] for p in bbox)
        height = max(p[1] for p in bbox) - min(p[1] for p in bbox)
        area = width * height
        
        # Calculate text density (characters per unit area)
        text_length = len(text_dict['text'])
        density = text_length / area if area > 0 else 0
        
        # Estimate line height (assuming relatively horizontal text)
        line_height = height / max(1, text_length // 20)  # Rough estimate of chars per line
        
        return TextBlock(
            text=text_dict['text'],
            bbox=bbox,
            center=(center_x, center_y),
            area=area,
            width=width,
            height=height,
            density=density,
            line_height=line_height
        )
    
    def _group_into_paragraphs(self, texts: List[Dict[str, Any]]) -> List[str]:
        """Group texts into paragraphs using block and paragraph structure."""
        if not texts:
            return []
        
        # Sort blocks by vertical position
        sorted_blocks = sorted(texts, key=lambda t: (
            sum(p[1] for p in t['bounding_box']) / len(t['bounding_box'])  # y-coordinate
        ))
        
        paragraphs = []
        current_paragraph = []
        
        for block in sorted_blocks:
            block_type = block.get('block_type', 'TEXT')
            
            # Skip non-text blocks
            if block_type != 'TEXT':
                continue
            
            block_text = block['text'].strip()
            
            # Check if block is a continuation of current paragraph
            if current_paragraph:
                prev_block = current_paragraph[-1]
                
                # Calculate vertical gap
                prev_center_y = sum(p[1] for p in prev_block['bounding_box']) / len(prev_block['bounding_box'])
                curr_center_y = sum(p[1] for p in block['bounding_box']) / len(block['bounding_box'])
                vertical_gap = curr_center_y - prev_center_y
                
                # Check for paragraph break conditions
                is_new_paragraph = (
                    vertical_gap > block['height'] * 1.5 or  # Large vertical gap
                    bool(re.match(r'[●・]', block_text)) or  # List item
                    bool(re.search(r'[※][0-9０-９]', block_text)) or  # Product info
                    block.get('paragraph_count', 1) > 1  # Multiple paragraphs in block
                )
                
                if is_new_paragraph:
                    # Process current paragraph
                    if current_paragraph:
                        paragraph_text = ' '.join(b['text'].strip() for b in current_paragraph)
                        if paragraph_text.strip():
                            paragraphs.append(paragraph_text)
                    current_paragraph = [block]
                else:
                    current_paragraph.append(block)
            else:
                current_paragraph.append(block)
        
        # Add final paragraph
        if current_paragraph:
            paragraph_text = ' '.join(b['text'].strip() for b in current_paragraph)
            if paragraph_text.strip():
                paragraphs.append(paragraph_text)
        
        return paragraphs
    
    async def test_translation_strategies(self, image_path: str, target_lang: str = 'en'):
        """Test different translation strategies on an image."""
        try:
            logger.info(f"Starting translation test for image: {image_path}")
            start_time = datetime.now()
            
            # Perform OCR
            texts, _ = await ocr_service.detect_text(
                file=None,
                source_lang='ja',  # Explicitly set to Japanese
                saved_file_path=str(Path(image_path).absolute())
            )
            
            if not texts:
                logger.warning("No text detected in image")
                return
            
            # Store original texts
            self.results['original_texts'] = texts
            
            # Strategy 1: Full Content Translation
            logger.info("\nStrategy 1: Full Content Translation")
            full_text = ' '.join(text['text'] for text in texts)
            full_translation = await translation_service.translate_text(
                texts=[full_text],
                target_lang=target_lang,
                source_lang='ja'  # Explicitly set to Japanese
            )
            self.results['full_content_translation'] = full_translation
            logger.info(f"Original Full Text: {full_text}")
            logger.info(f"Translated Full Text: {full_translation[0]['translated_text']}")
            logger.info("-" * 50)
            
            # Strategy 2: Sentence-based Translation
            logger.info("\nStrategy 2: Sentence-based Translation")
            sentences = self._split_into_sentences(texts)
            sentence_translations = await translation_service.translate_text(
                texts=sentences,
                target_lang=target_lang,
                source_lang='ja'  # Explicitly set to Japanese
            )
            self.results['sentence_translations'] = sentence_translations
            for orig, trans in zip(sentences, sentence_translations):
                logger.info(f"Original Sentence: {orig}")
                logger.info(f"Translated: {trans['translated_text']}")
                logger.info("-" * 50)
            
            # Strategy 3: Paragraph-based Translation
            logger.info("\nStrategy 3: Paragraph-based Translation")
            paragraphs = self._group_into_paragraphs(texts)
            paragraph_translations = await translation_service.translate_text(
                texts=paragraphs,
                target_lang=target_lang,
                source_lang='ja'  # Explicitly set to Japanese
            )
            self.results['paragraph_translations'] = paragraph_translations
            for orig, trans in zip(paragraphs, paragraph_translations):
                logger.info(f"Original Paragraph: {orig}")
                logger.info(f"Translated: {trans['translated_text']}")
                logger.info("-" * 50)
            
            # Compare translation statistics
            logger.info("\n=== Translation Statistics ===")
            logger.info(f"Full Content: 1 API call")
            logger.info(f"Sentences: {len(sentences)} segments in 1 API call")
            logger.info(f"Paragraphs: {len(paragraphs)} segments in 1 API call")
            
            # Save results to file
            output_file = Path(image_path).parent / f"translation_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"\nTest completed in {(datetime.now() - start_time).total_seconds():.2f}s")
            logger.info(f"Results saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error during translation test: {str(e)}")
            raise

async def main():
    """Run translation test with sample image."""
    if len(sys.argv) != 2:
        print("Usage: python test_translation_strategies.py <image_path>")
        return
        
    image_path = sys.argv[1]
    tester = TranslationTest()
    await tester.test_translation_strategies(image_path)

if __name__ == "__main__":
    import sys
    asyncio.run(main()) 