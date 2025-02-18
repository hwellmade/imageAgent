from typing import List, Dict, Any
import asyncio
from pathlib import Path
import json
from datetime import datetime
import logging
from .services.translation_service import translation_service
from .services.ocr_service import ocr_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationTest:
    def __init__(self):
        self.results: Dict[str, Any] = {}
    
    def _group_by_proximity(self, texts: List[Dict[str, Any]], threshold: float = 50) -> List[List[Dict[str, Any]]]:
        """Group text blocks that are close to each other."""
        if not texts:
            return []
            
        groups = []
        current_group = [texts[0]]
        
        for text in texts[1:]:
            # Get the last text in current group
            last_text = current_group[-1]
            
            # Calculate distance between bounding boxes
            last_box = last_text['bounding_box']
            current_box = text['bounding_box']
            
            # Use the center points of boxes for distance calculation
            last_center_y = sum(point[1] for point in last_box) / len(last_box)
            current_center_y = sum(point[1] for point in current_box) / len(current_box)
            
            # If texts are close enough vertically, consider them part of the same group
            if abs(current_center_y - last_center_y) < threshold:
                current_group.append(text)
            else:
                groups.append(current_group)
                current_group = [text]
        
        groups.append(current_group)
        return groups
    
    def _group_by_lines(self, texts: List[Dict[str, Any]], line_height_threshold: float = 30) -> List[List[Dict[str, Any]]]:
        """Group text blocks that appear to be on the same line."""
        if not texts:
            return []
            
        # Sort texts by vertical position (top to bottom)
        sorted_texts = sorted(texts, key=lambda t: sum(point[1] for point in t['bounding_box']) / len(t['bounding_box']))
        
        lines = []
        current_line = [sorted_texts[0]]
        current_line_y = sum(point[1] for point in sorted_texts[0]['bounding_box']) / len(sorted_texts[0]['bounding_box'])
        
        for text in sorted_texts[1:]:
            text_y = sum(point[1] for point in text['bounding_box']) / len(text['bounding_box'])
            
            if abs(text_y - current_line_y) < line_height_threshold:
                current_line.append(text)
            else:
                # Sort texts in line by horizontal position
                current_line.sort(key=lambda t: sum(point[0] for point in t['bounding_box']) / len(t['bounding_box']))
                lines.append(current_line)
                current_line = [text]
                current_line_y = text_y
        
        # Add the last line
        if current_line:
            current_line.sort(key=lambda t: sum(point[0] for point in t['bounding_box']) / len(t['bounding_box']))
            lines.append(current_line)
        
        return lines
    
    async def test_translation_strategies(self, image_path: str, target_lang: str = 'en'):
        """Test different translation strategies on an image."""
        try:
            logger.info(f"Starting translation test for image: {image_path}")
            start_time = datetime.now()
            
            # Perform OCR
            with open(image_path, 'rb') as f:
                texts, _ = await ocr_service.detect_text(f)
            
            if not texts:
                logger.warning("No text detected in image")
                return
            
            # Store original texts
            self.results['original_texts'] = texts
            
            # Strategy 1: Individual Translation
            individual_translations = await translation_service.translate_text(
                texts=[text['text'] for text in texts],
                target_lang=target_lang
            )
            self.results['individual_translations'] = individual_translations
            
            # Strategy 2: Proximity-based Grouping
            proximity_groups = self._group_by_proximity(texts)
            grouped_texts = [' '.join(text['text'] for text in group) for group in proximity_groups]
            proximity_translations = await translation_service.translate_text(
                texts=grouped_texts,
                target_lang=target_lang
            )
            self.results['proximity_translations'] = proximity_translations
            
            # Strategy 3: Line-based Grouping
            line_groups = self._group_by_lines(texts)
            line_texts = [' '.join(text['text'] for text in line) for line in line_groups]
            line_translations = await translation_service.translate_text(
                texts=line_texts,
                target_lang=target_lang
            )
            self.results['line_translations'] = line_translations
            
            # Strategy 4: Full Context Translation
            full_text = ' '.join(text['text'] for text in texts)
            full_context_translation = await translation_service.translate_text(
                texts=[full_text],
                target_lang=target_lang
            )
            self.results['full_context_translation'] = full_context_translation
            
            # Log results
            logger.info("\n=== Translation Results ===")
            
            logger.info("\nStrategy 1: Individual Translation")
            for orig, trans in zip(texts, individual_translations):
                logger.info(f"Original: {orig['text']}")
                logger.info(f"Translated: {trans['translated_text']}")
                logger.info("-" * 50)
            
            logger.info("\nStrategy 2: Proximity-based Grouping")
            for group, trans in zip(proximity_groups, proximity_translations):
                logger.info(f"Original Group: {' '.join(text['text'] for text in group)}")
                logger.info(f"Translated: {trans['translated_text']}")
                logger.info("-" * 50)
            
            logger.info("\nStrategy 3: Line-based Grouping")
            for line, trans in zip(line_groups, line_translations):
                logger.info(f"Original Line: {' '.join(text['text'] for text in line)}")
                logger.info(f"Translated: {trans['translated_text']}")
                logger.info("-" * 50)
            
            logger.info("\nStrategy 4: Full Context Translation")
            logger.info(f"Original Full Text: {full_text}")
            logger.info(f"Translated Full Text: {full_context_translation[0]['translated_text']}")
            logger.info("-" * 50)
            
            # Compare translation lengths and token counts
            logger.info("\n=== Translation Statistics ===")
            logger.info(f"Individual Translations: {len(individual_translations)} segments")
            logger.info(f"Proximity Groups: {len(proximity_translations)} segments")
            logger.info(f"Line Groups: {len(line_translations)} segments")
            logger.info("Full Context: 1 segment")
            
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