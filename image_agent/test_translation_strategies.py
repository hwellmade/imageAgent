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

class TranslationTest:
    def __init__(self):
        self.results: Dict[str, Any] = {}
        
    def _split_into_sentences(self, texts: List[Dict[str, Any]]) -> List[str]:
        """Split texts into sentences based on punctuation and spacing."""
        # Sort texts by position (top to bottom, left to right)
        sorted_texts = sorted(texts, key=lambda t: (
            sum(p[1] for p in t['bounding_box']) / len(t['bounding_box']),
            sum(p[0] for p in t['bounding_box']) / len(t['bounding_box'])
        ))
        
        # Combine all text with their original spacing
        combined_text = ""
        for text in sorted_texts:
            combined_text += text['text'] + " "
            
        # Define sentence delimiters
        delimiters = r'[,.．。，：:;；()（）\[\]「」『』\'\""\s]+'
        
        # Split into sentences while preserving delimiters
        parts = re.split(f'({delimiters})', combined_text)
        
        # Recombine parts into proper sentences
        sentences = []
        current_sentence = ""
        
        for part in parts:
            current_sentence += part
            if re.search(r'[.．。!\?！？]', part):
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add any remaining text as a sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
            
        return sentences
        
    def _calculate_text_block_features(self, text_dict: Dict[str, Any]) -> TextBlock:
        """Calculate features for a text block."""
        bbox = text_dict['bounding_box']
        # Calculate center
        center_x = sum(p[0] for p in bbox) / len(bbox)
        center_y = sum(p[1] for p in bbox) / len(bbox)
        # Calculate area
        width = max(p[0] for p in bbox) - min(p[0] for p in bbox)
        height = max(p[1] for p in bbox) - min(p[1] for p in bbox)
        area = width * height
        
        return TextBlock(
            text=text_dict['text'],
            bbox=bbox,
            center=(center_x, center_y),
            area=area
        )
    
    def _group_into_paragraphs(self, texts: List[Dict[str, Any]]) -> List[str]:
        """Group texts into paragraphs using DBSCAN clustering with adaptive eps."""
        if not texts:
            return []
            
        # Convert texts to TextBlock objects with features
        text_blocks = [self._calculate_text_block_features(t) for t in texts]
        
        # Extract centers for clustering
        centers = np.array([block.center for block in text_blocks])
        
        # Calculate adaptive eps based on nearest neighbor distances
        from sklearn.neighbors import NearestNeighbors
        k = min(2, len(centers))  # Use k=2 or less if we have fewer points
        nbrs = NearestNeighbors(n_neighbors=k).fit(centers)
        distances, _ = nbrs.kneighbors(centers)
        
        # Use the mean distance to nearest neighbor as eps
        eps = np.mean(distances[:, 1]) if k > 1 else np.mean(distances)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=1).fit(centers)
        
        # Group texts by cluster
        paragraphs = []
        for cluster_id in range(max(clustering.labels_) + 1):
            # Get indices of texts in this cluster
            cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
            
            # Sort texts in cluster by vertical position
            cluster_texts = [text_blocks[i] for i in cluster_indices]
            sorted_texts = sorted(cluster_texts, 
                                key=lambda b: (b.center[1], b.center[0]))
            
            # Combine texts in cluster
            paragraph = ' '.join(block.text for block in sorted_texts)
            paragraphs.append(paragraph)
            
        return paragraphs
    
    async def test_translation_strategies(self, image_path: str, target_lang: str = 'en'):
        """Test different translation strategies on an image."""
        try:
            logger.info(f"Starting translation test for image: {image_path}")
            start_time = datetime.now()
            
            # Perform OCR
            texts, _ = await ocr_service.detect_text(
                file=None,
                source_lang='auto',
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
                target_lang=target_lang
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
                target_lang=target_lang
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
                target_lang=target_lang
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