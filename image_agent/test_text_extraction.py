import asyncio
import logging
from pathlib import Path
import json
from image_agent.services.text_extraction_service import text_extraction_service
from image_agent.services.ocr_service import ocr_service
from image_agent.services.vision_service import vision_service
from image_agent.services.overlay_service import overlay_service
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_text_extraction(image_path: str, target_lang: str = "en"):
    """Test the combined text extraction and translation process."""
    try:
        logger.info(f"Starting text extraction test for image: {image_path}")
        
        # Ensure input image exists
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        
        # Create output directory
        output_dir = image_path.parent / "output" / "hybrid"
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Using output directory: {output_dir}")
        
        # Process image with debug info
        logger.info("Processing image...")
        result = await text_extraction_service.process_image(
            str(image_path),
            target_lang=target_lang,
            debug=True
        )
        
        # Save results
        output_path = output_dir / f"{image_path.stem}_combined_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved combined results to: {output_path}")
        
        # Generate overlays
        logger.info("Generating overlay images...")
        
        # Create output paths
        original_overlay_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_original.jpg")
        translated_overlay_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_translated.jpg")
        
        # Extract text blocks from paragraphs
        text_blocks = []
        for paragraph in result['paragraphs']:
            for line in paragraph['lines']:
                if line['coordinates'] and line['ocr_text']:
                    text_blocks.append({
                        'text': line['original_text'],
                        'translated_text': line['translated_text'],
                        'bounding_box': line['coordinates']
                    })
        
        # Generate overlays using overlay_service
        await overlay_service.draw_text_overlay(
            image_path=str(image_path),
            text_blocks=text_blocks,
            output_path=original_overlay_path,
            use_translated_text=False
        )
        
        await overlay_service.draw_text_overlay(
            image_path=str(image_path),
            text_blocks=text_blocks,
            output_path=translated_overlay_path,
            use_translated_text=True
        )
        
        # Print summary
        logger.info("\n=== Test Summary ===")
        logger.info(f"Total paragraphs: {result['metadata']['total_paragraphs']}")
        logger.info(f"Total lines: {result['metadata']['total_lines']}")
        logger.info(f"Image orientation: {result['metadata']['image_orientation']}")
        logger.info(f"Original language: {result['original_language']}")
        logger.info(f"Target language: {result['target_language']}")
        logger.info("===================\n")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        raise

def main():
    """Main entry point for testing."""
    # Test images
    test_images = [
        "frontend/test_images/test.jpg"
    ]
    
    for image_path in test_images:
        if not Path(image_path).exists():
            logger.warning(f"Test image not found: {image_path}")
            continue
            
        logger.info(f"\nTesting with image: {image_path}")
        try:
            asyncio.run(test_text_extraction(image_path))
        except Exception as e:
            logger.error(f"Test failed for {image_path}: {str(e)}")

if __name__ == "__main__":
    main() 