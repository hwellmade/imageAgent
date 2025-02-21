import asyncio
import logging
from pathlib import Path
import json
from datetime import datetime
import shutil
from .services.image_translation_service import image_translation_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_image_translation(image_path: str, target_lang: str = "en"):
    """Test the end-to-end image translation service."""
    try:
        logger.info(f"Starting image translation test for image: {image_path}")
        start_time = datetime.now()
        
        # Ensure input image exists
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = image_path.parent / "output" / "translation_test" / f"{image_path.stem}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using output directory: {output_dir}")
        
        # Copy original image to output directory
        original_copy = output_dir / f"original_{image_path.name}"
        shutil.copy2(image_path, original_copy)
        logger.info(f"Copied original image to: {original_copy}")
        
        # Process image with debug info
        logger.info("Processing image with image translation service...")
        result = await image_translation_service.process_image(
            str(image_path),
            target_lang=target_lang,
            debug=True
        )
        
        # Save results to JSON for inspection
        results_path = output_dir / "translation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved translation results to: {results_path}")
        
        # Copy overlay images to output directory
        original_overlay = Path(result['original_overlay_path'])
        translated_overlay = Path(result['translated_overlay_path'])
        
        if not original_overlay.exists():
            raise FileNotFoundError(f"Original overlay not generated: {original_overlay}")
        if not translated_overlay.exists():
            raise FileNotFoundError(f"Translated overlay not generated: {translated_overlay}")
            
        # Copy overlays to output directory
        output_original_overlay = output_dir / f"overlay_original.jpg"
        output_translated_overlay = output_dir / f"overlay_translated.jpg"
        
        shutil.copy2(original_overlay, output_original_overlay)
        shutil.copy2(translated_overlay, output_translated_overlay)
        logger.info(f"Copied overlay images to output directory")
            
        # Print test summary
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info("\n=== Test Summary ===")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Source language: {result['detected_source_lang']}")
        logger.info(f"Target language: {result['target_language']}")
        logger.info(f"Total text blocks: {len(result['text_blocks'])}")
        logger.info("\nOutput files:")
        logger.info(f"- Original image: {original_copy}")
        logger.info(f"- Original overlay: {output_original_overlay}")
        logger.info(f"- Translated overlay: {output_translated_overlay}")
        logger.info(f"- Results JSON: {results_path}")
        if 'metadata' in result:
            logger.info(f"\nText Statistics:")
            logger.info(f"- Total paragraphs: {result['metadata'].get('total_paragraphs', 'N/A')}")
            logger.info(f"- Total lines: {result['metadata'].get('total_lines', 'N/A')}")
        logger.info("===================\n")
        
        # Basic validation
        assert result['success'], "Translation process did not report success"
        assert len(result['text_blocks']) > 0, "No text blocks detected"
        assert output_original_overlay.exists(), "Original overlay not copied"
        assert output_translated_overlay.exists(), "Translated overlay not copied"
        
        logger.info("✅ Test completed successfully!")
        return result
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise

def main():
    """Main entry point for testing."""
    # Test images - you can add more test cases here
    test_images = [
        "frontend/test_images/test2.jpg",  # Update this path to your test image
    ]
    
    for image_path in test_images:
        if not Path(image_path).exists():
            logger.warning(f"Test image not found: {image_path}")
            continue
            
        logger.info(f"\nTesting with image: {image_path}")
        try:
            asyncio.run(test_image_translation(image_path))
        except Exception as e:
            logger.error(f"Test failed for {image_path}: {str(e)}")

if __name__ == "__main__":
    main() 