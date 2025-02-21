import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
from .text_extraction_service import text_extraction_service
from .overlay_service import overlay_service

logger = logging.getLogger(__name__)

class ImageTranslationService:
    """
    End-to-end service for image translation that combines:
    1. Text extraction (OCR + LLM)
    2. Overlay generation
    """
    
    async def process_image(
        self,
        image_path: str,
        target_lang: str = "en",
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Process an image end-to-end:
        1. Extract and translate text using hybrid approach
        2. Generate overlay images for both original and translated text
        
        Args:
            image_path: Path to the input image
            target_lang: Target language for translation
            debug: Whether to include debug information
            
        Returns:
            Dict containing:
            - Extraction results (text, coordinates, etc.)
            - Paths to generated overlay images
            - Metadata about the process
        """
        try:
            logger.info(f"Starting end-to-end image translation for: {image_path}")
            
            # Step 1: Extract and translate text
            extraction_result = await text_extraction_service.process_image(
                image_path,
                target_lang=target_lang,
                debug=debug
            )
            
            # Step 2: Prepare output paths
            input_path = Path(image_path)
            output_dir = input_path.parent  # This will be the year/month directory where the original image is stored

            # Generate overlay filenames with same naming convention
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = input_path.stem.split('_')[-1]  # Get the UUID from original filename
            original_overlay_path = output_dir / f"overlay_original_{timestamp}_{unique_id}.jpg"
            translated_overlay_path = output_dir / f"overlay_translated_{timestamp}_{unique_id}.jpg"

            logger.info(f"Will generate overlays at: {output_dir}")
            logger.info(f"Original overlay: {original_overlay_path}")
            logger.info(f"Translated overlay: {translated_overlay_path}")
            
            # Step 3: Convert extraction results to overlay format
            text_blocks = []
            for paragraph in extraction_result['paragraphs']:
                for line in paragraph['lines']:
                    if line['coordinates'] and line.get('original_text'):
                        text_blocks.append({
                            'text': line['original_text'],
                            'translated_text': line['translated_text'],
                            'bounding_box': line['coordinates']
                        })
            
            # Step 4: Generate overlay images
            await overlay_service.draw_text_overlay(
                image_path=image_path,
                text_blocks=text_blocks,
                output_path=str(original_overlay_path),
                use_translated_text=False
            )
            
            await overlay_service.draw_text_overlay(
                image_path=image_path,
                text_blocks=text_blocks,
                output_path=str(translated_overlay_path),
                use_translated_text=True
            )
            
            # Step 5: Prepare response
            response = {
                "success": True,
                "detected_source_lang": extraction_result['original_language'],
                "target_language": extraction_result['target_language'],
                "metadata": extraction_result['metadata'],
                "original_overlay_path": str(original_overlay_path),
                "translated_overlay_path": str(translated_overlay_path),
                "text_blocks": text_blocks  # Include for potential future use
            }
            
            if debug:
                response["debug_info"] = extraction_result.get("debug_info", {})
            
            return response
            
        except Exception as e:
            logger.error(f"Error in end-to-end image translation: {str(e)}")
            raise

# Create singleton instance
image_translation_service = ImageTranslationService() 