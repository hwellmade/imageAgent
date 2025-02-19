from typing import List, Dict, Any, Optional as OptionalType
from google.cloud.translate_v2 import Client as TranslateClient
import os
import logging

logger = logging.getLogger(__name__)

class TranslationService:
    """Service for translating text using Google Cloud Translation."""
    
    def __init__(self):
        self._client = None
        self._project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        self._initialized = False
        try:
            if self._project_id:
                self._client = TranslateClient()
                self._initialized = True
                logger.info("Translation service initialized successfully")
            else:
                raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")
        except Exception as e:
            logger.error(f"Translation service initialization failed: {str(e)}")
            logger.error("Translation features will be disabled")
            raise RuntimeError(f"Failed to initialize translation service: {str(e)}")
        
    async def translate_text(self,
                           texts: List[str],
                           target_lang: str,
                           source_lang: OptionalType[str] = None) -> List[Dict[str, str]]:
        """
        Translate a list of texts to the target language.
        Handles API limitations by batching requests.
        
        Args:
            texts: List of texts to translate
            target_lang: Target language code
            source_lang: Source language code (optional, will be auto-detected if not provided)
            
        Returns:
            List of dictionaries containing original and translated texts
        
        Raises:
            RuntimeError: If translation service is not initialized or translation fails
        """
        if not self._initialized:
            raise RuntimeError("Translation service is not properly initialized")
            
        try:
            if not texts:
                return []
            
            # Constants
            BATCH_SIZE = 128  # Google Translation API limit
            
            # Process in batches
            all_translations = []
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                
                # Translate batch
                results = self._client.translate(
                    batch,
                    target_language=target_lang,
                    source_language=source_lang
                )
                
                # Process batch results
                for text, result in zip(batch, results):
                    all_translations.append({
                        'original_text': text,
                        'translated_text': result['translatedText'],
                        'detected_source_language': result.get('detectedSourceLanguage', source_lang or 'unknown'),
                        'target_language': target_lang
                    })
                
                # Log batch statistics
                char_count = sum(len(text) for text in batch)
                logger.info(f"Translated batch of {len(batch)} texts ({char_count} characters)")
            
            # Log overall statistics
            total_char_count = sum(len(text) for text in texts)
            logger.info(f"Completed translation of {len(texts)} texts ({total_char_count} characters) in {len(texts) // BATCH_SIZE + 1} API calls")
            
            return all_translations
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise RuntimeError(f"Translation failed: {str(e)}")
    
    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of a text.
        
        Args:
            text: Text to detect language for
            
        Returns:
            Dictionary containing detected language and confidence
        
        Raises:
            RuntimeError: If language detection fails
        """
        if not self._initialized:
            raise RuntimeError("Translation service is not properly initialized")
            
        try:
            result = self._client.detect_language(text)
            
            if isinstance(result, list):
                result = result[0]
                
            return {
                'language': result['language'],
                'confidence': result.get('confidence', 0.0)
            }
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            raise RuntimeError(f"Language detection failed: {str(e)}")

# Create a singleton instance
translation_service = TranslationService() 