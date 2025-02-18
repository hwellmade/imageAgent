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
            
            # Batch translate all texts in a single API call
            results = self._client.translate(
                texts,
                target_language=target_lang,
                source_language=source_lang
            )
            
            # Process results
            translations = []
            for text, result in zip(texts, results):
                translations.append({
                    'original_text': text,
                    'translated_text': result['translatedText'],
                    'detected_source_language': result.get('detectedSourceLanguage', source_lang or 'unknown'),
                    'target_language': target_lang
                })
            
            # Log translation statistics
            char_count = sum(len(text) for text in texts)
            logger.info(f"Translated {len(texts)} texts ({char_count} characters) in one API call")
            
            return translations
            
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