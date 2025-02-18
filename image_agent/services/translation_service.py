from typing import List, Dict, Any, Optional as OptionalType
from google.cloud import translate
import os

class TranslationService:
    """Service for translating text using Google Cloud Translation."""
    
    def __init__(self):
        self._client = None
        self._project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        self._initialized = False
        try:
            if self._project_id:
                self._client = translate.TranslationServiceClient()
                self._location = "global"
                self._initialized = True
        except Exception as e:
            print(f"Warning: Translation service not initialized: {str(e)}")
            print("Translation features will be disabled.")
        
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
        """
        if not self._initialized:
            # Return original texts with a note that translation is unavailable
            return [{
                'original_text': text,
                'translated_text': text,
                'detected_source_language': 'unknown',
                'target_language': target_lang,
                'note': 'Translation service not available'
            } for text in texts]
            
        try:
            if not texts:
                return []
            
            parent = f"projects/{self._project_id}/locations/{self._location}"
            
            response = self._client.translate_text(
                request={
                    "parent": parent,
                    "contents": texts,
                    "target_language_code": target_lang,
                    "source_language_code": source_lang if source_lang else "",
                    "mime_type": "text/plain",
                }
            )
            
            # Format results
            translations = []
            for translation in response.translations:
                translations.append({
                    'original_text': texts[len(translations)],  # Match with original text
                    'translated_text': translation.translated_text,
                    'detected_source_language': translation.detected_language_code,
                    'target_language': target_lang
                })
            
            return translations
            
        except Exception as e:
            # On error, return original texts
            return [{
                'original_text': text,
                'translated_text': text,
                'detected_source_language': 'unknown',
                'target_language': target_lang,
                'error': str(e)
            } for text in texts]
    
    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of a text.
        
        Args:
            text: Text to detect language for
            
        Returns:
            Dictionary containing detected language and confidence
        """
        if not self._initialized:
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'note': 'Language detection service not available'
            }
            
        try:
            parent = f"projects/{self._project_id}/locations/{self._location}"
            
            response = self._client.detect_language(
                request={
                    "parent": parent,
                    "content": text,
                    "mime_type": "text/plain",
                }
            )
            
            detection = response.languages[0]
            return {
                'language': detection.language_code,
                'confidence': detection.confidence
            }
        except Exception as e:
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }

# Create a singleton instance
translation_service = TranslationService() 