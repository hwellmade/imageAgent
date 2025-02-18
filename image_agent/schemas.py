from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class ImageSource(str, Enum):
    """Source of the image."""
    CAMERA = "camera"
    UPLOAD = "upload"

class UploadResponse(BaseModel):
    """Response model for image upload."""
    success: bool
    filename: Optional[str] = None
    path: Optional[str] = None
    error: Optional[str] = None

class TextDetection(BaseModel):
    """Detected text in an image."""
    text: str
    confidence: float
    bounding_box: Optional[List[List[int]]] = None
    language: Optional[str] = None

class OCRResponse(BaseModel):
    """Response model for OCR processing."""
    success: bool
    texts: List[TextDetection] = []
    error: Optional[str] = None 