from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import io
from PIL import Image
import uuid
from datetime import datetime
import os
import shutil
import asyncio
from typing import List, Optional
from fastapi.responses import JSONResponse
import logging
import aiofiles
from fastapi.staticfiles import StaticFiles
import sys
import json

from .services.ocr_service import ocr_service
from .services.translation_service import translation_service
from .schemas import UploadResponse, OCRResponse, TextDetection
from .services.vision_service import vision_service
from .services.image_translation_service import image_translation_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Output to stdout
    ]
)

# Force immediate flush of logging
for handler in logging.root.handlers:
    handler.flush()
    if isinstance(handler, logging.StreamHandler):
        handler.terminator = '\n'
        handler.flush()

logger = logging.getLogger(__name__)

# Base directory for all uploads
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
MAX_FILE_AGE_HOURS = 1  # Files older than this will be deleted

def ensure_upload_dirs():
    """Ensure all required upload directories exist and return the current month directory."""
    # Create main uploads directory
    UPLOAD_DIR.mkdir(exist_ok=True)
    
    # Create year/month subdirectories
    current_date = datetime.now()
    year_dir = UPLOAD_DIR / str(current_date.year)
    month_dir = year_dir / f"{current_date.month:02d}"
    temp_dir = UPLOAD_DIR / "temp"
    
    # Create all directories
    year_dir.mkdir(exist_ok=True)
    month_dir.mkdir(exist_ok=True)
    temp_dir.mkdir(exist_ok=True)
    
    logger.info(f"Upload directories created/verified at {UPLOAD_DIR}")
    return month_dir  # Return the month directory for file saving

# Create required directories
ensure_upload_dirs()

app = FastAPI(title="Image Agent API")

# Mount uploads directory for static file serving with name="uploads"
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR), check_dir=False), name="uploads")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def cleanup_old_files():
    """Delete files older than MAX_FILE_AGE_HOURS."""
    try:
        current_time = datetime.now()
        if not UPLOAD_DIR.exists():
            return
            
        for year_dir in UPLOAD_DIR.iterdir():
            if not year_dir.is_dir():
                continue
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue
                for file_path in month_dir.iterdir():
                    if not file_path.is_file():
                        continue
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age.total_seconds() > MAX_FILE_AGE_HOURS * 3600:
                        file_path.unlink()
                        
                # Clean up empty directories
                if not any(month_dir.iterdir()):
                    month_dir.rmdir()
            if not any(year_dir.iterdir()):
                year_dir.rmdir()
                
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Run cleanup on startup and schedule periodic cleanup."""
    await cleanup_old_files()
    
    async def periodic_cleanup():
        while True:
            await asyncio.sleep(3600)  # Run every hour
            await cleanup_old_files()
    
    asyncio.create_task(periodic_cleanup())

def generate_unique_filename(original_filename: str) -> str:
    """Generate a unique filename with timestamp and UUID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    extension = Path(original_filename).suffix
    return f"image_{timestamp}_{unique_id}{extension}"

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    """
    Handle image upload and save to organized directory structure.
    Files are automatically deleted after MAX_FILE_AGE_HOURS.
    Returns the saved file path and other metadata.
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create directory structure
        save_dir = ensure_upload_dirs()
        
        # Generate unique filename
        filename = generate_unique_filename(file.filename)
        file_path = save_dir / filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
            
        # Generate relative path for response
        relative_path = file_path.relative_to(BASE_DIR)
            
        return {
            "success": True,
            "filename": filename,
            "path": str(relative_path),
            "full_path": str(file_path),
            "size": len(content),
            "upload_time": datetime.now().isoformat(),
            "expiry_time": (datetime.now().timestamp() + MAX_FILE_AGE_HOURS * 3600) * 1000  # milliseconds
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")

@app.post("/api/ocr")
async def perform_ocr(
    file: UploadFile = File(...),
    target_lang: str = 'en'
):
    """
    Process image using hybrid OCR+LLM approach and return overlay images.
    """
    start_time = datetime.now()
    try:
        logger.info(f"[Translation] Starting analysis process at {start_time}")
        
        # Create upload directory for this month
        save_dir = ensure_upload_dirs()
        
        # Generate unique filename for the uploaded file
        filename = generate_unique_filename(file.filename)
        file_path = save_dir / filename
        
        # Save uploaded file
        save_start = datetime.now()
        content = await file.read()
        async with aiofiles.open(file_path, 'wb') as out_file:
            await out_file.write(content)
        logger.info(f"[Translation] File saved to: {file_path}")
        logger.info(f"[Translation] File save took: {(datetime.now() - save_start).total_seconds():.2f}s")
        
        try:
            # Process with hybrid approach
            process_start = datetime.now()
            result = await image_translation_service.process_image(
                str(file_path),
                target_lang=target_lang
            )
            logger.info(f"[Translation] Processing took: {(datetime.now() - process_start).total_seconds():.2f}s")
            
            if not result:
                logger.warning("[Translation] No text detected in image")
                return JSONResponse(
                    status_code=200,
                    content={
                        "message": "No text detected in image",
                        "original_image": None,
                        "overlay_image": None
                    }
                )
            
            # Convert paths to URLs
            logger.info(f"[Translation] Raw overlay paths from result: {result['original_overlay_path']}, {result['translated_overlay_path']}")
            
            original_overlay_url = f"/uploads/{Path(result['original_overlay_path']).relative_to(UPLOAD_DIR)}"
            translated_overlay_url = f"/uploads/{Path(result['translated_overlay_path']).relative_to(UPLOAD_DIR)}"
            
            logger.info(f"[Translation] Converted URLs:")
            logger.info(f"  Original: {original_overlay_url}")
            logger.info(f"  Translated: {translated_overlay_url}")
            
            response_data = {
                "success": True,
                "detected_source_lang": result['detected_source_lang'],
                "target_language": result['target_language'],
                "metadata": result['metadata'],
                "original_image": f"/uploads/{filename}",
                "original_overlay": original_overlay_url,
                "translated_overlay": translated_overlay_url
            }
            
            logger.info(f"[Translation] Final response data: {json.dumps(response_data, indent=2)}")
            return response_data
            
        except Exception as e:
            logger.error(f"[Translation] Processing error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Image processing failed: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"[Translation] Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {str(e)}"
        )

@app.post("/api/translate")
async def translate_texts(
    texts: list[str],
    target_lang: str,
    source_lang: Optional[str] = None
):
    """
    Translate texts to the target language.
    
    Args:
        texts: List of texts to translate
        target_lang: Target language code
        source_lang: Source language code (optional)
        
    Returns:
        JSON response containing original and translated texts
    """
    try:
        translations = await translation_service.translate_text(
            texts=texts,
            target_lang=target_lang,
            source_lang=source_lang
        )
        
        return JSONResponse(
            status_code=200,
            content={"translations": translations}
        )
        
    except Exception as e:
        logger.error(f"Error translating texts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "ocr_service": "available",
        "translation_service": "available" if translation_service._initialized else "unavailable"
    }

@app.get("/uploads/{year}/{month}/{filename}")
async def serve_image(year: str, month: str, filename: str):
    """Serve image files from the uploads directory."""
    file_path = UPLOAD_DIR / year / month / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path)) 