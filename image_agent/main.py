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

from .services.ocr_service import ocr_service
from .services.translation_service import translation_service
from .schemas import UploadResponse, OCRResponse, TextDetection

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
    source_lang: str = 'auto'
):
    """
    Perform OCR on an uploaded image file.
    """
    start_time = datetime.now()
    try:
        logger.info(f"[OCR] Starting OCR process at {start_time}")
        
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
        logger.info(f"[OCR] File saved to: {file_path}")
        logger.info(f"[OCR] File save took: {(datetime.now() - save_start).total_seconds():.2f}s")
        
        # Reset file pointer for OCR
        file.file.seek(0)
        
        try:
            # Perform OCR with the saved file path
            ocr_start = datetime.now()
            texts, overlay_path = await ocr_service.detect_text(file=file, source_lang=source_lang, saved_file_path=str(file_path))
            logger.info(f"[OCR] OCR processing took: {(datetime.now() - ocr_start).total_seconds():.2f}s")
            
            if not overlay_path:
                logger.warning("[OCR] No text detected in image")
                return JSONResponse(
                    status_code=200,
                    content={
                        "message": "No text detected in image",
                        "original_image": None,
                        "overlay_image": None
                    }
                )
            
            # Prepare response
            response_start = datetime.now()
            
            # Get current date for URL construction
            current_date = datetime.now()
            year = str(current_date.year)
            month = f"{current_date.month:02d}"
            
            # Construct URLs using the new endpoint format
            original_filename = Path(file_path).name
            overlay_filename = Path(overlay_path).name if overlay_path else None
            
            original_url = f"/uploads/{year}/{month}/{original_filename}"
            overlay_url = f"/uploads/{year}/{month}/{overlay_filename}" if overlay_filename else None
            
            logger.info(f"[OCR] Original image URL: {original_url}")
            if overlay_url:
                logger.info(f"[OCR] Overlay image URL: {overlay_url}")
            
            response = JSONResponse(
                status_code=200,
                content={
                    "detected_source_lang": source_lang,
                    "original_image": original_url,
                    "overlay_image": overlay_url
                }
            )
            logger.info(f"[OCR] Response preparation took: {(datetime.now() - response_start).total_seconds():.2f}s")
            
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"[OCR] Total processing time: {total_time:.2f}s")
            return response
            
        except Exception as ocr_error:
            error_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"[OCR] OCR processing error after {error_time:.2f}s: {str(ocr_error)}")
            # Clean up the saved file if OCR failed
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"[OCR] Cleaned up file after error: {file_path}")
            except Exception as cleanup_error:
                logger.error(f"[OCR] Failed to clean up file after error: {str(cleanup_error)}")
            
            if "GOAWAY received" in str(ocr_error) or "session_timed_out" in str(ocr_error):
                raise HTTPException(
                    status_code=503,
                    detail="Google Vision API temporarily unavailable. Please try again."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"OCR processing failed: {str(ocr_error)}"
                )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        error_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"[OCR] Error after {error_time:.2f}s: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
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