"""
Deepfake AI Detection System - FastAPI Backend
Main application file with API endpoints for image and text detection.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import uuid
import shutil
from typing import Optional
from datetime import datetime
import logging

# Import model handlers
from image_model import initialize_image_model
from text_model import initialize_text_model
from utils import (
    generate_heatmap_gradcam,
    encode_image_to_base64,
    save_uploaded_file,
    validate_image_file,
    get_prediction_color,
    COLORS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Deepfake AI Detection System",
    description="API for detecting AI-generated images and text",
    version="1.0.0"
)

# CORS middleware - allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "data", "images")
HEATMAP_DIR = os.path.join(BASE_DIR, "..", "data", "images", "heatmaps")

# Model paths
IMAGE_MODEL_PATH = os.path.join(MODELS_DIR, "image_model.pth")
TEXT_MODEL_PATH = os.path.join(MODELS_DIR, "text_model")

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Global model instances
image_model_handler = None
text_model_handler = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global image_model_handler, text_model_handler
    
    logger.info("Initializing Deepfake Detection Models...")
    
    # Initialize image model
    try:
        if os.path.exists(IMAGE_MODEL_PATH):
            image_model_handler = initialize_image_model(IMAGE_MODEL_PATH)
            logger.info(f"✓ Image model loaded from: {IMAGE_MODEL_PATH}")
        else:
            image_model_handler = initialize_image_model()
            logger.warning("⚠ Using uninitialized image model (no weights found)")
    except Exception as e:
        logger.error(f"Failed to initialize image model: {e}")
        image_model_handler = None
    
    # Initialize text model
    try:
        if os.path.exists(TEXT_MODEL_PATH):
            text_model_handler = initialize_text_model(TEXT_MODEL_PATH)
            logger.info(f"✓ Text model loaded from: {TEXT_MODEL_PATH}")
        else:
            text_model_handler = initialize_text_model()
            logger.warning("⚠ Using uninitialized text model (no weights found)")
    except Exception as e:
        logger.error(f"Failed to initialize text model: {e}")
        text_model_handler = None
    
    logger.info("✓ Model initialization complete")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Deepfake AI Detection System API",
        "version": "1.0.0",
        "endpoints": {
            "detect_image": "POST /detect-image",
            "detect_text": "POST /detect-text",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "image_model_loaded": image_model_handler is not None,
            "text_model_loaded": text_model_handler is not None,
            "cuda_available": image_model_handler.device.type == 'cuda' if image_model_handler else False
        }
    }


@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    """
    Detect whether an uploaded image is AI-generated (deepfake) or real.
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with prediction and confidence score
    """
    if image_model_handler is None:
        raise HTTPException(status_code=503, detail="Image detection model not available. Please train or download model weights.")
    
    try:
        # Validate file type
        allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {allowed_extensions}"
            )
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = save_uploaded_file(await file.read(), UPLOAD_DIR, unique_filename)
        
        # Validate image
        validate_image_file(file_path, max_size_mb=10.0)
        
        # Make prediction
        prediction, confidence = image_model_handler.predict(file_path)
        
        # Prepare response
        response = {
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "confidence_percentage": f"{confidence:.2f}%",
            "color": get_prediction_color(prediction),
            "filename": unique_filename,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Image prediction: {prediction} ({confidence:.2f}%)")
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during image detection: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/detect-image-with-heatmap")
async def detect_image_with_heatmap(file: UploadFile = File(...)):
    """
    Detect deepfake image and generate heatmap showing manipulated regions.
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with prediction, confidence, and heatmap
    """
    if image_model_handler is None:
        raise HTTPException(status_code=500, detail="Image model not initialized")
    
    try:
        # Validate file type
        allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {allowed_extensions}"
            )
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = save_uploaded_file(await file.read(), UPLOAD_DIR, unique_filename)
        
        # Validate image
        validate_image_file(file_path, max_size_mb=10.0)
        
        # Make prediction
        prediction, confidence = image_model_handler.predict(file_path)
        
        # Generate heatmap
        use_cuda = image_model_handler.device.type == 'cuda'
        original_img, heatmap_overlay = generate_heatmap_gradcam(
            model=image_model_handler.model,
            image_path=file_path,
            use_cuda=use_cuda
        )
        
        # Save heatmap
        heatmap_filename = f"heatmap_{unique_filename}"
        heatmap_path = os.path.join(HEATMAP_DIR, heatmap_filename)
        
        # Convert RGB to BGR for OpenCV save
        import cv2
        heatmap_bgr = cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(heatmap_path, heatmap_bgr)
        
        # Encode heatmap to base64 for response
        heatmap_base64 = encode_image_to_base64(heatmap_overlay)
        
        # Prepare response
        response = {
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "confidence_percentage": f"{confidence:.2f}%",
            "color": get_prediction_color(prediction),
            "heatmap_url": f"/static/heatmaps/{heatmap_filename}",
            "heatmap_base64": f"data:image/jpeg;base64,{heatmap_base64}",
            "filename": unique_filename,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Image with heatmap: {prediction} ({confidence:.2f}%)")
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during image detection with heatmap: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/detect-text")
async def detect_text(request: Request):
    """
    Detect whether text is human-written or AI-generated.
    
    Args:
        request: JSON body with 'text' field
        
    Returns:
        JSON response with prediction and confidence score
    """
    if text_model_handler is None:
        raise HTTPException(status_code=503, detail="Text detection model not available. Please train or download model weights.")
    
    try:
        # Parse request body
        data = await request.json()
        text = data.get('text', '')
        
        # Validate input
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Empty text input")
        
        if len(text) < 10:
            raise HTTPException(
                status_code=400,
                detail="Text too short. Minimum 10 characters required."
            )
        
        # Truncate very long texts
        if len(text) > 5000:
            text = text[:5000] + "..."
        
        # Make prediction
        prediction, confidence = text_model_handler.predict(text)
        
        # Prepare response
        response = {
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "confidence_percentage": f"{confidence:.2f}%",
            "color": get_prediction_color(prediction),
            "text_length": len(text),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Text prediction: {prediction} ({confidence:.2f}%)")
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during text detection: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.get("/model-info")
async def get_model_info():
    """Get information about loaded models"""
    return {
        "image_model": {
            "loaded": image_model_handler is not None,
            "path": IMAGE_MODEL_PATH,
            "exists": os.path.exists(IMAGE_MODEL_PATH),
            "device": str(image_model_handler.device) if image_model_handler else None,
            "size_mb": os.path.getsize(IMAGE_MODEL_PATH) / (1024 * 1024) if os.path.exists(IMAGE_MODEL_PATH) else 0
        },
        "text_model": {
            "loaded": text_model_handler is not None,
            "path": TEXT_MODEL_PATH,
            "exists": os.path.exists(TEXT_MODEL_PATH),
            "device": str(text_model_handler.device) if text_model_handler else None,
            "type": "BERT-based classifier"
        },
        "system": {
            "cuda_available": torch.cuda.is_available() if 'torch' in globals() else False,
            "upload_dir": UPLOAD_DIR,
            "heatmap_dir": HEATMAP_DIR
        }
    }


# Mount static files for serving heatmaps
app.mount("/static/heatmaps", StaticFiles(directory=HEATMAP_DIR), name="heatmaps")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": f"Internal server error: {str(exc)}",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
