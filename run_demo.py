#!/usr/bin/env python3
"""
Deepfake AI Detection - Demo Backend Launcher
Handles missing ML libraries gracefully
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

print("="*60)
print("Deepfake AI Detection System - Starting Backend")
print("="*60)

# Check for ML libraries
ml_available = True
missing = []

try:
  import torch
except ImportError:
   ml_available = False
   missing.append('torch')

try:
  import transformers
except ImportError:
   ml_available = False
   missing.append('transformers')

if missing:
  print(f"\n⚠ ML libraries missing: {', '.join(missing)}")
  print("Running in DEMO MODE with mock predictions")
  print("\nTo enable full functionality:")
  print("  pip install torch torchvision transformers opencv-python\n")
else:
  print("\n✓ All ML libraries available")

print(f"Device: {'CUDA' if ml_available and torch.cuda.is_available() else 'CPU'}")
print("="*60)

# Import FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import uuid
import random
from datetime import datetime

# Create app
app = FastAPI(title="Deepfake AI Detection", version="1.0.0-demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "images")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
  return {
       "message": "Deepfake AI Detection API (Demo Mode)" if not ml_available else "Deepfake AI Detection API",
       "version": "1.0.0",
       "ml_available": ml_available,
       "endpoints": {
           "detect_image": "POST /detect-image",
           "detect_text": "POST /detect-text",
           "health": "GET /health"
       }
   }

@app.get("/health")
async def health():
  return {
       "status": "healthy",
       "timestamp": datetime.now().isoformat(),
       "models": {
           "image_model_loaded": ml_available,
           "text_model_loaded": ml_available
       }
   }

@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
  if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {ext}")
    
  try:
        # Save file
        filename = f"{uuid.uuid4()}{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)
      content = await file.read()
        with open(filepath, 'wb') as f:
            f.write(content)
        
      if ml_available:
            # Use real model
          from image_model import initialize_image_model
            handler = initialize_image_model()
          prediction, confidence = handler.predict(filepath)
        else:
            # Mock prediction
          prediction = random.choice(["Real", "Fake"])
          confidence = random.uniform(75.0, 98.0)
        
      return {
           "success": True,
           "prediction": prediction,
           "confidence": round(confidence, 2),
           "confidence_percentage": f"{confidence:.2f}%",
           "color": "#EF4444" if prediction == "Fake" else "#10B981",
           "filename": filename,
           "timestamp": datetime.now().isoformat(),
           "demo_mode": not ml_available
       }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-text")
async def detect_text(request: Request):
    data = await request.json()
    text = data.get('text', '')
    
  if not text or len(text) < 10:
        raise HTTPException(status_code=400, detail="Text too short (min 10 chars)")
    
  if ml_available:
      from text_model import initialize_text_model
        handler = initialize_text_model()
      prediction, confidence = handler.predict(text)
    else:
      prediction = random.choice(["Human", "AI Generated"])
      confidence = random.uniform(70.0, 95.0)
    
  return {
       "success": True,
       "prediction": prediction,
       "confidence": round(confidence, 2),
       "confidence_percentage": f"{confidence:.2f}%",
       "color": "#EF4444" if "AI" in prediction else "#10B981",
       "text_length": len(text),
       "timestamp": datetime.now().isoformat(),
       "demo_mode": not ml_available
   }

@app.get("/model-info")
async def model_info():
  return {
       "image_model": {
           "loaded": ml_available,
           "type": "ResNet50" if ml_available else "Mock (install torch)"
       },
       "text_model": {
           "loaded": ml_available,
           "type": "BERT" if ml_available else "Mock (install transformers)"
       },
       "system": {
           "ml_available": ml_available,
           "note": "Install requirements.txt for full functionality"
       }
   }

# Start server
if __name__ == "__main__":
  print("\nStarting server...")
  print("URL: http://localhost:8000")
  print("Docs: http://localhost:8000/docs")
  print("Health: http://localhost:8000/health")
  print("\nPress Ctrl+C to stop\n")
  print("-"*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
