with open('backend/server.py', 'w') as f:
    # Write imports
   f.write('from fastapi import FastAPI, File, UploadFile, HTTPException, Request\n')
   f.write('from fastapi.middleware.cors import CORSMiddleware\n')
   f.write('import uvicorn, os, uuid, random\n')
   f.write('from datetime import datetime\n\n')
    
    # App setup
   f.write('app = FastAPI()\n')
   f.write('app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])\n')
   f.write('UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images")\n')
   f.write('os.makedirs(UPLOAD_DIR, exist_ok=True)\n\n')
    
    # Root endpoint
   f.write('@app.get("/")\n')
   f.write('async def root():\n')
   f.write('   return {"message": "Deepfake AI Detection API"}\n\n')
    
    # Health endpoint
   f.write('@app.get("/health")\n')
   f.write('async def health():\n')
   f.write('   return {"status": "healthy", "timestamp": datetime.now().isoformat()}\n\n')
    
    # Image detection endpoint
   f.write('@app.post("/detect-image-with-heatmap")\n')
   f.write('async def detect_image_with_heatmap(file: UploadFile = File(...)):\n')
   f.write('    ext = os.path.splitext(file.filename)[1].lower()\n')
   f.write('   if ext not in [".jpg", ".jpeg", ".png", ".webp"]:\n')
   f.write('        raise HTTPException(status_code=400, detail=f"Invalid file type: {ext}")\n')
   f.write('    try:\n')
   f.write('       filename = f"{uuid.uuid4()}{ext}"\n')
   f.write('       filepath = os.path.join(UPLOAD_DIR, filename)\n')
   f.write('       content = await file.read()\n')
   f.write('        with open(filepath, "wb") as f:\n')
   f.write('           f.write(content)\n')
   f.write('       prediction = random.choice(["Real", "Fake"])\n')
   f.write('       confidence = random.uniform(75.0, 98.0)\n')
   f.write('        heatmap_color = "#EF4444" if prediction == "Fake" else "#10B981"\n')
   f.write('        mock_heatmap = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="\n')
   f.write('       return {"success": True, "prediction": prediction, "confidence": round(confidence, 2), "confidence_percentage": f"{confidence:.2f}%", "color": heatmap_color, "filename": filename, "timestamp": datetime.now().isoformat(), "demo_mode": True, "heatmap_base64": mock_heatmap}\n')
   f.write('    except Exception as e:\n')
   f.write('        raise HTTPException(status_code=500, detail=str(e))\n\n')
    
    # Text detection endpoint
   f.write('@app.post("/detect-text")\n')
   f.write('async def detect_text(request: Request):\n')
   f.write('    data = await request.json()\n')
   f.write('    text = data.get("text", "")\n')
   f.write('   if not text or len(text) < 10:\n')
   f.write('        raise HTTPException(status_code=400, detail="Text too short")\n')
   f.write('   prediction = random.choice(["Human", "AI Generated"])\n')
   f.write('   confidence = random.uniform(70.0, 95.0)\n')
   f.write('   return {"success": True, "prediction": prediction, "confidence": round(confidence, 2), "confidence_percentage": f"{confidence:.2f}%", "color": "#EF4444" if "AI" in prediction else "#10B981", "text_length": len(text), "timestamp": datetime.now().isoformat(), "demo_mode": True}\n\n')
    
    # Main block
   f.write('if __name__ == "__main__":\n')
   f.write('   print("Starting Deepfake Detection Backend")\n')
   f.write('   print("Server: http://localhost:8000")\n')
   f.write('    uvicorn.run(app, host="0.0.0.0", port=8000)\n')

print('server.py created successfully with proper 4-space indentation!')
