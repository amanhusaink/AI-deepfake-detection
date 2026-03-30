from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn, os, uuid, random
from datetime import datetime

app = FastAPI(title='Deepfake AI Detection', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'images')
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get('/')
async def root():
    return {'message': 'Deepfake AI Detection API- DEMO MODE', 'version': '1.0.0', 'endpoints': {'detect_image': 'POST /detect-image', 'detect_text': 'POST /detect-text', 'health': 'GET /health'}}

@app.get('/health')
async def health():
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat(), 'models': {'image_model_loaded': False, 'text_model_loaded': False}}

@app.post('/detect-image')
async def detect_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
        raise HTTPException(status_code=400, detail=f'Invalid file type: {ext}')
    try:
        filename = f'{uuid.uuid4()}{ext}'
        filepath = os.path.join(UPLOAD_DIR, filename)
        content = await file.read()
        with open(filepath, 'wb') as f:
            f.write(content)
        prediction = random.choice(['Real', 'Fake'])
        confidence = random.uniform(75.0, 98.0)
        return {'success': True, 'prediction': prediction, 'confidence': round(confidence, 2), 'confidence_percentage': f'{confidence:.2f}%', 'color': '#EF4444' if prediction == 'Fake' else '#10B981', 'filename': filename, 'timestamp': datetime.now().isoformat(), 'demo_mode': True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/detect-text')
async def detect_text(request: Request):
    data = await request.json()
    text = data.get('text', '')
    if not text or len(text) < 10:
        raise HTTPException(status_code=400, detail='Text too short (min 10 chars)')
    prediction = random.choice(['Human', 'AI Generated'])
    confidence = random.uniform(70.0, 95.0)
    return {'success': True, 'prediction': prediction, 'confidence': round(confidence, 2), 'confidence_percentage': f'{confidence:.2f}%', 'color': '#EF4444' if 'AI' in prediction else '#10B981', 'text_length': len(text), 'timestamp': datetime.now().isoformat(), 'demo_mode': True}

if __name__ == '__main__':
    print('='*60)
    print('Starting Deepfake Detection Backend - DEMO MODE')
    print('='*60)
    print('Server: http://localhost:8000')
    print('Docs: http://localhost:8000/docs')
    print('Health: http://localhost:8000/health')
    print('\nRunning with MOCK predictions (no ML libraries)')
    print('-'*60)
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
