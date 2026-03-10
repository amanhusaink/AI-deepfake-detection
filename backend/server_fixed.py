from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORMiddleware
from fastapi.responses import JSONResponse
import uvicorn, os, uuid, random
from datetime import datetime

app = FastAPI(title='Deepfake AI Detection', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'images')
os.makedirs(UPLOAD_DIR, exist_ok=True)
