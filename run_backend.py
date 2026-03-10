#!/usr/bin/env python3
"""
Backend Server Startup Script
Starts the FastAPI server with optional ML model support
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

print("="*60)
print("Deepfake AI Detection System - Backend Server")
print("="*60)

# Check optional dependencies
ml_available = True
missing_deps = []

try:
    import torch
except ImportError:
    ml_available = False
    missing_deps.append('torch')

try:
    import transformers
except ImportError:
    ml_available = False
    missing_deps.append('transformers')

try:
    import cv2
except ImportError:
    ml_available = False
    missing_deps.append('opencv-python')

if missing_deps:
   print("\n⚠ WARNING: ML/AI libraries not installed:")
   for dep in missing_deps:
       print(f"  - {dep}")
   print("\nModels will use random initialization.")
   print("To enable full functionality, install:")
   print("  pip install torch torchvision transformers opencv-python")
else:
   print("\n✓ All ML libraries available")

print(f"\nDevice: {'CUDA' if ml_available and torch.cuda.is_available() else 'CPU'}")

# Start server
print("\n" + "="*60)
print("Starting FastAPI Server...")
print("="*60)
print("\nServer URL: http://localhost:8000")
print("API Docs: http://localhost:8000/docs")
print("Health Check: http://localhost:8000/health")
print("\nPress Ctrl+C to stop\n")
print("-"*60)

# Import and run uvicorn
import uvicorn

uvicorn.run(
    "app:app",
    host="0.0.0.0",
    port=8000,
   reload=True,
    log_level="info"
)
