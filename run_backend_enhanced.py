#!/usr/bin/env python3
"""
Improved Backend Server Startup Script
Starts the FastAPI server with enhanced deepfake detection capabilities
"""

import sys
import os
import logging

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

print("="*60)
print("Deepfake AI Detection System - Enhanced Backend v2.0")
print("="*60)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check dependencies
logger.info("\nChecking dependencies...")
dependencies = {
    'tensorflow': False,
    'opencv': False,
    'numpy': False,
    'fastapi': False,
    'uvicorn': False
}

errors = []

try:
    import tensorflow
    dependencies['tensorflow'] = True
    print("✓ TensorFlow available")
except ImportError:
    errors.append('tensorflow')
    print("✗ TensorFlow not found")

try:
    import cv2
    dependencies['opencv'] = True
    print("✓ OpenCV available")
except ImportError:
    errors.append('opencv-python')
    print("✗ OpenCV not found")

try:
    import numpy
    dependencies['numpy'] = True
    print("✓ NumPy available")
except ImportError:
    errors.append('numpy')
    print("✗ NumPy not found")

try:
    import fastapi
    dependencies['fastapi'] = True
    print("✓ FastAPI available")
except ImportError:
    errors.append('fastapi')
    print("✗ FastAPI not found")

try:
    import uvicorn
    dependencies['uvicorn'] = True
    print("✓ Uvicorn available")
except ImportError:
    errors.append('uvicorn')
    print("✗ Uvicorn not found")

if errors:
    print("\n⚠ WARNING: Missing dependencies:")
    for dep in errors:
        print(f"  - {dep}")
    print("\nTo install missing dependencies, run:")
    print(f"  pip install {' '.join(errors)}")
    print("\nNote: The server may still run with limited functionality.")
else:
    print("\n✓ All required dependencies available")

# Check for GPU
print("\nChecking GPU availability...")
try:
    import tensorflow as tf
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    if gpu_available:
        print(f"✓ GPU detected: {tf.config.list_physical_devices('GPU')}")
        device = "GPU"
    else:
        print("✓ Running on CPU (training will be slower)")
        device = "CPU"
except Exception as e:
    print(f"✗ Error checking GPU: {e}")
    device = "CPU"

# Start server
print("\n" + "="*60)
print("Starting Enhanced FastAPI Backend Server...")
print("="*60)
print(f"\nBackend Information:")
print(f"  Version: 2.0.0")
print(f"  Device: {device}")
print(f"  Server URL: http://localhost:8000")
print(f"  API Docs: http://localhost:8000/docs")
print(f"  Health Check: http://localhost:8000/health")
print(f"  Model Info: http://localhost:8000/model-info")
print(f"\nFeatures:")
print(f"  - Face detection with Haar Cascade")
print(f"  - Xception deep learning model")
print(f"  - Image preprocessing pipeline")
print(f"  - Confidence scoring")
print(f"  - Heatmap visualization")
print(f"\nPress Ctrl+C to stop\n")
print("-"*60)

# Import and run uvicorn
try:
    import uvicorn
    
    # Use the improved app
    uvicorn.run(
        "app_improved:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
except ImportError:
    logger.error("Uvicorn not installed. Cannot start server.")
    logger.info("Install with: pip install uvicorn")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error starting server: {e}", exc_info=True)
    sys.exit(1)
