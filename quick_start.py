#!/usr/bin/env python3
"""
Quick Start Script for Deepfake Detection System
This script helps you set up and run the system quickly.
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner(text):
    """Print a formatted banner"""
   print("\n" + "="*60)
   print(text.center(60))
   print("="*60 + "\n")


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
   print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
   if version.major < 3 or (version.major == 3 and version.minor < 8):
       print("❌ Error: Python 3.8 or higher is required!")
        return False
    
   print("✓ Python version is compatible")
    return True


def install_dependencies():
    """Install required dependencies"""
   print_banner("Installing Dependencies")
    
    requirements_path= Path("backend/requirements.txt")
    
   if not requirements_path.exists():
       print("❌ Error: requirements.txt not found!")
        return False
    
   try:
       print("Installing packages... (this may take a few minutes)")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
        ])
       print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
       print(f"❌ Error installing dependencies: {e}")
        return False


def check_model_status():
    """Check if models are available"""
   print_banner("Checking Model Status")
    
    image_model = Path("models/image_model.pth")
    text_model = Path("models/text_model")
    
   models_available = True
    
    # Check image model
   if image_model.exists():
        size_mb = image_model.stat().st_size / (1024 * 1024)
       print(f"✓ Image model found: {image_model} ({size_mb:.2f} MB)")
    else:
       print(f"⚠ Image model not found: {image_model}")
       print("  → You need to train the model or download pretrained weights")
       models_available = False
    
    # Check text model
   if text_model.exists() and text_model.is_dir():
       print(f"✓ Text model found: {text_model}")
        
        # Check essential files
        essential_files = ['config.json', 'pytorch_model.bin']
        missing_files = []
        for file in essential_files:
           if not (text_model / file).exists():
                missing_files.append(file)
        
       if missing_files:
           print(f"  ⚠ Missing files: {', '.join(missing_files)}")
           print("  → Model may not be fully trained")
           models_available = False
        else:
           print("  ✓ All model files present")
    else:
       print(f"⚠ Text model directory not found: {text_model}")
       print("  → You need to train the model or download pretrained weights")
       models_available = False
    
    return models_available


def create_directories():
    """Create necessary directories"""
   print_banner("Creating Directories")
    
    dirs = [
        "backend",
        "frontend",
        "models",
        "data/images",
        "data/sample_data",
        "data/training/images/real",
        "data/training/images/fake",
        "data/training/text/human",
        "data/training/text/ai"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
       print(f"✓ Created: {dir_path}")
    
    # Create .gitkeep files
    for dir_path in ["data/images", "data/sample_data"]:
        gitkeep = Path(dir_path) / ".gitkeep"
       if not gitkeep.exists():
            gitkeep.touch()
    
   print("✓ Directory structure created")


def start_backend():
    """Start the FastAPI backend server"""
   print_banner("Starting Backend Server")
    
   backend_dir = Path("backend")
    
   if not (backend_dir / "app.py").exists():
       print("❌ Error: app.py not found in backend/")
        return False
    
   try:
       print("Starting FastAPI server at http://localhost:8000")
       print("Press Ctrl+C to stop the server")
        
        # Change to backend directory
        os.chdir(backend_dir)
        
        # Start uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn", "app:app", "--reload",
            "--host", "0.0.0.0", "--port", "8000"
        ])
        
        return True
    except KeyboardInterrupt:
       print("\n✓ Backend server stopped")
        return True
    except Exception as e:
       print(f"❌ Error starting backend: {e}")
        return False


def show_instructions():
    """Show usage instructions"""
   print_banner("Setup Complete! Next Steps")
    
    instructions = """
✅ Installation Complete!

To use the Deepfake Detection System:

1. TRAIN OR DOWNLOAD MODELS (Required)
   
   Option A: Train your own models
   ────────────────────────────────
   For image model:
   $ python train_image_model.py
   
   For text model:
   $ python train_text_model.py --data-dir data/training/text
   
   Option B: Download pretrained models
   ─────────────────────────────────────
   Place pretrained weights in models/ directory
   - image_model.pth
   - text_model/ (directory)

2. START THE BACKEND SERVER
   
   $ cd backend
   $ uvicorn app:app --reload --host 0.0.0.0 --port 8000
   
   Or use the quick start:
   $ python quick_start.py start

3. OPEN THE FRONTEND
   
   - Open frontend/index.html in your browser
   - Or serve with: python-m http.server 3000
   - Navigate to http://localhost:3000

4. TEST THE SYSTEM
   
   - Upload an image for deepfake detection
   - Enter text for AI detection
   - View results with confidence scores

5. RUN DEMO (Optional)
   
   $ python test_demo.py
   
   This will test the models with sample data.

📖 For detailed documentation, see README.md

🔗 API Documentation: http://localhost:8000/docs (after starting backend)

"""
    
   print(instructions)


def main():
    """Main entry point"""
   print_banner("Deepfake Detection System - Quick Start")
    
    # Check Python version
   if not check_python_version():
        sys.exit(1)
    
    # Parse command line arguments
   if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
       if command == "install":
            success = install_dependencies()
            sys.exit(0 if success else 1)
        
        elif command == "start":
            start_backend()
            sys.exit(0)
        
        elif command == "check":
            check_model_status()
            sys.exit(0)
        
        else:
           print(f"Unknown command: {command}")
           print("Available commands: install, start, check")
            sys.exit(1)
    
    # Interactive setup
   print("Welcome to the Deepfake Detection System!\n")
   print("This script will help you set up the system.\n")
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Install dependencies
    response = input("\nWould you like to install dependencies? (y/n): ").strip().lower()
   if response == 'y':
       if not install_dependencies():
           print("\n⚠ Dependency installation failed. You can install manually later.")
    
    # Step 3: Check model status
   print("\n")
   models_ready = check_model_status()
    
   if not models_ready:
       print("\n⚠ Models are not ready. You need to:")
       print("  1. Train models using the training scripts, OR")
       print("  2. Download pretrained model weights")
       print("\nSee README.md for detailed instructions.\n")
    
    # Step 4: Show next steps
    show_instructions()
    
    # Ask if user wants to start backend
   if models_ready:
        response = input("Would you like to start the backend server now? (y/n): ").strip().lower()
       if response == 'y':
            start_backend()
    
   print("\n✓ Setup assistant completed!")
   print("\nTo restart the backend later, run: python quick_start.py start\n")


if __name__ == "__main__":
    main()
