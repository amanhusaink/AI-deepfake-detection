"""
Demo script to test the Deepfake Detection System with sample data.
Use this to test the models without running the full web application.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.image_model import initialize_image_model
from backend.text_model import initialize_text_model


def test_image_detection():
    """Test image deepfake detection"""
    print("\n" + "="*60)
    print("Testing Image Deepfake Detection")
    print("="*60)
    
    # Initialize model
    print("\nInitializing image model...")
    handler = initialize_image_model()
    
    print(f"✓ Model loaded on: {handler.device}")
    
    # Check if test images exist
    test_dir = 'data/sample_data'
    if not os.path.exists(test_dir):
        print(f"\n⚠ Sample data directory not found: {test_dir}")
        print("Add sample images to test the model.")
        return
    
    # Find test images
    image_files = [f for f in os.listdir(test_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    
    if len(image_files) == 0:
        print(f"\n⚠ No images found in {test_dir}")
        print("Add sample images to test the model.")
        return
    
    print(f"\nFound {len(image_files)} test images")
    
    # Test each image
    for img_name in image_files:
        img_path = os.path.join(test_dir, img_name)
        print(f"\nAnalyzing: {img_name}")
        
        try:
            prediction, confidence = handler.predict(img_path)
            
            print(f"  Prediction: {prediction}")
            print(f"  Confidence: {confidence:.2f}%")
            
            if 'Fake' in prediction:
                print(f"  ⚠ This image appears to be AI-generated/fake!")
            else:
                print(f"  ✓ This image appears to be real!")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")


def test_text_detection():
    """Test AI text detection"""
    print("\n" + "="*60)
    print("Testing AI Text Detection")
    print("="*60)
    
    # Initialize model
    print("\nInitializing text model...")
    handler = initialize_text_model()
    
    print(f"✓ Model loaded on: {handler.device}")
    
    # Test texts
    test_texts = [
        ("Human-written sample", 
         "The morning sun cast long shadows across the quiet street as I made my way to the corner café. There was something comforting about the familiar routine - the smell of freshly ground coffee beans, the gentle hum of conversation, the warmth of the ceramic mug in my hands."),
        
        ("AI-generated sample",
         "Artificial intelligence has revolutionized numerous industries in the twenty-first century. Machine learning algorithms process vast amounts of data to identify patterns and make predictions. These systems utilize neural networks with multiple layers to achieve increasingly sophisticated tasks."),
    ]
    
    print("\n" + "-"*60)
    for name, text in test_texts:
        print(f"\nTest: {name}")
        print(f"Text: {text[:80]}...")
        
        try:
            prediction, confidence = handler.predict(text)
            
            print(f"  Prediction: {prediction}")
            print(f"  Confidence: {confidence:.2f}%")
            
            if 'AI' in prediction:
                print(f"  🤖 Detected as AI-generated")
            else:
                print(f"  👤 Detected as Human-written")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")


def check_model_status():
    """Check if models are properly initialized"""
    print("\n" + "="*60)
    print("Model Status Check")
    print("="*60)
    
    # Check image model
    print("\n📷 Image Detection Model:")
    img_model_path = 'models/image_model.pth'
    if os.path.exists(img_model_path):
        size_mb = os.path.getsize(img_model_path) / (1024 * 1024)
        print(f"  ✓ Found: {img_model_path}")
        print(f"  Size: {size_mb:.2f} MB")
    else:
        print(f"  ✗ Not found: {img_model_path}")
        print(f"  → Run training or download pretrained weights")
    
    # Check text model
    print("\n📝 Text Detection Model:")
    text_model_path = 'models/text_model'
    if os.path.exists(text_model_path):
        print(f"  ✓ Found: {text_model_path}")
        
        # Check required files
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        for file in required_files:
            file_path = os.path.join(text_model_path, file)
            if os.path.exists(file_path):
                print(f"    ✓ {file}")
            else:
                print(f"    ✗ {file} missing")
    else:
        print(f"  ✗ Not found: {text_model_path}")
        print(f"  → Run training or download pretrained weights")
    
    # Check CUDA
    import torch
    print("\n💻 System Info:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  → Running on CPU (GPU recommended for faster inference)")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Deepfake Detection System - Demo & Testing")
    print("="*60)
    
    # Check model status
    check_model_status()
    
    # Ask user what to test
    print("\n" + "="*60)
    print("What would you like to test?")
    print("1. Image Detection")
    print("2. Text Detection")
    print("3. Both")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        test_image_detection()
    elif choice == '2':
        test_text_detection()
    elif choice == '3':
        test_image_detection()
        test_text_detection()
    elif choice == '4':
        print("\nGoodbye!")
        exit(0)
    else:
        print("\nInvalid choice. Please run again and select 1-4.")
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)
