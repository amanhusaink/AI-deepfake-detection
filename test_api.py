"""
API Testing Script for Deepfake Detection System
Test all backend endpoints programmatically.
"""

import requests
import json
import os
from pathlib import Path

BASE_URL = "http://localhost:8000"


def test_health_endpoint():
    """Test health check endpoint"""
   print("\n" + "="*60)
   print("Testing Health Endpoint")
   print("="*60)
    
   try:
        response = requests.get(f"{BASE_URL}/health")
        
       if response.status_code == 200:
            data = response.json()
           print("✓ Health check passed")
           print(f"\nStatus: {data['status']}")
           print(f"Image Model Loaded: {data['models']['image_model_loaded']}")
           print(f"Text Model Loaded: {data['models']['text_model_loaded']}")
           print(f"CUDA Available: {data['models']['cuda_available']}")
            return True
        else:
           print(f"✗ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
       print("✗ Cannot connect to backend server!")
       print("  Make sure the backend is running:")
       print("  $ cd backend && uvicorn app:app --reload")
        return False
    except Exception as e:
       print(f"✗ Error: {e}")
        return False


def test_root_endpoint():
    """Test root endpoint"""
   print("\n" + "="*60)
   print("Testing Root Endpoint")
   print("="*60)
    
   try:
        response = requests.get(f"{BASE_URL}/")
        
       if response.status_code == 200:
            data = response.json()
           print("✓ Root endpoint accessible")
           print(f"\nMessage: {data['message']}")
           print(f"Version: {data['version']}")
           print("\nAvailable Endpoints:")
           for endpoint, method in data['endpoints'].items():
               print(f"  {method} /{endpoint}")
            return True
        else:
           print(f"✗ Root endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
       print(f"✗ Error: {e}")
        return False


def test_image_detection(image_path=None):
    """Test image detection endpoint"""
   print("\n" + "="*60)
   print("Testing Image Detection")
   print("="*60)
    
    # Find a test image
   if image_path is None:
        sample_dir = Path("data/sample_data")
       if sample_dir.exists():
            images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
           if images:
                image_path= images[0]
    
   if image_path is None or not os.path.exists(image_path):
       print("⚠ No test image found. Skipping image detection test.")
       print("  Add an image to data/sample_data/ to test.")
        return None
    
   try:
       print(f"\nTesting with image: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{BASE_URL}/detect-image",
                files=files
            )
        
       if response.status_code == 200:
            data = response.json()
           print("✓ Image detection successful")
           print(f"\nPrediction: {data['prediction']}")
           print(f"Confidence: {data['confidence_percentage']}")
           print(f"Color: {data['color']}")
           print(f"Filename: {data['filename']}")
            return True
        else:
           print(f"✗ Image detection failed: {response.status_code}")
           print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
       print(f"✗ Error: {e}")
        return False


def test_image_with_heatmap(image_path=None):
    """Test image detection with heatmap"""
   print("\n" + "="*60)
   print("Testing Image Detection with Heatmap")
   print("="*60)
    
    # Find a test image
   if image_path is None:
        sample_dir = Path("data/sample_data")
       if sample_dir.exists():
            images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
           if images:
                image_path= images[0]
    
   if image_path is None or not os.path.exists(image_path):
       print("⚠ No test image found. Skipping heatmap test.")
        return None
    
   try:
       print(f"\nTesting with image: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{BASE_URL}/detect-image-with-heatmap",
                files=files
            )
        
       if response.status_code == 200:
            data = response.json()
           print("✓ Heatmap detection successful")
           print(f"\nPrediction: {data['prediction']}")
           print(f"Confidence: {data['confidence_percentage']}")
           print(f"Heatmap URL: {data.get('heatmap_url', 'N/A')}")
           print(f"Heatmap Base64 (length): {len(data.get('heatmap_base64', ''))}")
            
            # Save heatmap if base64 provided
           if data.get('heatmap_base64'):
                import base64
                from PIL import Image
                import io
                
                # Decode base64
                img_data = base64.b64decode(data['heatmap_base64'].split(',')[1])
                img = Image.open(io.BytesIO(img_data))
                
                # Save heatmap
               heatmap_path = "test_heatmap.jpg"
                img.save(heatmap_path)
               print(f"✓ Heatmap saved to: {heatmap_path}")
            
            return True
        else:
           print(f"✗ Heatmap detection failed: {response.status_code}")
            return False
            
    except Exception as e:
       print(f"✗ Error: {e}")
        return False


def test_text_detection(test_texts=None):
    """Test text detection endpoint"""
   print("\n" + "="*60)
   print("Testing Text Detection")
   print("="*60)
    
   if test_texts is None:
        test_texts= [
            ("Human sample", "The quick brown fox jumps over the lazy dog. This is a simple sentence that humans often write."),
            ("AI sample", "Artificial intelligence has revolutionized many industries. Machine learning algorithms can process vast amounts of data efficiently.")
        ]
    
    results= []
    
   for name, text in test_texts:
       try:
           print(f"\nTesting {name}:")
           print(f"Text: {text[:50]}...")
            
            response = requests.post(
                f"{BASE_URL}/detect-text",
               json={"text": text}
            )
            
           if response.status_code == 200:
                data = response.json()
               print(f"✓ Text detection successful")
               print(f"  Prediction: {data['prediction']}")
               print(f"  Confidence: {data['confidence_percentage']}")
                results.append(True)
            else:
               print(f"✗ Text detection failed: {response.status_code}")
                results.append(False)
                
        except Exception as e:
           print(f"✗ Error: {e}")
            results.append(False)
    
    return all(results) if results else False


def test_model_info():
    """Test model info endpoint"""
   print("\n" + "="*60)
   print("Testing Model Info Endpoint")
   print("="*60)
    
   try:
        response = requests.get(f"{BASE_URL}/model-info")
        
       if response.status_code == 200:
            data = response.json()
           print("✓ Model info retrieved")
            
           print("\nImage Model:")
            img_info = data.get('image_model', {})
           print(f"  Loaded: {img_info.get('loaded')}")
           print(f"  Path: {img_info.get('path')}")
           print(f"  Exists: {img_info.get('exists')}")
           print(f"  Size (MB): {img_info.get('size_mb', 0):.2f}")
            
           print("\nText Model:")
            text_info = data.get('text_model', {})
           print(f"  Loaded: {text_info.get('loaded')}")
           print(f"  Path: {text_info.get('path')}")
           print(f"  Type: {text_info.get('type')}")
            
           print("\nSystem:")
            sys_info = data.get('system', {})
           print(f"  CUDA Available: {sys_info.get('cuda_available')}")
            
            return True
        else:
           print(f"✗ Model info failed: {response.status_code}")
            return False
            
    except Exception as e:
       print(f"✗ Error: {e}")
        return False


def run_all_tests():
    """Run all API tests"""
   print("\n" + "="*60)
   print("Deepfake Detection System - API Test Suite")
   print("="*60)
    
    results = {}
    
    # Test basic endpoints
    results['Health'] = test_health_endpoint()
    results['Root'] = test_root_endpoint()
    
    # Test model info
    results['Model Info'] = test_model_info()
    
    # Test detection endpoints
    results['Text Detection'] = test_text_detection()
    results['Image Detection'] = test_image_detection()
    results['Image + Heatmap'] = test_image_with_heatmap()
    
    # Summary
   print("\n" + "="*60)
   print("Test Summary")
   print("="*60)
    
   passed = sum(1 for v in results.values() if v)
    total = len(results)
    
   for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
       print(f"{status} - {test_name}")
    
   print(f"\nTotal: {passed}/{total} tests passed ({100*passed/total:.1f}%)" if total > 0 else "\nNo tests completed")
    
   if passed == total and total > 0:
       print("\n🎉 All tests passed! System is working correctly.")
    elif passed > 0:
       print("\n⚠ Some tests failed. Check the output above for details.")
    else:
       print("\n❌ All tests failed. Please check your setup.")
    
   print("\n" + "="*60)
    
    return passed == total and total > 0


if __name__ == "__main__":
    import sys
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
