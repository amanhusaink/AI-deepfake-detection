"""
Image Deepfake Detection Model using ResNet50
This module implements a CNN-based model for detecting AI-generated/fake images.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import os


class ImageDeepfakeDetector(nn.Module):
    """
    ResNet50-based binary classifier for deepfake image detection.
    Real (0) vs Fake (1)
    """
    
    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.5):
        super(ImageDeepfakeDetector, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Replace the final fully connected layer for binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2),  # Binary classification
        )
        
    def forward(self, x):
        return self.resnet(x)
    
    def predict_proba(self, x):
        """Return probability scores for each class"""
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities


class ImageModelHandler:
    """
    Handler class for loading model and making predictions on images.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the image model handler.
        
        Args:
            model_path: Path to trained model weights (.pth file)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = ImageDeepfakeDetector(pretrained=True)
        self.model_path = model_path
        
        # Image preprocessing parameters
        self.img_size = 224
        self.mean = [0.485, 0.456, 0.406]  # ImageNet mean
        self.std = [0.229, 0.224, 0.225]   # ImageNet std
        
        # Load model weights if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
    def load_model(self, model_path: str):
        """
        Load trained model weights.
        
        Args:
            model_path: Path to .pth file
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
                
            print(f"✓ Loaded image model from: {model_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not load model weights: {e}")
            print("Using randomly initialized weights")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for model input using OpenCV.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed tensor
        """
        # Read image with OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet statistics
        image = (image - self.mean) / self.std
        
        # Convert to tensor and change channel order (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Predict whether an image is real or fake.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (prediction_label, confidence_score)
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get predicted class and confidence
                confidence, predicted = torch.max(probabilities, 1)
                
                prediction = "Fake" if predicted.item() == 1 else "Real"
                confidence_score = confidence.item() * 100  # Convert to percentage
                
            return prediction, confidence_score
            
        except Exception as e:
            raise Exception(f"Error during image prediction: {e}")
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Predict multiple images in batch.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of (prediction, confidence) tuples
        """
        results = []
        
        for img_path in image_paths:
            try:
                prediction, confidence = self.predict(img_path)
                results.append((img_path, prediction, confidence))
            except Exception as e:
                results.append((img_path, f"Error: {str(e)}", 0.0))
        
        return results


def initialize_image_model(model_path: Optional[str] = None) -> ImageModelHandler:
    """
    Factory function to initialize image model.
    
    Args:
        model_path: Path to trained model weights
        
    Returns:
        ImageModelHandler instance
    """
    return ImageModelHandler(model_path=model_path)


# Example usage and testing
if __name__ == "__main__":
    # Test initialization
    print("Initializing Image Deepfake Detector...")
    handler = initialize_image_model()
    print(f"Model loaded on device: {handler.device}")
    print("Model architecture:")
    print(handler.model)
