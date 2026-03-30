"""
Deep Learning Deepfake Image Detection Model
Uses TensorFlow/Keras with Xception architecture for robust deepfake detection.
"""

import numpy as np
import logging
import os
from typing import Tuple, Optional, Dict
import json

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import Xception, EfficientNetB3, ResNet50
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Model functionality will be limited.")


class DeepfakeImageModel:
    """
    Deep learning model for detecting AI-generated/deepfake images.
    Uses transfer learning with pretrained Xception architecture.
    """
    
    def __init__(
        self, 
        architecture: str = 'xception',
        input_size: int = 224,
        pretrained: bool = True,
        dropout_rate: float = 0.5
    ):
        """
        Initialize the deepfake detection model.
        
        Args:
            architecture: 'xception', 'efficientnet', or 'resnet50'
            input_size: Input image size
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate to prevent overfitting
        """
        self.architecture = architecture.lower()
        self.input_size = input_size
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.model = None
        self.threshold = 0.5
        
        if TENSORFLOW_AVAILABLE:
            self._build_model()
        else:
            logger.warning("Cannot build model without TensorFlow")
    
    def _build_model(self) -> models.Model:
        """
        Build the neural network model using transfer learning.
        
        Returns:
            Compiled Keras model
        """
        try:
            input_shape = (self.input_size, self.input_size, 3)
            
            # Select base model architecture
            if self.architecture == 'xception':
                base_model = Xception(
                    input_shape=input_shape,
                    include_top=False,
                    weights='imagenet' if self.pretrained else None
                )
            elif self.architecture == 'efficientnet':
                base_model = EfficientNetB3(
                    input_shape=input_shape,
                    include_top=False,
                    weights='imagenet' if self.pretrained else None
                )
            elif self.architecture == 'resnet50':
                base_model = ResNet50(
                    input_shape=input_shape,
                    include_top=False,
                    weights='imagenet' if self.pretrained else None
                )
            else:
                raise ValueError(f"Unknown architecture: {self.architecture}")
            
            # Freeze base model layers
            base_model.trainable = False
            
            # Build custom head
            model = models.Sequential([
                keras.Input(shape=input_shape),
                layers.Rescaling(1./255),  # Normalize input
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_rate),
                layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_rate),
                layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_rate),
                layers.Dense(1, activation='sigmoid')  # Binary classification
            ])
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC()]
            )
            
            self.model = model
            logger.info(f"✓ Model built successfully ({self.architecture} architecture)")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise
    
    def predict(
        self, 
        image: np.ndarray,
        return_probability: bool = True
    ) -> Dict[str, any]:
        """
        Make prediction on a single image.
        
        Args:
            image: Preprocessed image (H, W, 3) with values in [0, 1]
            return_probability: Whether to return raw probability
            
        Returns:
            Dict with keys:
            - 'prediction': 'REAL' or 'FAKE'
            - 'confidence': Confidence score (0-100)
            - 'probability': Raw probability [0, 1]
            - 'is_fake': Boolean
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Please load or train a model first.")
            
            # Ensure image is batch
            if len(image.shape) == 3:
                batch = np.expand_dims(image, axis=0)
            else:
                batch = image
            
            # Make prediction
            raw_output = self.model.predict(batch, verbose=0)
            probability = float(raw_output[0][0])
            
            # Determine prediction and confidence
            is_fake = probability >= self.threshold
            prediction = 'FAKE' if is_fake else 'REAL'
            confidence = abs(probability - 0.5) * 2  # Scale to 0-1
            confidence_percent = confidence * 100
            
            return {
                'prediction': prediction,
                'confidence': round(confidence_percent, 2),
                'confidence_percentage': f"{confidence_percent:.2f}%",
                'probability': round(probability, 4),
                'is_fake': is_fake,
                'fake_score': round(probability * 100, 2),
                'real_score': round((1 - probability) * 100, 2)
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def predict_batch(
        self,
        images: np.ndarray
    ) -> list:
        """
        Make predictions on a batch of images.
        
        Args:
            images: Batch of images (N, H, W, 3)
            
        Returns:
            List of prediction dictionaries
        """
        try:
            predictions = self.model.predict(images, verbose=0)
            results = []
            
            for prob in predictions:
                probability = float(prob[0])
                is_fake = probability >= self.threshold
                prediction = 'FAKE' if is_fake else 'REAL'
                confidence = abs(probability - 0.5) * 2
                
                results.append({
                    'prediction': prediction,
                    'confidence': round(confidence * 100, 2),
                    'probability': round(probability, 4),
                    'is_fake': is_fake
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during batch prediction: {e}")
            raise
    
    def save(self, save_path: str):
        """
        Save model to disk.
        
        Args:
            save_path: Path to save model
        """
        try:
            if self.model is None:
                raise RuntimeError("No model to save")
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save(save_path)
            logger.info(f"✓ Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load(self, load_path: str):
        """
        Load model from disk.
        
        Args:
            load_path: Path to load model from
        """
        try:
            if not os.path.exists(load_path):
                logger.warning(f"Model not found at {load_path}. Using untrained model.")
                return False
            
            self.model = keras.models.load_model(load_path)
            logger.info(f"✓ Model loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Using untrained model.")
            return False
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information and summary."""
        return {
            'architecture': self.architecture,
            'input_size': self.input_size,
            'pretrained': self.pretrained,
            'dropout_rate': self.dropout_rate,
            'threshold': self.threshold,
            'model_loaded': self.model is not None,
            'total_params': self.model.count_params() if self.model else 0
        }


class ModelHandler:
    """
    Main handler for model loading and management.
    Ensures model is loaded only once (singleton pattern).
    """
    
    _instance = None
    _model = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern - ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        architecture: str = 'xception'
    ):
        """
        Initialize model handler.
        
        Args:
            model_path: Path to saved model
            architecture: Model architecture to use
        """
        if self._model is not None:
            return  # Already initialized
        
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required but not installed")
            self._model = None
            return
        
        try:
            self._model = DeepfakeImageModel(architecture=architecture)
            
            # Try to load pretrained model
            if model_path and os.path.exists(model_path):
                self._model.load(model_path)
                logger.info(f"✓ Loaded model from {model_path}")
            else:
                logger.info("Using untrained model - predictions will be random")
                
        except Exception as e:
            logger.error(f"Error initializing model handler: {e}")
            self._model = None
    
    @property
    def model(self) -> Optional[DeepfakeImageModel]:
        """Get the loaded model."""
        return self._model
    
    def is_ready(self) -> bool:
        """Check if model is ready for inference."""
        return self._model is not None
