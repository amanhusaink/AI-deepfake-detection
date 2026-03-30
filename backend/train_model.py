"""
Training Script for Deepfake Detection Model
Trains the model on a dataset of real and fake images.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import json
from pathlib import Path
from datetime import datetime
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from image_preprocessing import ImagePreprocessor
from image_model_tf import DeepfakeImageModel

# Configuration
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2
IMG_SIZE = 224
LEARNING_RATE = 1e-4


def load_dataset_from_directory(
    data_dir: str,
    real_subdir: str = 'real',
    fake_subdir: str = 'fake',
    img_size: int = 224,
    max_samples: int = None
) -> tuple:
    """
    Load images from directory structure.
    Expected structure:
        data_dir/
            real/
            fake/
    
    Args:
        data_dir: Root data directory
        real_subdir: Subdirectory containing real images
        fake_subdir: Subdirectory containing fake images
        img_size: Target image size
        max_samples: Maximum number of samples (per class)
        
    Returns:
        Tuple of (images, labels, metadata)
    """
    logger.info(f"Loading dataset from {data_dir}")
    
    images = []
    labels = []
    metadata = {
        'real_count': 0,
        'fake_count': 0,
        'failed_real': 0,
        'failed_fake': 0,
        'preprocessor_used': False
    }
    
    preprocessor = ImagePreprocessor(img_size=img_size)
    
    # Load real images
    real_dir = os.path.join(data_dir, real_subdir)
    if os.path.exists(real_dir):
        real_images = [f for f in os.listdir(real_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        if max_samples:
            real_images = real_images[:max_samples]
        
        logger.info(f"Loading {len(real_images)} real images...")
        
        for img_file in real_images:
            try:
                img_path = os.path.join(real_dir, img_file)
                processed_img, meta = preprocessor.preprocess_image(
                    img_path,
                    detect_face=True
                )
                
                if processed_img is not None:
                    images.append(processed_img)
                    labels.append(0)  # 0 = Real
                    metadata['real_count'] += 1
                else:
                    metadata['failed_real'] += 1
                    
            except Exception as e:
                logger.warning(f"Error processing {img_file}: {e}")
                metadata['failed_real'] += 1
    
    # Load fake images
    fake_dir = os.path.join(data_dir, fake_subdir)
    if os.path.exists(fake_dir):
        fake_images = [f for f in os.listdir(fake_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        if max_samples:
            fake_images = fake_images[:max_samples]
        
        logger.info(f"Loading {len(fake_images)} fake images...")
        
        for img_file in fake_images:
            try:
                img_path = os.path.join(fake_dir, img_file)
                processed_img, meta = preprocessor.preprocess_image(
                    img_path,
                    detect_face=True
                )
                
                if processed_img is not None:
                    images.append(processed_img)
                    labels.append(1)  # 1 = Fake
                    metadata['fake_count'] += 1
                else:
                    metadata['failed_fake'] += 1
                    
            except Exception as e:
                logger.warning(f"Error processing {img_file}: {e}")
                metadata['failed_fake'] += 1
    
    if len(images) == 0:
        raise ValueError("No images loaded. Check your data directory structure.")
    
    # Convert to numpy arrays
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    logger.info(f"Successfully loaded {len(images)} images")
    logger.info(f"  Real: {metadata['real_count']} (Failed: {metadata['failed_real']})")
    logger.info(f"  Fake: {metadata['fake_count']} (Failed: {metadata['failed_fake']})")
    
    return X, y, metadata


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    architecture: str = 'xception',
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    validation_split: float = VALIDATION_SPLIT,
    save_path: str = None
) -> keras.Model:
    """
    Train the deepfake detection model.
    
    Args:
        X_train: Training images
        y_train: Training labels
        architecture: Model architecture to use
        batch_size: Batch size for training
        epochs: Number of epochs
        validation_split: Validation split ratio
        save_path: Path to save trained model
        
    Returns:
        Trained model
    """
    logger.info(f"\nTraining with {len(X_train)} images")
    logger.info(f"Model architecture: {architecture}")
    logger.info(f"Batch size: {batch_size}, Epochs: {epochs}")
    
    # Initialize model
    model = DeepfakeImageModel(
        architecture=architecture,
        input_size=IMG_SIZE,
        pretrained=True,
        dropout_rate=0.5
    )
    
    # Create data augmentation for training
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomBrightness(0.2),
        keras.layers.RandomContrast(0.2)
    ])
    
    # Create augmented training dataset
    def augment_dataset(images, labels):
        return augmentation(images, training=True), labels
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(len(X_train))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(augment_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    
    # Train model
    history = model.model.fit(
        train_dataset,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        logger.info(f"✓ Model saved to {save_path}")
        
        # Save training history
        history_path = save_path.replace('.h5', '_history.json')
        history_dict = {
            'loss': history.history['loss'],
            'accuracy': history.history['accuracy'],
            'val_loss': history.history['val_loss'],
            'val_accuracy': history.history['val_accuracy']
        }
        with open(history_path, 'w') as f:
            json.dump(history_dict, f)
        logger.info(f"✓ Training history saved to {history_path}")
    
    return model.model


def main():
    """Main training script"""
    
    logger.info("="*60)
    logger.info("Deepfake Detection Model Training")
    logger.info("="*60)
    
    # Get data directory from command line or use default
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        # Use default location
        data_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "training",
            "images"
        )
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Expected structure:")
        logger.info("  data/training/images/")
        logger.info("    real/")
        logger.info("    fake/")
        sys.exit(1)
    
    # Load dataset
    try:
        X, y, metadata = load_dataset_from_directory(
            data_dir,
            img_size=IMG_SIZE,
            max_samples=None
        )
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Train model
    model_save_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "deepfake_detection_model.h5"
    )
    
    try:
        train_model(
            X, y,
            architecture='xception',
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT,
            save_path=model_save_path
        )
        logger.info("✓ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
