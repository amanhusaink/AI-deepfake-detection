"""
Advanced Image Preprocessing Pipeline with Face Detection
Handles face detection, validation, and image normalization for deepfake detection.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging
import os

logger = logging.getLogger(__name__)

# Initialize cascade classifier for face detection
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)


class ImagePreprocessor:
    """
    Handles all image preprocessing operations including face detection,
    resizing, and normalization.
    """
    
    def __init__(self, img_size: int = 224):
        """
        Initialize the preprocessor.
        
        Args:
            img_size: Target image size (default: 224x224)
        """
        self.img_size = img_size
        self.face_cascade = face_cascade
        
    def detect_faces(self, image: np.ndarray, min_confidence: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image using Haar Cascade.
        
        Args:
            image: Input image (BGR or RGB)
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        try:
            # Convert to grayscale for detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                maxSize=(500, 500),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return list(faces) if len(faces) > 0 else []
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def extract_face_region(
        self, 
        image: np.ndarray, 
        face_bbox: Tuple[int, int, int, int],
        padding: float = 0.1
    ) -> Optional[np.ndarray]:
        """
        Extract face region from image with optional padding.
        
        Args:
            image: Input image
            face_bbox: Face bounding box (x, y, w, h)
            padding: Padding ratio (0.1 = 10% padding)
            
        Returns:
            Extracted face region or None if extraction fails
        """
        try:
            x, y, w, h = face_bbox
            h_img, w_img = image.shape[:2]
            
            # Add padding
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)
            
            face_region = image[y1:y2, x1:x2]
            
            if face_region.size > 0:
                return face_region
            return None
            
        except Exception as e:
            logger.error(f"Error extracting face region: {e}")
            return None
    
    def preprocess_image(
        self, 
        image_path_or_array: any,
        detect_face: bool = True,
        return_face_bbox: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Complete preprocessing pipeline: load, detect faces, resize, normalize.
        
        Args:
            image_path_or_array: Path to image or numpy array
            detect_face: Whether to detect and extract faces
            return_face_bbox: Whether to return face bounding box info
            
        Returns:
            Tuple of (preprocessed_image, metadata_dict)
            metadata_dict contains: {
                'face_detected': bool,
                'face_bbox': Tuple or None,
                'original_size': Tuple,
                'num_faces': int
            }
        """
        metadata = {
            'face_detected': False,
            'face_bbox': None,
            'original_size': None,
            'num_faces': 0
        }
        
        try:
            # Load image
            if isinstance(image_path_or_array, str):
                if not os.path.exists(image_path_or_array):
                    raise FileNotFoundError(f"Image not found: {image_path_or_array}")
                image = cv2.imread(image_path_or_array)
                if image is None:
                    raise ValueError(f"Could not read image: {image_path_or_array}")
            else:
                image = image_path_or_array.copy()
            
            if image is None or image.size == 0:
                return None, metadata
            
            # Store original size
            h, w = image.shape[:2]
            metadata['original_size'] = (h, w)
            
            # Convert BGR to RGB for consistency
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Detect faces if requested
            if detect_face:
                faces = self.detect_faces(image)
                metadata['num_faces'] = len(faces)
                
                if len(faces) > 0:
                    # Use the largest face
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    face_region = self.extract_face_region(image_rgb, largest_face, padding=0.1)
                    
                    if face_region is not None:
                        metadata['face_detected'] = True
                        metadata['face_bbox'] = tuple(largest_face)
                        image_rgb = face_region
                else:
                    # No faces detected
                    logger.warning("No faces detected in image")
                    return None, metadata
            
            # Resize to target size
            image_resized = cv2.resize(image_rgb, (self.img_size, self.img_size), 
                                      interpolation=cv2.INTER_CUBIC)
            
            # Normalize to [0, 1]
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Ensure shape is (H, W, C)
            if len(image_normalized.shape) == 2:
                image_normalized = np.stack([image_normalized] * 3, axis=-1)
            
            return image_normalized, metadata
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            return None, metadata
    
    def preprocess_batch(
        self,
        images: List[np.ndarray],
        detect_face: bool = True
    ) -> Tuple[Optional[np.ndarray], List[dict]]:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of image arrays or paths
            detect_face: Whether to detect faces
            
        Returns:
            Tuple of (batch_array, metadata_list)
        """
        batch_data = []
        metadata_list = []
        
        for img in images:
            processed_img, metadata = self.preprocess_image(img, detect_face=detect_face)
            if processed_img is not None:
                batch_data.append(processed_img)
                metadata_list.append(metadata)
        
        if len(batch_data) == 0:
            return None, metadata_list
        
        # Stack into batch
        batch_array = np.stack(batch_data, axis=0)
        return batch_array, metadata_list
