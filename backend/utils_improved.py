"""
Utility functions for Deepfake Detection System
Includes image encoding, heatmap generation, and helper utilities.
"""

import cv2
import numpy as np
import base64
import io
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def encode_image_to_base64(image: np.ndarray, format: str = 'png') -> str:
    """
    Encode image to base64 string.
    
    Args:
        image: Image as numpy array (BGR)
        format: Image format ('png' or 'jpg')
        
    Returns:
        Base64 encoded image string
    """
    try:
        if format.lower() == 'png':
            _, buffer = cv2.imencode('.png', image)
        else:
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        b64_string = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/{format};base64,{b64_string}"
        
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return ""


def generate_heatmap_visualization(
    image: np.ndarray,
    prediction: str = 'REAL',
    confidence: float = 50.0,
    overlay_alpha: float = 0.3
) -> Optional[str]:
    """
    Generate a heatmap visualization overlay on the original image
    showing detection result and confidence.
    
    Args:
        image: Original image (BGR)
        prediction: 'REAL' or 'FAKE'
        confidence: Confidence percentage (0-100)
        overlay_alpha: Alpha blending factor
        
    Returns:
        Base64 encoded heatmap image or None
    """
    try:
        if image is None or image.size == 0:
            return None
        
        # Make a copy for visualization
        vis_image = image.copy()
        h, w = image.shape[:2]
        
        # Create color based on prediction
        if prediction == 'FAKE':
            # Red for fake
            color = (0, 0, 255)  # BGR
            # Create red overlay
            overlay = np.zeros_like(image)
            overlay[:] = color
        else:
            # Green for real
            color = (0, 255, 0)  # BGR
            overlay = np.zeros_like(image)
            overlay[:] = color
        
        # Blend overlay with original image
        vis_image = cv2.addWeighted(image, 1 - overlay_alpha, overlay, overlay_alpha, 0)
        
        # Add text box with result
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_color = color
        thickness = 3
        
        # Prediction text
        text = f"{prediction} ({confidence:.2f}%)"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 60
        
        # Draw background rectangle for text
        cv2.rectangle(
            vis_image,
            (text_x - 10, text_y - text_size[1] - 10),
            (text_x + text_size[0] + 10, text_y + 10),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            vis_image,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
        
        # Draw confidence bar at bottom
        bar_height = 30
        bar_y = h - bar_height
        bar_x1 = 50
        bar_x2 = w - 50
        bar_width = bar_x2 - bar_x1
        
        # Draw background bar
        cv2.rectangle(vis_image, (bar_x1, bar_y), (bar_x2, h), color, -1)
        
        # Draw filled portion based on confidence
        filled_width = int(bar_width * (confidence / 100.0))
        cv2.rectangle(
            vis_image,
            (bar_x1, bar_y),
            (bar_x1 + filled_width, h),
            color,
            -1
        )
        
        # Draw confidence text
        conf_text = f"{confidence:.1f}%"
        conf_size = cv2.getTextSize(conf_text, font, 0.8, 2)[0]
        conf_x = bar_x1 + (bar_width - conf_size[0]) // 2
        conf_y = h - 8
        
        cv2.putText(
            vis_image,
            conf_text,
            (conf_x, conf_y),
            font,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Encode to base64
        base64_img = encode_image_to_base64(vis_image, format='png')
        return base64_img
        
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        return None


def create_confidence_visualization(
    confidence: float,
    prediction: str = 'REAL',
    width: int = 300,
    height: int = 150
) -> Optional[str]:
    """
    Create a visual representation of confidence score.
    
    Args:
        confidence: Confidence percentage (0-100)
        prediction: 'REAL' or 'FAKE'
        width: Image width
        height: Image height
        
    Returns:
        Base64 encoded confidence visualization
    """
    try:
        # Create blank image
        image = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # Create color based on prediction
        if prediction == 'FAKE':
            color = (0, 0, 255)  # Red for fake
        else:
            color = (0, 255, 0)  # Green for real
        
        # Draw confidence bar
        bar_height = 40
        bar_y = height // 2 - bar_height // 2
        bar_x = 30
        bar_width = width - 60
        
        # Background rectangle
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
        
        # Filled rectangle based on confidence
        filled_width = int(bar_width * (confidence / 100.0))
        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + filled_width, bar_y + bar_height),
            color,
            -1
        )
        
        # Draw prediction text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{prediction}: {confidence:.2f}%"
        text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 40
        
        cv2.putText(image, text, (text_x, text_y), font, 1.0, color, 2)
        
        # Encode to base64
        return encode_image_to_base64(image, format='png')
        
    except Exception as e:
        logger.error(f"Error creating confidence visualization: {e}")
        return None


def validate_image_file(filename: str, allowed_extensions: list = None) -> bool:
    """
    Validate image file extension.
    
    Args:
        filename: Image filename
        allowed_extensions: List of allowed extensions
        
    Returns:
        True if valid, False otherwise
    """
    if allowed_extensions is None:
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
    
    _, ext = os.path.splitext(filename.lower())
    return ext in allowed_extensions


def get_image_properties(image_path: str) -> Optional[dict]:
    """
    Extract image properties and metadata.
    
    Args:
        image_path: Path to image
        
    Returns:
        Dictionary with image properties
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        h, w = image.shape[:2]
        
        # Check if grayscale or color
        if len(image.shape) == 2:
            channels = 1
        else:
            channels = image.shape[2]
        
        # Calculate file size
        file_size = os.path.getsize(image_path)
        
        return {
            'width': w,
            'height': h,
            'channels': channels,
            'file_size_bytes': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'aspect_ratio': round(w / h, 2)
        }
        
    except Exception as e:
        logger.error(f"Error getting image properties: {e}")
        return None


# Color definitions
COLORS = {
    'real': (0, 255, 0),      # Green (BGR)
    'fake': (0, 0, 255),       # Red (BGR)
    'uncertain': (0, 165, 255) # Orange (BGR)
}


def get_prediction_color(prediction: str, confidence: float = None) -> Tuple[int, int, int]:
    """
    Get color for prediction visualization.
    
    Args:
        prediction: 'REAL', 'FAKE', or 'UNCERTAIN'
        confidence: Confidence score (0-100), used for color intensity
        
    Returns:
        BGR color tuple
    """
    if prediction.upper() == 'FAKE':
        return COLORS['fake']
    elif prediction.upper() == 'REAL':
        return COLORS['real']
    else:
        return COLORS['uncertain']


import os
