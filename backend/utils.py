"""
Utility functions for Deepfake Detection System
Includes image preprocessing, heatmap generation, and helper utilities.
"""

import cv2
import numpy as np
import torch
from PIL import Image
import io
import base64
from typing import Optional, Tuple
import os


def preprocess_image_for_detection(image_path: str, img_size: int = 224) -> np.ndarray:
    """
    Preprocess image for model inference using OpenCV.
    
    Args:
        image_path: Path to image file
        img_size: Target size for resizing
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, (img_size, img_size))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image


def generate_heatmap_gradcam(
    model: torch.nn.Module,
    image_path: str,
    target_layer: Optional[str] = None,
    use_cuda: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM heatmap for image to highlight manipulated regions.
    
    Args:
        model: PyTorch model
        image_path: Path to image file
        target_layer: Name of target layer for Grad-CAM (default: last conv layer)
        use_cuda: Whether to use GPU
        
    Returns:
        Tuple of (original_image, heatmap_overlay)
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils import reshape_transform
    except ImportError:
        print("Warning: grad-cam package not installed. Skipping heatmap generation.")
        # Return original image without heatmap
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize for visualization
    input_tensor = cv2.resize(rgb_image, (224, 224))
    input_tensor = input_tensor.astype(np.float32) / 255.0
    input_tensor = (input_tensor - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).unsqueeze(0)
    
    if use_cuda:
        input_tensor = input_tensor.cuda()
    
    # Set up GradCAM
    # For ResNet, use layer4[-1] as target layer
    if target_layer is None:
        target_layer = model.resnet.layer4[-1]
    
    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        use_cuda=use_cuda,
        reshape_transform=reshape_transform
    )
    
    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    
    # Resize heatmap to match original image size
    heatmap = cv2.resize(grayscale_cam, (rgb_image.shape[1], rgb_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap (JET color map)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend heatmap with original image
    original_resized = cv2.resize(rgb_image, (heatmap_colored.shape[1], heatmap_colored.shape[0]))
    overlay = cv2.addWeighted(heatmap_colored, 0.5, original_resized, 0.5, 0)
    
    return original_resized, overlay


def encode_image_to_base64(image_array: np.ndarray, format: str = 'JPEG') -> str:
    """
    Encode numpy array image to base64 string.
    
    Args:
        image_array: Image as numpy array (RGB format)
        format: Image format (JPEG, PNG, etc.)
        
    Returns:
        Base64 encoded string
    """
    # Convert numpy array to PIL Image
    if image_array.dtype == np.uint8:
        pil_image = Image.fromarray(image_array)
    else:
        # Normalize to [0, 255] if needed
        image_normalized = np.uint8((image_array - image_array.min()) / 
                                   (image_array.max() - image_array.min()) * 255)
        pil_image = Image.fromarray(image_normalized)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)
    
    # Encode to base64
    image_bytes = buffer.getvalue()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    
    return base64_string


def decode_base64_to_image(base64_string: str) -> Image.Image:
    """
    Decode base64 string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image object
    """
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    return image


def save_uploaded_file(file_content: bytes, upload_dir: str, filename: str) -> str:
    """
    Save uploaded file to directory.
    
    Args:
        file_content: File content as bytes
        upload_dir: Directory to save file
        filename: Filename
        
    Returns:
        Full path to saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save file
    file_path = os.path.join(upload_dir, filename)
    with open(file_path, 'wb') as f:
        f.write(file_content)
    
    return file_path


def validate_image_file(file_path: str, max_size_mb: float = 10.0) -> bool:
    """
    Validate image file type and size.
    
    Args:
        file_path: Path to image file
        max_size_mb: Maximum file size in MB
        
    Returns:
        True if valid, raises exception otherwise
    """
    # Check file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext not in allowed_extensions:
        raise ValueError(f"Invalid file type: {ext}. Allowed: {allowed_extensions}")
    
    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(f"File too large: {file_size_mb:.2f}MB. Max: {max_size_mb}MB")
    
    # Try to open image
    try:
        img = Image.open(file_path)
        img.verify()  # Verify it's a valid image
        return True
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")


def calculate_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    """
    Calculate classification metrics.
    
    Args:
        tp: True positives
        tn: True negatives
        fp: False positives
        fn: False negatives
        
    Returns:
        Dictionary with accuracy, precision, recall, f1
    """
    total = tp + tn + fp + fn
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_samples': total
    }


def format_confidence(confidence: float, decimals: int = 2) -> str:
    """
    Format confidence score for display.
    
    Args:
        confidence: Confidence value (0-100)
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{confidence:.{decimals}f}%"


def get_model_info(model_path: Optional[str] = None) -> dict:
    """
    Get information about model.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Dictionary with model info
    """
    info = {
        'model_loaded': False,
        'model_path': model_path,
        'cuda_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if model_path and os.path.exists(model_path):
        info['model_loaded'] = True
        info['model_size_mb'] = os.path.getsize(model_path) / (1024 * 1024)
    
    return info


# Color constants for UI
COLORS = {
    'real': '#10B981',      # Green
    'fake': '#EF4444',      # Red
    'human': '#10B981',     # Green
    'ai': '#EF4444',        # Red
    'neutral': '#6B7280',   # Gray
    'primary': '#4F46E5'    # Indigo
}


def get_prediction_color(prediction: str) -> str:
    """
    Get color for prediction label.
    
    Args:
        prediction: Prediction label
        
    Returns:
        Hex color code
    """
    prediction_lower = prediction.lower()
    
    if 'fake' in prediction_lower or 'ai' in prediction_lower:
        return COLORS['fake']
    elif 'real' in prediction_lower or 'human' in prediction_lower:
        return COLORS['human']
    else:
        return COLORS['neutral']
