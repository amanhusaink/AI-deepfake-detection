"""
Rule-Based Deepfake Detector
Uses image forensics techniques to detect manipulation artifacts without ML training.
This provides consistent, explainable predictions based on forensic analysis.
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn


class ForensicDeepfakeDetector:
    """
    Detects deepfakes using multiple forensic analysis techniques:
    1. Frequency domain analysis (FFT)
    2. Edge consistency
    3. Color distribution
    4. Noise patterns
    5. Compression artifacts
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_frequency(self, img_gray):
        """Analyze frequency domain for manipulation artifacts"""
        f_transform = np.fft.fft2(img_gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1e-6)
        
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Check for unusual high-frequency patterns
        mask_size = min(h, w) // 4
        low_freq = magnitude_spectrum[center_h-mask_size:center_h+mask_size, 
                                       center_w-mask_size:center_w+mask_size]
        high_freq = magnitude_spectrum.copy()
        high_freq[center_h-mask_size:center_h+mask_size, 
                  center_w-mask_size:center_w+mask_size] = 0
        
        ratio = np.sum(high_freq) / (np.sum(low_freq) + 1e-6)
        return min(1.0, ratio / 8.0)
    
    def analyze_edges(self, img_gray):
        """Check edge consistency - deepfakes often have inconsistent edges"""
        edges = cv2.Canny(img_gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Real images typically have 5-15% edge density
        if 0.05 <= edge_density <= 0.15:
            return 0.2  # Low fake score
        elif edge_density < 0.02 or edge_density > 0.25:
            return 0.8  # High fake score
        else:
            return 0.5
    
    def analyze_noise(self, img):
        """Analyze noise patterns - deepfakes have different noise characteristics"""
        denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        noise = cv2.absdiff(img, denoised)
        noise_level = np.mean(cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY))
        
        # Very low or very high noise can indicate manipulation
        if noise_level < 5 or noise_level > 30:
            return 0.7
        else:
            return 0.3
    
    def analyze_color(self, img):
        """Analyze color distribution inconsistencies"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        a_std = np.std(lab[:,:,1])
        b_std = np.std(lab[:,:,2])
        
        # Unusual color distributions can indicate GAN artifacts
        color_variation = (a_std + b_std) / 2
        if color_variation < 15 or color_variation > 60:
            return 0.6
        else:
            return 0.3
    
    def analyze_compression(self, img):
        """Check for double compression artifacts"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate local variance
        kernel = np.ones((3,3), np.float32) / 9
        mean = cv2.filter2D(gray.astype(float), -1, kernel)
        variance = cv2.filter2D((gray.astype(float) - mean)**2, -1, kernel)
        
        # High variance uniformity can indicate tampering
        var_std = np.std(variance)
        if var_std < 50:
            return 0.4
        else:
            return 0.6
    
    def predict(self, image_path: str) -> tuple:
        """
        Predict whether image is real or fake.
        
        Returns:
            Tuple of (prediction_label, confidence_score)
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Perform all analyses
            fft_score = self.analyze_frequency(img_gray)
            edge_score = self.analyze_edges(img_gray)
            noise_score = self.analyze_noise(img)
            color_score = self.analyze_color(img)
            compression_score = self.analyze_compression(img)
            
            # Weighted combination
            weights = [0.30, 0.20, 0.20, 0.15, 0.15]  # FFT, Edges, Noise, Color, Compression
            fake_confidence = (weights[0] * fft_score + 
                              weights[1] * edge_score +
                              weights[2] * noise_score +
                              weights[3] * color_score +
                              weights[4] * compression_score)
            
            # Determine prediction with threshold
            # Lower threshold (0.42) makes it more sensitive to potential fakes
            threshold = 0.42
            
            if fake_confidence >= threshold:
                prediction = "Fake"
                confidence = max(55.0, min(95.0, fake_confidence * 100))
            else:
                prediction = "Real"
                confidence = max(55.0, min(95.0, (1 - fake_confidence) * 100))
            
            print(f"[Forensic Analysis] Fake confidence: {fake_confidence:.3f} -> {prediction} ({confidence:.1f}%)")
            
            return prediction, round(confidence, 2)
            
        except Exception as e:
            print(f"Error in forensic analysis: {e}")
            # Conservative fallback - lean towards Real when uncertain
            return "Real", 50.0


# Monkey-patch the backend model handler to use forensic analysis
def patched_predict(self, image_path: str):
    """Override predict method to use forensic analysis instead of random"""
    detector = ForensicDeepfakeDetector()
    return detector.predict(image_path)


if __name__ == "__main__":
    # Test on sample images
    import sys
    from pathlib import Path
    
    test_dir = Path("data/images")
    detector = ForensicDeepfakeDetector()
    
    print("Testing Forensic Deepfake Detector")
    print("="*60)
    
    results = {"Real": 0, "Fake": 0}
    
    for img_path in list(test_dir.glob("*.png"))[:10]:
        prediction, confidence = detector.predict(str(img_path))
        results[prediction] += 1
        print(f"{img_path.name}: {prediction} ({confidence:.1f}%)")
    
    print("\n" + "="*60)
    print(f"Results: {results}")
