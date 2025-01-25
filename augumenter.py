import cv2
import numpy as np

class Effect:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def run(self, img: np.ndarray):
        raise NotImplementedError("Subclasses must implement run method")

class NightVisionEffect(Effect):
    def run(self, img: np.ndarray) -> np.ndarray:
        """Apply night vision-like effect to image"""
        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gamma = self.kwargs.get('gamma', 2.0)
        inv_gamma = 1.0 / gamma
        
        # Create gamma correction lookup table
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                          for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction
        src_gray = cv2.LUT(src_gray, table)
        
        # Convert to 3-channel image
        return cv2.merge((src_gray, src_gray, src_gray))

class HistogramEqualizationEffect(Effect):
    def run(self, img: np.ndarray) -> np.ndarray:
        """Equalize histogram for each color channel"""
        b, g, r = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        return cv2.merge((b, g, r))

class HSVHistogramEqualizationEffect(Effect):
    def run(self, img: np.ndarray) -> np.ndarray:
        """Equalize histogram in HSV color space"""
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Equalize V channel
        v = cv2.equalizeHist(v)
        
        # Merge back
        result = cv2.merge((h, s, v))
        
        # Convert back to BGR
        return cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

class BrightnessContrastEffect(Effect):
    def run(self, img: np.ndarray) -> np.ndarray:
        """Adjust brightness and contrast"""
        brightness = self.kwargs.get('brightness', 0)
        contrast = self.kwargs.get('contrast', 1.0)
        
        # Create brightness/contrast adjusted image
        return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

def apply_effects(img: np.ndarray, effects: list) -> np.ndarray:
    """
    Apply multiple image augmentation effects
    
    Args:
        img: Input image
        effects: List of Effect objects to apply
    
    Returns:
        Augmented image
    """
    result = img.copy()
    for effect in effects:
        result = effect.run(result)
    return result
