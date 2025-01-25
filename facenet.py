import cv2
import numpy as np
import dlib
import os
from typing import List, Optional

class FaceDetector:
    def __init__(self, 
                 ssd_model_file: Optional[str] = None, 
                 ssd_config_file: Optional[str] = None, 
                 conf_threshold: float = 0.6):
        """
        Initialize face detector with configurable SSD model.
        
        Args:
            ssd_model_file: Path to pre-trained SSD model weights
            ssd_config_file: Path to SSD model configuration
            conf_threshold: Confidence threshold for face detection
        """
        # Default model paths if not provided
        if ssd_model_file is None:
            ssd_model_file = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        if ssd_config_file is None:
            ssd_config_file = "models/deploy.prototxt"
        
        # Validate model files exist
        if not os.path.exists(ssd_model_file):
            raise FileNotFoundError(f"SSD model file not found: {ssd_model_file}")
        if not os.path.exists(ssd_config_file):
            raise FileNotFoundError(f"SSD config file not found: {ssd_config_file}")
        
        self.conf_threshold = conf_threshold
        
        try:
            # Use more robust network loading
            self.net = cv2.dnn.readNet(ssd_config_file, ssd_model_file)
        except Exception as e:
            print(f"Error loading SSD model: {e}")
            raise

    def detect(self, img: np.ndarray) -> List[dlib.rectangle]:
        """
        Detect faces in the input image with advanced preprocessing.
        
        Args:
            img: Input image as NumPy array
        
        Returns:
            List of detected face rectangles
        """
        # Validate input image
        if img is None or img.size == 0:
            print("Warning: Empty or invalid input image")
            return []

        # Advanced histogram equalization for improved detection
        try:
            b, g, r = cv2.split(img)
            b = cv2.equalizeHist(b)
            g = cv2.equalizeHist(g)
            r = cv2.equalizeHist(r)
            img_he = cv2.merge((b, g, r))
        except Exception as e:
            print(f"Histogram equalization failed: {e}")
            img_he = img  # Fallback to original image

        # Prepare image for detection
        try:
            blob = cv2.dnn.blobFromImage(img_he, 1.0, (300, 300), 
                                         [104, 117, 123], False, False)
            self.net.setInput(blob)
        except Exception as e:
            print(f"Image preprocessing failed: {e}")
            return []

        # Detect faces
        try:
            frameHeight, frameWidth = img.shape[:2]
            detections = self.net.forward()
        except Exception as e:
            print(f"Face detection failed: {e}")
            return []

        rects = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                # Convert coordinates
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                
                # Ensure valid rectangle coordinates
                if x1 < x2 and y1 < y2:
                    rects.append(dlib.rectangle(x1, y1, x2, y2))

        return rects

    def draw_detections(self, img: np.ndarray, 
                        color: tuple = (0, 255, 0), 
                        thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.
        
        Args:
            img: Input image
            color: Bounding box color (B,G,R)
            thickness: Bounding box line thickness
        
        Returns:
            Image with face detections drawn
        """
        # Create a copy to avoid modifying original
        output_img = img.copy()
        
        # Detect faces
        detections = self.detect(img)
        
        for rect in detections:
            x1, y1 = rect.left(), rect.top()
            x2, y2 = rect.right(), rect.bottom()
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, thickness)
        
        return output_img
