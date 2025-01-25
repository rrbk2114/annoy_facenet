import cv2
import numpy as np
import dlib

class FaceDetector:
    def __init__(self, 
                 ssd_model_file="models/res10_300x300_ssd_iter_140000_fp16.caffemodel", 
                 ssd_config_file="models/deploy.prototxt"):
        """
        Initialize face detector with SSD model
        
        Args:
            ssd_model_file: Path to pre-trained SSD model weights
            ssd_config_file: Path to SSD model configuration
        """
        self.conf_threshold = 0.6
        try:
            self.net = cv2.dnn.readNetFromCaffe(ssd_config_file, ssd_model_file)
        except Exception as e:
            print(f"Error loading SSD model: {e}")
            raise

    def detect(self, img: np.ndarray) -> list:
        """
        Detect faces in the input image
        
        Args:
            img: Input image as NumPy array
        
        Returns:
            List of detected face rectangles
        """
        # Histogram equalization for improved detection
        b, g, r = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img_he = cv2.merge((b, g, r))

        # Prepare image for detection
        blob = cv2.dnn.blobFromImage(img_he, 1.0, (300, 300), 
                                     [104, 117, 123], False, False)
        self.net.setInput(blob)

        # Detect faces
        frameHeight, frameWidth = img.shape[:2]
        detections = self.net.forward()

        rects = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                # Convert coordinates
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                
                rects.append(dlib.rectangle(x1, y1, x2, y2))

        return rects

    def draw_detections(self, img: np.ndarray) -> np.ndarray:
        """
        Draw bounding boxes around detected faces
        
        Args:
            img: Input image
        
        Returns:
            Image with face detections drawn
        """
        detections = self.detect(img)
        for rect in detections:
            x1, y1 = rect.left(), rect.top()
            x2, y2 = rect.right(), rect.bottom()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img
