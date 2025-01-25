import cv2
import numpy as np
import dlib

class FaceDetector:
    def __init__(self, 
                 ssd_model_file="models/res10_300x300_ssd_iter_140000_fp16.caffemodel", 
                 ssd_config_file="models/deploy.prototxt"):
        self.conf_threshold = 0.63
        self.net = cv2.dnn.readNetFromCaffe(ssd_config_file, ssd_model_file)

    def detect(self, img: np.array):
        """
        Detects faces in the given image using SSD model.
        :param img: Input image as a NumPy array.
        :return: List of bounding rectangles (dlib.rectangle).
        """
        b, g, r = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img_he = cv2.merge((b, g, r))

        blob = cv2.dnn.blobFromImage(img_he, 1.0, (300, 300), [104, 117, 123], False, False)
        self.net.setInput(blob)

        frameHeight, frameWidth = img.shape[:2]
        detections = self.net.forward()

        rects = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                rects.append(dlib.rectangle(x1, y1, x2, y2))

        return rects

