import cv2
import numpy as np

class Effect:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def run(self, img: np.array):
        raise NotImplementedError()

class NightVisionEffect(Effect):
    def run(self, img: np.array):
        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        invGamma = 1.0 / self.kwargs.get('gamma', 2.0)
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        src_gray = cv2.LUT(src_gray, table)
        return cv2.merge((src_gray, src_gray, src_gray))

class HistogramEqualizationEffect(Effect):
    def run(self, img: np.array):
        b, g, r = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        return cv2.merge((b, g, r))

class HsvHistogramEqualizationEffect(Effect):
    def run(self, img: np.array):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h = cv2.equalizeHist(h)
        s = cv2.equalizeHist(s)
        v = cv2.equalizeHist(v)
        result = cv2.merge((h, s, v))
        return cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

