from __future__ import annotations

import cv2
import numpy as np

from .config import DetectorConfig
from .structures import Detections


class ColorDetector:
    def __init__(self, cfg: DetectorConfig):
        self.low = np.asarray(cfg.hsv_low, np.uint8)
        self.high = np.asarray(cfg.hsv_high, np.uint8)
        self.min_area = cfg.min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def __call__(self, frame: np.ndarray) -> Detections:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an RGB image with shape (H, W, 3)")
        if frame.dtype != np.uint8:
            raise ValueError("frame must use uint8 pixels")
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        fg = cv2.inRange(hsv, self.low, self.high)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(fg)

        boxes, scores, masks = [], [], []
        h, w = frame.shape[:2]
        for label in range(1, count):
            x, y, bw, bh, area = stats[label]
            if area < self.min_area:
                continue
            boxes.append([x, y, x + bw, y + bh])
            scores.append(min(0.99, 0.55 + area / max(1, bw * bh)))
            masks.append(labels == label)
        if not boxes:
            return Detections.empty(h, w)
        return Detections(np.asarray(boxes, np.float32), np.asarray(scores, np.float32), np.stack(masks))
