from __future__ import annotations

import cv2
import numpy as np


def extract_crop(
    frame: np.ndarray, box: np.ndarray, mask: np.ndarray | None, size: int, padding: float
) -> tuple[np.ndarray, np.ndarray, float]:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (H, W, 3)")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("crop size must be a positive integer")
    if not 0 <= padding <= 1:
        raise ValueError("crop padding must be in [0, 1]")
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("box must contain four finite coordinates")
    if mask is not None and np.asarray(mask).shape != frame.shape[:2]:
        raise ValueError("mask shape must match the frame")

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive width and height")
    pad = padding * max(x2 - x1, y2 - y1)
    raw = np.array([x1 - pad, y1 - pad, x2 + pad, y2 + pad])
    clipped = np.array([max(0, raw[0]), max(0, raw[1]), min(w, raw[2]), min(h, raw[3])])
    visible = np.prod(clipped[2:] - clipped[:2]) / max(1, np.prod(raw[2:] - raw[:2]))
    xa, ya, xb, yb = np.rint(clipped).astype(int)
    if xb <= xa or yb <= ya:
        empty_image = np.zeros((size, size, frame.shape[2]), frame.dtype)
        return empty_image, np.zeros((size, size), np.uint8), 0.0
    crop = frame[ya:yb, xa:xb]
    crop_mask = (
        mask[ya:yb, xa:xb].astype(np.uint8) * 255
        if mask is not None
        else np.full(crop.shape[:2], 255, np.uint8)
    )
    return _letterbox(crop, size), _letterbox(crop_mask, size, nearest=True), float(visible)


def _letterbox(image: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    out_w, out_h = max(1, round(w * scale)), max(1, round(h * scale))
    mode = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    resized = cv2.resize(image, (out_w, out_h), interpolation=mode)
    shape = (size, size) if image.ndim == 2 else (size, size, image.shape[2])
    out = np.zeros(shape, image.dtype)
    x, y = (size - out_w) // 2, (size - out_h) // 2
    out[y : y + out_h, x : x + out_w] = resized
    return out
