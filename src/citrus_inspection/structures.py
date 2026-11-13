from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class Detections:
    xyxy: np.ndarray
    confidence: np.ndarray
    mask: np.ndarray | None = None
    tracker_id: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.xyxy = np.asarray(self.xyxy, dtype=np.float32)
        self.confidence = np.asarray(self.confidence, dtype=np.float32)
        if self.xyxy.ndim != 2 or self.xyxy.shape[1:] != (4,):
            raise ValueError("xyxy must have shape (N, 4)")
        if self.confidence.shape != (len(self.xyxy),):
            raise ValueError("confidence must have shape (N,)")
        if not np.isfinite(self.xyxy).all() or not np.isfinite(self.confidence).all():
            raise ValueError("detections must contain finite values")
        if np.any(self.xyxy[:, 2:] < self.xyxy[:, :2]):
            raise ValueError("detection boxes must satisfy x2 >= x1 and y2 >= y1")
        if np.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("confidence values must be in [0, 1]")

        if self.mask is not None:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.ndim != 3 or len(self.mask) != len(self.xyxy):
                raise ValueError("mask must have shape (N, H, W)")
        if self.tracker_id is not None:
            self.tracker_id = np.asarray(self.tracker_id, dtype=np.int64)
            if self.tracker_id.shape != (len(self.xyxy),):
                raise ValueError("tracker_id must have shape (N,)")

    def __len__(self) -> int:
        return len(self.xyxy)

    @classmethod
    def empty(cls, height: int = 0, width: int = 0) -> Detections:
        masks = np.empty((0, height, width), bool) if height and width else None
        return cls(np.empty((0, 4), np.float32), np.empty(0, np.float32), masks)

    def take(self, idx: list[int] | np.ndarray) -> Detections:
        idx = np.asarray(idx, int)
        if idx.ndim != 1:
            raise ValueError("detection indices must be one-dimensional")
        masks = self.mask[idx] if self.mask is not None else None
        ids = self.tracker_id[idx] if self.tracker_id is not None else None
        return Detections(self.xyxy[idx], self.confidence[idx], masks, ids)

    def with_tracker(self, ids: list[int]) -> Detections:
        if len(ids) != len(self):
            raise ValueError("one tracker id is required for each detection")
        return replace(self, tracker_id=np.asarray(ids, np.int64))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _boxes(a, "a")
    b = _boxes(b, "b")
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _boxes(value: np.ndarray, name: str) -> np.ndarray:
    boxes = np.asarray(value, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"{name} must have shape (N, 4)")
    if not np.isfinite(boxes).all():
        raise ValueError(f"{name} must contain finite values")
    if np.any(boxes[:, 2:] < boxes[:, :2]):
        raise ValueError(f"{name} contains invalid boxes")
    return boxes
