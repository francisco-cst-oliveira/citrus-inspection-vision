from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .structures import Detections, box_iou


@dataclass
class Track:
    id: int
    box: np.ndarray
    score: float
    missed: int = 0
    hits: int = 1

    def update(self, box: np.ndarray, score: float) -> None:
        self.box = box.copy()
        self.score = score
        self.missed = 0
        self.hits += 1


class ByteTracker:
    def __init__(self, activation: float = 0.5, low: float = 0.15, match_iou: float = 0.2, buffer: int = 20):
        if not 0 <= low <= activation <= 1:
            raise ValueError("tracking thresholds must satisfy 0 <= low <= activation <= 1")
        if not 0 < match_iou <= 1:
            raise ValueError("match_iou must be in (0, 1]")
        if isinstance(buffer, bool) or not isinstance(buffer, int) or buffer < 0:
            raise ValueError("buffer must be a non-negative integer")
        self.activation = activation
        self.low = low
        self.match_iou = match_iou
        self.buffer = buffer
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections: Detections) -> Detections:
        high = np.flatnonzero(detections.confidence >= self.activation).tolist()
        low = np.flatnonzero(
            (detections.confidence >= self.low) & (detections.confidence < self.activation)
        ).tolist()
        unmatched_tracks = list(self.tracks)
        assigned: dict[int, int] = {}

        pairs, unmatched_tracks, high = self._match(unmatched_tracks, high, detections)
        self._apply(pairs, detections, assigned)

        pairs, unmatched_tracks, _ = self._match(unmatched_tracks, low, detections)
        self._apply(pairs, detections, assigned)

        for tid in unmatched_tracks:
            self.tracks[tid].missed += 1

        for idx in high:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(tid, detections.xyxy[idx].copy(), float(detections.confidence[idx]))
            assigned[idx] = tid

        self.tracks = {tid: track for tid, track in self.tracks.items() if track.missed <= self.buffer}
        idx = sorted(assigned)
        return detections.take(idx).with_tracker([assigned[i] for i in idx])

    def _match(
        self, track_ids: list[int], det_ids: list[int], detections: Detections
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not track_ids or not det_ids:
            return [], track_ids, det_ids
        boxes = np.stack([self.tracks[tid].box for tid in track_ids])
        iou = box_iou(boxes, detections.xyxy[det_ids])
        rows, cols = linear_sum_assignment(1 - iou)
        pairs = []
        used_tracks, used_det = set(), set()
        for row, col in zip(rows, cols, strict=True):
            if iou[row, col] < self.match_iou:
                continue
            tid, did = track_ids[row], det_ids[col]
            pairs.append((tid, did))
            used_tracks.add(tid)
            used_det.add(did)
        return (
            pairs,
            [x for x in track_ids if x not in used_tracks],
            [x for x in det_ids if x not in used_det],
        )

    def _apply(self, pairs: list[tuple[int, int]], detections: Detections, assigned: dict[int, int]) -> None:
        for tid, idx in pairs:
            self.tracks[tid].update(detections.xyxy[idx], float(detections.confidence[idx]))
            assigned[idx] = tid
