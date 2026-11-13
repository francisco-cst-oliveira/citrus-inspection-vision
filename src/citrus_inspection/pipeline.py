from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np

from .config import Config
from .image import extract_crop
from .model import Scorer, validate_prediction
from .structures import Detections
from .tracking import ByteTracker


class Status(str, Enum):
    PENDING = "pending"
    REVIEW = "review"
    NORMAL = "normal"
    ANOMALY = "anomaly"


class Detector(Protocol):
    def __call__(self, frame: np.ndarray) -> Detections: ...


@dataclass
class View:
    score: float
    frame: int


@dataclass
class TrackRecord:
    track_id: int
    display_id: int
    first_frame: int
    last_frame: int
    views: list[View] = field(default_factory=list)
    status: Status = Status.PENDING
    score: float | None = None


@dataclass
class FrameResult:
    frame_idx: int
    detections: Detections
    statuses: list[Status]
    scores: list[float | None]
    display_ids: list[int]


class Inspector:
    def __init__(self, cfg: Config, detector: Detector, scorer: Scorer, fps: float = 30):
        cfg.validate()
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a positive finite number")
        if not 0 <= scorer.threshold <= 1:
            raise ValueError("scorer threshold must be in [0, 1]")
        self.cfg = cfg
        self.detector = detector
        self.scorer = scorer
        self.fps = fps
        tc = cfg.tracker
        self.tracker = ByteTracker(tc.activation_threshold, tc.low_threshold, tc.match_iou, tc.lost_buffer)
        self.tracks: dict[int, TrackRecord] = {}
        self.history: deque[TrackRecord] = deque(maxlen=cfg.inspection.history_limit)
        self.completed = {status: 0 for status in Status}
        self.frame_idx = -1
        self.next_display = 1

    def process(self, frame: np.ndarray) -> FrameResult:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (H, W, 3)")
        if frame.dtype != np.uint8:
            raise ValueError("frame must use uint8 pixels")
        self.frame_idx += 1
        detections = self.detector(frame)
        if detections.mask is not None and detections.mask.shape[1:] != frame.shape[:2]:
            raise ValueError("detection masks must match the frame size")
        det = self.tracker.update(detections)
        self._archive_expired()
        due = []
        for i in range(len(det)):
            tid = int(det.tracker_id[i])
            rec = self.tracks.get(tid)
            if rec is None:
                rec = TrackRecord(tid, self.next_display, self.frame_idx, self.frame_idx)
                self.tracks[tid] = rec
                self.next_display += 1
            rec.last_frame = self.frame_idx
            age = self.frame_idx - rec.first_frame
            ic = self.cfg.inspection
            if len(rec.views) >= ic.max_views or age % ic.score_every:
                continue
            mask = det.mask[i] if det.mask is not None else None
            crop, crop_mask, visible = extract_crop(frame, det.xyxy[i], mask, ic.crop_size, ic.crop_padding)
            if visible >= ic.min_visible_ratio:
                due.append((rec, crop, crop_mask))

        if due:
            pred = validate_prediction(
                self.scorer.predict([x[1] for x in due], [x[2] for x in due]), len(due)
            )
            for (rec, _, _), score in zip(due, pred.scores, strict=True):
                rec.views.append(View(float(score), self.frame_idx))
                self._decide(rec)

        statuses, scores, ids = [], [], []
        for tid in det.tracker_id if det.tracker_id is not None else []:
            rec = self.tracks[int(tid)]
            statuses.append(rec.status)
            scores.append(rec.score)
            ids.append(rec.display_id)
        return FrameResult(self.frame_idx, det, statuses, scores, ids)

    def _decide(self, rec: TrackRecord) -> None:
        values = sorted((x.score for x in rec.views), reverse=True)
        how = self.cfg.inspection.aggregation
        if how == "max":
            rec.score = values[0]
        elif how == "mean":
            rec.score = float(np.mean(values))
        elif how == "top2_mean":
            rec.score = float(np.mean(values[:2]))
        else:
            raise ValueError(f"unknown aggregation: {how}")
        if len(values) < self.cfg.inspection.min_views:
            rec.status = Status.REVIEW
        elif rec.status != Status.ANOMALY:
            rec.status = Status.ANOMALY if rec.score >= self.scorer.threshold else Status.NORMAL

    def _archive_expired(self) -> None:
        active = set(self.tracker.tracks)
        for track_id, record in list(self.tracks.items()):
            if track_id in active:
                continue
            self.completed[record.status] += 1
            self.history.append(record)
            del self.tracks[track_id]

    def summary(self) -> dict[str, int]:
        values = list(self.tracks.values())
        normal = self.completed[Status.NORMAL] + sum(x.status == Status.NORMAL for x in values)
        anomalies = self.completed[Status.ANOMALY] + sum(x.status == Status.ANOMALY for x in values)
        return {
            "inspected": normal + anomalies,
            "normal": normal,
            "anomalies": anomalies,
            "in_review": self.completed[Status.REVIEW] + sum(x.status == Status.REVIEW for x in values),
        }

    def report(self) -> list[dict]:
        records = [*self.history, *self.tracks.values()]
        return [
            {
                "item": rec.display_id,
                "track_id": rec.track_id,
                "status": rec.status.value,
                "score": rec.score,
                "views": len(rec.views),
                "first_seen_s": round(rec.first_frame / self.fps, 2),
                "last_seen_s": round(rec.last_frame / self.fps, 2),
            }
            for rec in sorted(records, key=lambda x: x.display_id)
        ]
