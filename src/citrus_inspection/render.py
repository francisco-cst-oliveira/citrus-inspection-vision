from __future__ import annotations

import cv2
import numpy as np

from .pipeline import FrameResult, Inspector, Status

COLORS = {
    Status.PENDING: (150, 150, 150),
    Status.REVIEW: (235, 190, 45),
    Status.NORMAL: (55, 205, 130),
    Status.ANOMALY: (245, 75, 75),
}


def dashboard(frame: np.ndarray, result: FrameResult, inspector: Inspector) -> np.ndarray:
    view = frame.copy()
    overlay = view.copy()
    det = result.detections
    for i, status in enumerate(result.statuses):
        color = COLORS[status]
        if det.mask is not None:
            overlay[det.mask[i]] = color
        x1, y1, x2, y2 = det.xyxy[i].astype(int)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        score = result.scores[i]
        label = f"#{result.display_ids[i]} {status.value}"
        if score is not None:
            label += f" {score:.2f}"
        cv2.putText(view, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    view = cv2.addWeighted(overlay, 0.18, view, 0.82, 0)

    panel = np.full((frame.shape[0], 320, 3), 18, np.uint8)
    stats = inspector.summary()
    cv2.putText(panel, "LINE INSPECTION", (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2)
    rows = [
        ("Inspected", stats["inspected"], (235, 235, 235)),
        ("Normal", stats["normal"], COLORS[Status.NORMAL]),
        ("Anomalies", stats["anomalies"], COLORS[Status.ANOMALY]),
        ("In review", stats["in_review"], COLORS[Status.REVIEW]),
    ]
    for i, (label, value, color) in enumerate(rows):
        y = 105 + i * 72
        cv2.putText(panel, label.upper(), (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (135, 145, 155), 1)
        cv2.putText(panel, str(value), (22, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 2)
    rate = 100 * stats["anomalies"] / max(1, stats["inspected"])
    cv2.putText(panel, "ANOMALY RATE", (22, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (135, 145, 155), 1)
    cv2.putText(panel, f"{rate:.1f}%", (22, 448), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (235, 235, 235), 2)
    return np.hstack([view, panel])
