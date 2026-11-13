import numpy as np

from citrus_inspection.structures import Detections, box_iou
from citrus_inspection.tracking import ByteTracker


def detections(boxes, scores):
    return Detections(np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32))


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    np.testing.assert_allclose(box_iou(a, b)[0], [1, 1 / 3, 0], atol=1e-6)


def test_ids_remain_stable():
    tracker = ByteTracker()
    seen = []
    for step in range(12):
        output = tracker.update(
            detections(
                [[20 + 3 * step, 20, 80 + 3 * step, 80], [220 - 2 * step, 120, 280 - 2 * step, 180]],
                [0.9, 0.85],
            )
        )
        seen.append(tuple(output.tracker_id))
    assert len(set(seen)) == 1


def test_low_confidence_detection_keeps_track():
    tracker = ByteTracker(activation=0.6)
    first = tracker.update(detections([[10, 10, 60, 60]], [0.9]))
    track_id = first.tracker_id[0]
    output = tracker.update(detections([[12, 10, 62, 60]], [0.3]))
    assert output.tracker_id.tolist() == [track_id]


def test_zero_buffer_keeps_continuous_track():
    tracker = ByteTracker(buffer=0)
    first = tracker.update(detections([[10, 10, 60, 60]], [0.9]))
    second = tracker.update(detections([[12, 10, 62, 60]], [0.9]))

    assert second.tracker_id.tolist() == first.tracker_id.tolist()
