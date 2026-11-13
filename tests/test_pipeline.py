import numpy as np

from citrus_inspection.config import Config
from citrus_inspection.model import Prediction
from citrus_inspection.pipeline import Inspector, Status
from citrus_inspection.structures import Detections


class FixedDetector:
    def __call__(self, frame):
        mask = np.zeros(frame.shape[:2], bool)
        mask[20:100, 30:110] = True
        return Detections(np.array([[30, 20, 110, 100]], np.float32), np.array([0.9]), mask[None])


class FixedScorer:
    threshold = 0.55

    def __init__(self, score: float):
        self.score = score

    def predict(self, images, masks):
        scores = np.full(len(images), self.score, np.float32)
        maps = np.zeros((len(images), *masks[0].shape), np.float32)
        return Prediction(scores, maps)


class SequenceDetector:
    def __init__(self):
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        if self.calls == 1:
            return FixedDetector()(frame)
        return Detections.empty(*frame.shape[:2])


def test_pipeline_aggregates_multiple_views():
    cfg = Config()
    cfg.inspection.score_every = 1
    cfg.inspection.min_views = 2
    inspector = Inspector(cfg, FixedDetector(), FixedScorer(0.2))
    frame = np.full((140, 160, 3), (235, 145, 35), np.uint8)
    for _ in range(3):
        result = inspector.process(frame)
    assert result.statuses == [Status.NORMAL]
    assert inspector.summary()["inspected"] == 1


def test_high_score_is_flagged():
    cfg = Config()
    cfg.inspection.score_every = 1
    cfg.inspection.min_views = 2
    inspector = Inspector(cfg, FixedDetector(), FixedScorer(0.8))
    frame = np.full((140, 160, 3), (235, 145, 35), np.uint8)
    for _ in range(3):
        result = inspector.process(frame)
    assert result.statuses == [Status.ANOMALY]


def test_expired_tracks_are_archived_without_losing_totals():
    cfg = Config()
    cfg.tracker.lost_buffer = 0
    cfg.inspection.score_every = 1
    cfg.inspection.min_views = 1
    inspector = Inspector(cfg, SequenceDetector(), FixedScorer(0.2))
    frame = np.full((140, 160, 3), (235, 145, 35), np.uint8)

    inspector.process(frame)
    inspector.process(frame)

    assert not inspector.tracks
    assert len(inspector.history) == 1
    assert inspector.summary()["normal"] == 1
