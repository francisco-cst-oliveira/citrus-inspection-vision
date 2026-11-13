import cv2
import numpy as np
import pytest

from citrus_inspection.config import Config
from citrus_inspection.model import Prediction
from citrus_inspection.runtime import resolve_source, run_live


class FixedScorer:
    threshold = 0.55

    def predict(self, images, masks):
        count = len(images)
        return Prediction(np.full(count, 0.2, np.float32), np.zeros((count, 32, 32), np.float32))


class FakeCapture:
    def __init__(self, frame):
        self.frame = frame
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        assert prop == cv2.CAP_PROP_FPS
        return 30

    def read(self):
        if self.frame is None:
            return False, None
        frame, self.frame = self.frame, None
        return True, frame

    def release(self):
        self.released = True


def test_resolve_source_accepts_camera_index():
    assert resolve_source(" 2 ") == (2, True)


def test_resolve_source_rejects_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_source(str(tmp_path / "missing.mp4"))


def test_live_runtime_releases_video_source(tmp_path, monkeypatch):
    path = tmp_path / "input.mp4"
    path.touch()
    frame = np.full((100, 120, 3), 40, np.uint8)
    frame[20:80, 30:90] = (35, 145, 235)
    capture = FakeCapture(frame)

    monkeypatch.setattr("citrus_inspection.runtime.cv2.VideoCapture", lambda _: capture)
    monkeypatch.setattr("citrus_inspection.runtime.cv2.namedWindow", lambda *_: None)
    monkeypatch.setattr("citrus_inspection.runtime.cv2.imshow", lambda *_: None)
    monkeypatch.setattr("citrus_inspection.runtime.cv2.waitKey", lambda _: -1)
    monkeypatch.setattr("citrus_inspection.runtime.cv2.getWindowProperty", lambda *_: 1)
    monkeypatch.setattr("citrus_inspection.runtime.cv2.destroyWindow", lambda *_: None)

    config = Config()
    config.detector.min_area = 100
    config.inspection.crop_size = 32
    config.inspection.min_views = 1
    result = run_live(config, FixedScorer(), str(path))

    assert result.frames == 1
    assert result.summary["normal"] == 1
    assert capture.released


def test_live_runtime_rejects_video_without_frames(tmp_path, monkeypatch):
    path = tmp_path / "empty.mp4"
    path.touch()
    capture = FakeCapture(None)

    monkeypatch.setattr("citrus_inspection.runtime.cv2.VideoCapture", lambda _: capture)
    monkeypatch.setattr("citrus_inspection.runtime.cv2.namedWindow", lambda *_: None)
    monkeypatch.setattr("citrus_inspection.runtime.cv2.destroyWindow", lambda *_: None)

    with pytest.raises(RuntimeError, match="no frames"):
        run_live(Config(), FixedScorer(), str(path))

    assert capture.released
