from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep

import cv2
import numpy as np

from .config import Config
from .detection import ColorDetector
from .model import Scorer
from .pipeline import Inspector
from .render import dashboard

WINDOW_NAME = "Citrus Inspection"


@dataclass(frozen=True)
class RunResult:
    frames: int
    elapsed_seconds: float
    stopped_by_user: bool
    summary: dict[str, int]

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "frames": self.frames,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "stopped_by_user": self.stopped_by_user,
            **self.summary,
        }


def run_live(cfg: Config, scorer: Scorer, source: str) -> RunResult:
    capture_source, is_camera = resolve_source(source)
    try:
        capture = cv2.VideoCapture(capture_source)
    except cv2.error as exc:
        raise RuntimeError(f"could not open video source: {source}") from exc
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open video source: {source}")

    fps = _capture_fps(capture)
    started = perf_counter()
    frames = 0
    failures = 0
    stopped_by_user = False
    window_open = False

    try:
        inspector = Inspector(cfg, ColorDetector(cfg.detector), scorer, fps)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        window_open = True
        while True:
            ok, bgr = capture.read()
            if not ok:
                if not is_camera:
                    break
                failures += 1
                if failures >= 3:
                    raise RuntimeError("camera stopped returning frames")
                sleep(0.05)
                continue

            failures = 0
            if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
                raise RuntimeError("video source returned an invalid frame")
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = inspector.process(frame)
            canvas = dashboard(frame, result, inspector)
            cv2.imshow(WINDOW_NAME, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            frames += 1

            key = cv2.waitKey(_wait_time(started, frames, fps, is_camera)) & 0xFF
            if (
                key in (27, ord("q"), ord("Q"))
                or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            ):
                stopped_by_user = True
                break
    except cv2.error as exc:
        raise RuntimeError("OpenCV could not create or update the inspection window") from exc
    finally:
        capture.release()
        if window_open:
            with suppress(cv2.error):
                cv2.destroyWindow(WINDOW_NAME)

    if frames == 0:
        raise RuntimeError(f"video source returned no frames: {source}")
    return RunResult(frames, perf_counter() - started, stopped_by_user, inspector.summary())


def resolve_source(source: str) -> tuple[int | str, bool]:
    value = source.strip()
    if not value:
        raise ValueError("video source cannot be empty")
    if value.isdigit():
        return int(value), True
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"video file not found: {path}")
    return str(path.resolve()), False


def _capture_fps(capture: cv2.VideoCapture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    return fps if np.isfinite(fps) and 1 <= fps <= 240 else 30.0


def _wait_time(started: float, frames: int, fps: float, is_camera: bool) -> int:
    if is_camera:
        return 1
    remaining = started + frames / fps - perf_counter()
    return max(1, min(1000, round(remaining * 1000)))
