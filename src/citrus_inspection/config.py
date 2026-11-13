from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RuntimeConfig:
    threshold: float = 0.55
    device: str = "cpu"


@dataclass
class DetectorConfig:
    hsv_low: list[int] = field(default_factory=lambda: [0, 70, 50])
    hsv_high: list[int] = field(default_factory=lambda: [35, 255, 255])
    min_area: int = 900


@dataclass
class TrackerConfig:
    activation_threshold: float = 0.5
    low_threshold: float = 0.15
    match_iou: float = 0.2
    lost_buffer: int = 20


@dataclass
class InspectionConfig:
    crop_size: int = 224
    score_every: int = 3
    max_views: int = 6
    min_views: int = 2
    aggregation: str = "top2_mean"
    crop_padding: float = 0.06
    min_visible_ratio: float = 0.85
    history_limit: int = 1000


@dataclass
class Config:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    inspection: InspectionConfig = field(default_factory=InspectionConfig)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
        try:
            with path.open(encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"config root must be a mapping: {path}")
        config = _build(cls, raw)
        config.validate()
        return config

    def validate(self) -> None:
        _ratio("runtime.threshold", self.runtime.threshold)
        if not isinstance(self.runtime.device, str) or not self.runtime.device.strip():
            raise ValueError("runtime.device must be a non-empty string")

        _hsv("detector.hsv_low", self.detector.hsv_low)
        _hsv("detector.hsv_high", self.detector.hsv_high)
        if any(low > high for low, high in zip(self.detector.hsv_low, self.detector.hsv_high, strict=True)):
            raise ValueError("detector.hsv_low must not exceed detector.hsv_high")
        _positive_int("detector.min_area", self.detector.min_area)

        _ratio("tracker.activation_threshold", self.tracker.activation_threshold)
        _ratio("tracker.low_threshold", self.tracker.low_threshold)
        _ratio("tracker.match_iou", self.tracker.match_iou, allow_zero=False)
        if self.tracker.low_threshold > self.tracker.activation_threshold:
            raise ValueError("tracker.low_threshold must not exceed tracker.activation_threshold")
        _non_negative_int("tracker.lost_buffer", self.tracker.lost_buffer)

        _positive_int("inspection.crop_size", self.inspection.crop_size)
        _positive_int("inspection.score_every", self.inspection.score_every)
        _positive_int("inspection.max_views", self.inspection.max_views)
        _positive_int("inspection.min_views", self.inspection.min_views)
        if self.inspection.min_views > self.inspection.max_views:
            raise ValueError("inspection.min_views must not exceed inspection.max_views")
        if self.inspection.aggregation not in {"max", "mean", "top2_mean"}:
            raise ValueError("inspection.aggregation must be max, mean or top2_mean")
        _ratio("inspection.crop_padding", self.inspection.crop_padding)
        _ratio("inspection.min_visible_ratio", self.inspection.min_visible_ratio)
        _non_negative_int("inspection.history_limit", self.inspection.history_limit)


def _build(cls: type, raw: dict[str, Any]):
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"unknown config keys in {cls.__name__}: {sorted(unknown)}")
    defaults = cls()
    values = {}
    for name, value in raw.items():
        default = getattr(defaults, name)
        values[name] = _build(type(default), value) if is_dataclass(default) else value
    return cls(**values)


def _ratio(name: str, value: float, allow_zero: bool = True) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if value < 0 or value > 1 or (not allow_zero and value == 0):
        interval = "(0, 1]" if not allow_zero else "[0, 1]"
        raise ValueError(f"{name} must be in {interval}")


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _hsv(name: str, value: list[int]) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three integers")
    if any(
        isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255
        for channel in value
    ):
        raise ValueError(f"{name} channels must be integers from 0 to 255")
