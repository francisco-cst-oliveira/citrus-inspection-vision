from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class Prediction:
    scores: np.ndarray
    maps: np.ndarray


class Scorer(Protocol):
    threshold: float

    def predict(self, images: list[np.ndarray], masks: list[np.ndarray]) -> Prediction: ...


class TorchScriptScorer:
    def __init__(self, path: str | Path, threshold: float, device: str = "cpu"):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('PyTorch is required; install the project with ".[inference]"') from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")

        self.torch = torch
        self.device = device
        self.threshold = float(threshold)
        try:
            self.model = torch.jit.load(str(path), map_location=device).eval()
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(f"could not load TorchScript model: {path}") from exc

    def predict(self, images: list[np.ndarray], masks: list[np.ndarray]) -> Prediction:
        _validate_batch(images, masks)
        x = np.stack(images).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        m = (np.stack(masks)[:, None] > 0).astype(np.float32)
        with self.torch.inference_mode():
            output = self.model(
                self.torch.from_numpy(x).to(self.device), self.torch.from_numpy(m).to(self.device)
            )
        scores, maps = _unpack(output)
        prediction = Prediction(_numpy(scores), _numpy(maps))
        return validate_prediction(prediction, len(images))


def validate_prediction(prediction: Prediction, batch_size: int) -> Prediction:
    scores = np.asarray(prediction.scores, dtype=np.float32)
    maps = np.asarray(prediction.maps, dtype=np.float32)
    if scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores[:, 0]
    if scores.shape != (batch_size,):
        raise ValueError(f"model scores must have shape ({batch_size},)")
    if maps.ndim == 4 and maps.shape[1] == 1:
        maps = maps[:, 0]
    if maps.ndim != 3 or maps.shape[0] != batch_size:
        raise ValueError(f"model maps must have shape ({batch_size}, H, W)")
    if not np.isfinite(scores).all() or not np.isfinite(maps).all():
        raise ValueError("model output contains non-finite values")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("model scores must be in [0, 1]")
    return Prediction(scores, maps)


def _validate_batch(images: list[np.ndarray], masks: list[np.ndarray]) -> None:
    if not images or len(images) != len(masks):
        raise ValueError("images and masks must contain the same non-zero number of items")
    image_shape = np.asarray(images[0]).shape
    mask_shape = np.asarray(masks[0]).shape
    if len(image_shape) != 3 or image_shape[2] != 3:
        raise ValueError("images must have shape (N, H, W, 3)")
    if len(mask_shape) != 2 or mask_shape != image_shape[:2]:
        raise ValueError("masks must have shape (N, H, W)")
    for image, mask in zip(images, masks, strict=True):
        if np.asarray(image).shape != image_shape or np.asarray(mask).shape != mask_shape:
            raise ValueError("all images and masks in a batch must share the same shape")
        if np.asarray(image).dtype != np.uint8:
            raise ValueError("model images must use uint8 pixels")


def _unpack(output):
    if isinstance(output, dict):
        if "scores" not in output or "maps" not in output:
            raise ValueError("model output dictionary must contain scores and maps")
        return output["scores"], output["maps"]
    if isinstance(output, (tuple, list)) and len(output) == 2:
        return output
    raise ValueError("model must return (scores, maps) or a dictionary with scores and maps")


def _numpy(value) -> np.ndarray:
    if not hasattr(value, "detach"):
        raise ValueError("model outputs must be tensors")
    return value.detach().cpu().numpy()
