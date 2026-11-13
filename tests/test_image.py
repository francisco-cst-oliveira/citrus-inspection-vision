import numpy as np
import pytest

from citrus_inspection.image import extract_crop


def test_crop_outside_frame_returns_empty_input():
    frame = np.full((60, 80, 3), 127, np.uint8)

    crop, mask, visible = extract_crop(frame, np.array([100, 20, 130, 50]), None, 32, 0.1)

    assert crop.shape == (32, 32, 3)
    assert mask.shape == (32, 32)
    assert not crop.any()
    assert not mask.any()
    assert visible == 0


def test_crop_rejects_mask_with_wrong_shape():
    frame = np.zeros((60, 80, 3), np.uint8)
    mask = np.zeros((30, 40), bool)

    with pytest.raises(ValueError, match="mask shape"):
        extract_crop(frame, np.array([10, 10, 30, 30]), mask, 32, 0.1)
