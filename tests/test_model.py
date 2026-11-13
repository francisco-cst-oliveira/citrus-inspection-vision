import numpy as np
import pytest

from citrus_inspection.model import Prediction, validate_prediction


def test_prediction_normalizes_single_channel_outputs():
    prediction = Prediction(
        scores=np.array([[0.2], [0.8]], np.float32),
        maps=np.zeros((2, 1, 16, 16), np.float32),
    )

    normalized = validate_prediction(prediction, 2)

    assert normalized.scores.shape == (2,)
    assert normalized.maps.shape == (2, 16, 16)


def test_prediction_rejects_non_finite_values():
    prediction = Prediction(
        scores=np.array([np.nan], np.float32),
        maps=np.zeros((1, 16, 16), np.float32),
    )

    with pytest.raises(ValueError, match="non-finite"):
        validate_prediction(prediction, 1)


def test_prediction_rejects_wrong_batch_size():
    prediction = Prediction(
        scores=np.array([0.2], np.float32),
        maps=np.zeros((1, 16, 16), np.float32),
    )

    with pytest.raises(ValueError, match="scores"):
        validate_prediction(prediction, 2)


def test_prediction_rejects_score_outside_unit_interval():
    prediction = Prediction(
        scores=np.array([1.2], np.float32),
        maps=np.zeros((1, 16, 16), np.float32),
    )

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_prediction(prediction, 1)
