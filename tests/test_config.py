import pytest

from citrus_inspection.config import Config


def test_loads_valid_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("runtime:\n  threshold: 0.7\ninspection:\n  min_views: 1\n", encoding="utf-8")

    config = Config.load(path)

    assert config.runtime.threshold == 0.7
    assert config.inspection.min_views == 1
    assert config.detector.min_area == 900


def test_rejects_unknown_config_key(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("inspection:\n  batch_size: 8\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config keys"):
        Config.load(path)


def test_rejects_invalid_view_limits():
    config = Config()
    config.inspection.min_views = 4
    config.inspection.max_views = 3

    with pytest.raises(ValueError, match="min_views"):
        config.validate()
