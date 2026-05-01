# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for EngineCalibrator -- init, batching, and cache operations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from trtutils.builder._calibrator import EngineCalibrator


@pytest.mark.parametrize(
    ("cache_arg", "expected_type"),
    [
        pytest.param(None, "tempfile", id="none-uses-tempfile"),
        pytest.param("custom.cache", "custom", id="string-path"),
        pytest.param(Path("custom.cache"), "custom", id="path-object"),
    ],
)
def test_cache_path_init(tmp_path, cache_arg, expected_type) -> None:
    """Constructor resolves cache path correctly for all input types."""
    if cache_arg is not None:
        cache_arg = tmp_path / cache_arg if isinstance(cache_arg, str) else tmp_path / cache_arg.name
    cal = EngineCalibrator(calibration_cache=cache_arg)
    assert isinstance(cal._cache_path, Path)
    assert cal._cache_path.is_absolute()
    if expected_type == "tempfile":
        assert cal._cache_path.suffix == ".cache"
    else:
        assert cal._cache_path == (tmp_path / "custom.cache").resolve()


def test_batcher_lifecycle() -> None:
    """Batcher starts None, set_batcher assigns it, batch_size reflects batcher."""
    cal = EngineCalibrator()
    assert cal._batcher is None
    assert cal.get_batch_size() == 1

    mock_batcher = MagicMock()
    mock_batcher.batch_size = 8
    cal.set_batcher(mock_batcher)
    assert cal._batcher is mock_batcher
    assert cal.get_batch_size() == 8


@pytest.mark.parametrize(
    "has_batcher",
    [
        pytest.param(False, id="no-batcher"),
        pytest.param(True, id="batcher-exhausted"),
    ],
)
def test_get_batch_no_data(has_batcher) -> None:
    """get_batch returns None when no batcher is set or batcher is exhausted."""
    cal = EngineCalibrator()
    if has_batcher:
        mock_batcher = MagicMock()
        mock_batcher.get_next_batch.return_value = None
        cal.set_batcher(mock_batcher)
    result = cal.get_batch(["input"])
    assert result is None


def test_get_batch_returns_gpu_ptr() -> None:
    """get_batch allocates GPU memory and returns [int] pointer."""
    cal = EngineCalibrator()
    mock_batcher = MagicMock()
    fake_data = np.zeros((1, 3, 8, 8), dtype=np.float32)
    mock_batcher.get_next_batch.return_value = fake_data
    mock_batcher.batch_size = 1
    cal.set_batcher(mock_batcher)

    result = cal.get_batch(["input"])
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], int)
    assert result[0] > 0


def test_write_then_read_cache(tmp_path) -> None:
    """write_calibration_cache then read_calibration_cache roundtrip works."""
    cache_path = tmp_path / "test.cache"
    cal = EngineCalibrator(calibration_cache=cache_path)

    test_data = b"calibration_data_bytes_here"
    cal.write_calibration_cache(test_data)
    assert cache_path.exists()

    result = cal.read_calibration_cache()
    assert result == test_data


def test_cache_edge_cases(tmp_path) -> None:
    """Read returns None for missing file; both read/write are no-ops when _cache_path is None."""
    # missing file
    cal = EngineCalibrator(calibration_cache=tmp_path / "nonexistent.cache")
    assert cal.read_calibration_cache() is None

    # none path
    cal._cache_path = None
    assert cal.read_calibration_cache() is None
    cal.write_calibration_cache(b"some data")  # should not raise
