# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for EngineCalibrator -- init, batching, and cache operations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np


def _calibrator_import():
    """Import EngineCalibrator lazily."""
    from trtutils.builder._calibrator import EngineCalibrator

    return EngineCalibrator


# ===========================================================================
# Initialization tests
# ===========================================================================
class TestCalibratorInit:
    """Tests for EngineCalibrator.__init__."""

    def test_default_cache_path(self) -> None:
        """Default cache path is 'calibration.cache' resolved."""
        EngineCalibrator = _calibrator_import()
        cal = EngineCalibrator()
        assert cal._cache_path == Path("calibration.cache").resolve()

    def test_custom_cache_path(self, tmp_path: Path) -> None:
        """Custom calibration_cache path is used."""
        EngineCalibrator = _calibrator_import()
        custom = tmp_path / "custom.cache"
        cal = EngineCalibrator(calibration_cache=custom)
        assert cal._cache_path == custom.resolve()

    def test_custom_cache_path_string(self, tmp_path: Path) -> None:
        """String calibration_cache path is converted to Path."""
        EngineCalibrator = _calibrator_import()
        custom = str(tmp_path / "string.cache")
        cal = EngineCalibrator(calibration_cache=custom)
        assert cal._cache_path == Path(custom).resolve()

    def test_none_cache_path(self) -> None:
        """calibration_cache=None uses default path."""
        EngineCalibrator = _calibrator_import()
        cal = EngineCalibrator(calibration_cache=None)
        assert cal._cache_path == Path("calibration.cache").resolve()

    def test_batcher_initially_none(self) -> None:
        """_batcher is None on creation."""
        EngineCalibrator = _calibrator_import()
        cal = EngineCalibrator()
        assert cal._batcher is None


# ===========================================================================
# Batcher tests
# ===========================================================================
class TestCalibratorBatcher:
    """Tests for set_batcher and get_batch_size."""

    def test_set_batcher(self) -> None:
        """set_batcher assigns the batcher reference."""
        EngineCalibrator = _calibrator_import()
        cal = EngineCalibrator()
        mock_batcher = MagicMock()
        mock_batcher.batch_size = 4
        cal.set_batcher(mock_batcher)
        assert cal._batcher is mock_batcher

    def test_get_batch_size_with_batcher(self) -> None:
        """get_batch_size returns batcher.batch_size when batcher is set."""
        EngineCalibrator = _calibrator_import()
        cal = EngineCalibrator()
        mock_batcher = MagicMock()
        mock_batcher.batch_size = 8
        cal.set_batcher(mock_batcher)
        assert cal.get_batch_size() == 8

    def test_get_batch_size_no_batcher(self) -> None:
        """get_batch_size returns 1 when no batcher is set."""
        EngineCalibrator = _calibrator_import()
        cal = EngineCalibrator()
        assert cal.get_batch_size() == 1


# ===========================================================================
# Batch retrieval tests
# ===========================================================================
class TestCalibratorBatch:
    """Tests for get_batch."""

    def test_get_batch_no_batcher(self) -> None:
        """get_batch returns None when no batcher is set."""
        EngineCalibrator = _calibrator_import()
        cal = EngineCalibrator()
        result = cal.get_batch(["input"])
        assert result is None

    def test_get_batch_batcher_exhausted(self) -> None:
        """get_batch returns None when batcher.get_next_batch() returns None."""
        EngineCalibrator = _calibrator_import()
        cal = EngineCalibrator()
        mock_batcher = MagicMock()
        mock_batcher.get_next_batch.return_value = None
        cal.set_batcher(mock_batcher)
        result = cal.get_batch(["input"])
        assert result is None

    def test_get_batch_returns_gpu_ptr(self) -> None:
        """get_batch allocates GPU memory and returns [int] pointer."""
        EngineCalibrator = _calibrator_import()
        cal = EngineCalibrator()

        # Create a mock batcher that returns a valid numpy array
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
        assert result[0] > 0  # Valid GPU pointer is non-zero

    def test_get_batch_names_ignored(self) -> None:
        """The 'names' parameter is unused (marked noqa: ARG002)."""
        EngineCalibrator = _calibrator_import()
        cal = EngineCalibrator()
        mock_batcher = MagicMock()
        mock_batcher.get_next_batch.return_value = None
        cal.set_batcher(mock_batcher)
        # Pass various names -- should not affect behavior
        result = cal.get_batch(["input1", "input2", "foobar"])
        assert result is None


# ===========================================================================
# Calibration cache tests
# ===========================================================================
class TestCalibratorCache:
    """Tests for read_calibration_cache and write_calibration_cache."""

    def test_read_cache_file_not_exists(self, tmp_path: Path) -> None:
        """read_calibration_cache returns None if cache file does not exist."""
        EngineCalibrator = _calibrator_import()
        cal = EngineCalibrator(calibration_cache=tmp_path / "nonexistent.cache")
        result = cal.read_calibration_cache()
        assert result is None

    def test_write_then_read_cache(self, tmp_path: Path) -> None:
        """write_calibration_cache then read_calibration_cache roundtrip works."""
        EngineCalibrator = _calibrator_import()
        cache_path = tmp_path / "test.cache"
        cal = EngineCalibrator(calibration_cache=cache_path)

        # Write cache data
        test_data = b"calibration_data_bytes_here"
        cal.write_calibration_cache(test_data)

        # Verify file was created
        assert cache_path.exists()

        # Read it back
        result = cal.read_calibration_cache()
        assert result == test_data

    def test_read_cache_none_path(self) -> None:
        """read_calibration_cache returns None when _cache_path is None."""
        EngineCalibrator = _calibrator_import()
        cal = EngineCalibrator()
        # Manually set to None to hit the None-check branch
        cal._cache_path = None
        result = cal.read_calibration_cache()
        assert result is None

    def test_write_cache_none_path(self) -> None:
        """write_calibration_cache is a no-op when _cache_path is None."""
        EngineCalibrator = _calibrator_import()
        cal = EngineCalibrator()
        cal._cache_path = None
        # Should not raise
        cal.write_calibration_cache(b"some data")

    def test_read_cache_existing_file(self, tmp_path: Path) -> None:
        """read_calibration_cache reads bytes from an existing file."""
        EngineCalibrator = _calibrator_import()
        cache_path = tmp_path / "existing.cache"
        expected = b"\x00\x01\x02\x03"
        cache_path.write_bytes(expected)

        cal = EngineCalibrator(calibration_cache=cache_path)
        result = cal.read_calibration_cache()
        assert result == expected
