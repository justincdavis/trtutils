# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for EngineCalibrator -- init, batching, and cache operations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from trtutils._flags import FLAGS

if not FLAGS.INT8_CALIBRATION:
    pytest.skip("implicit INT8 calibration removed in TensorRT 11", allow_module_level=True)

from trtutils.builder._calibrator import EngineCalibrator, MinMaxCalibrator
from trtutils.compat._libs import trt
from trtutils.core._memory import memcpy_device_to_host

_A = np.linspace(-2.0, 2.0, 48, dtype=np.float32).reshape(1, 3, 4, 4)
_B = np.arange(-8, 8, dtype=np.int8).reshape(1, 16)


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


@pytest.mark.parametrize(
    ("calibrator", "base"),
    [
        pytest.param(EngineCalibrator, "IInt8EntropyCalibrator2", id="entropy"),
        pytest.param(MinMaxCalibrator, "IInt8MinMaxCalibrator", id="minmax"),
    ],
)
def test_write_then_read_cache(tmp_path, calibrator, base) -> None:
    """Both calibrators are their TensorRT calibrator and round-trip the cache."""
    cache_path = tmp_path / "test.cache"
    cal = calibrator(calibration_cache=cache_path)
    assert isinstance(cal, getattr(trt, base))

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


@pytest.mark.parametrize(
    ("batch", "names"),
    [
        pytest.param(np.zeros((1, 3, 8, 8), dtype=np.float32), ["input"], id="one-input"),
        pytest.param([_A, _A], ["a", "b"], id="two-inputs"),
    ],
)
@patch("trtutils.builder._calibrator.cuda_free")
@patch("trtutils.builder._calibrator.memcpy_host_to_device")
@patch("trtutils.builder._calibrator.cuda_malloc", side_effect=[1234, 5678])
def test_get_batch_reuses_allocation(malloc, memcpy, free, batch, names) -> None:
    """get_batch allocates device memory once per input across batches and frees it on close."""
    cal = EngineCalibrator()
    mock_batcher = MagicMock()
    mock_batcher.get_next_batch.return_value = batch
    cal.set_batcher(mock_batcher)

    ptrs = [1234, 5678][: len(names)]
    for _ in range(3):
        assert cal.get_batch(names) == ptrs
    assert malloc.call_count == len(names)
    assert memcpy.call_count == 3 * len(names)
    cal.close()
    assert sorted(call.args[0] for call in free.call_args_list) == ptrs
    cal.close()
    assert free.call_count == len(names)


def test_close_removes_temp_cache() -> None:
    """Close deletes the temporary cache file but keeps a user-provided one."""
    cal = EngineCalibrator()
    cal.write_calibration_cache(b"x")
    assert cal._cache_path.exists()
    cal.close()
    assert not cal._cache_path.exists()


@pytest.mark.parametrize(
    ("batch_dtype", "input_dtypes", "expected"),
    [
        pytest.param(np.float32, {"input": np.float16}, np.float16, id="cast-to-input-dtype"),
        pytest.param(np.float16, {"input": np.float16}, np.float16, id="matching-dtype"),
        pytest.param(np.float32, {"other": np.float16}, np.float32, id="unknown-input"),
        # tensorrt binds an int8 input as float32 while calibrating (checked on 8.6)
        pytest.param(np.float32, {"input": np.int8}, np.float32, id="int8-input-float32-batch"),
        pytest.param(np.int8, {"input": np.int8}, np.float32, id="int8-input-int8-batch"),
        pytest.param(np.float32, {"input": np.int32}, np.int32, id="int32-input"),
    ],
)
@patch("trtutils.builder._calibrator.cuda_free")
@patch("trtutils.builder._calibrator.memcpy_host_to_device")
@patch("trtutils.builder._calibrator.cuda_malloc", return_value=1234)
def test_get_batch_matches_input_dtype(
    malloc, memcpy, free, batch_dtype, input_dtypes, expected
) -> None:
    """get_batch copies batches in the input dtype TensorRT reads them as."""
    cal = EngineCalibrator()
    cal.set_input_dtypes(input_dtypes)
    batch = np.linspace(-2.0, 2.0, 48).astype(batch_dtype).reshape(1, 3, 4, 4)
    mock_batcher = MagicMock()
    mock_batcher.get_next_batch.return_value = batch
    cal.set_batcher(mock_batcher)

    assert cal.get_batch(["input"]) == [1234]
    sent = memcpy.call_args.args[1]
    assert sent.dtype == expected
    np.testing.assert_array_equal(sent, batch.astype(expected))
    malloc.assert_called_once_with(sent.nbytes)
    cal.close()
    free.assert_called_once_with(1234)


@pytest.mark.parametrize(
    ("batch", "names", "expected"),
    [
        pytest.param(_A, ["a"], [_A.astype(np.float16)], id="array"),
        pytest.param(
            [_A, _B], ["a", "b"], [_A.astype(np.float16), _B.astype(np.float32)], id="list"
        ),
        pytest.param(
            (_A, _B), ["a", "b"], [_A.astype(np.float16), _B.astype(np.float32)], id="tuple"
        ),
        pytest.param(
            {"b": _B, "a": _A}, ["a", "b"], [_A.astype(np.float16), _B.astype(np.float32)], id="dict"
        ),
        pytest.param(
            {"a": _A, "b": _B},
            ["b", "a"],
            [_B.astype(np.float32), _A.astype(np.float16)],
            id="dict-in-names-order",
        ),
    ],
)
def test_get_batch_copies_each_input(batch, names, expected) -> None:
    """get_batch copies each input to its own reused buffer, in names order and its read dtype."""
    cal = EngineCalibrator()
    # a float16 input, and an int8 input that tensorrt binds as float32 while calibrating
    cal.set_input_dtypes({"a": np.float16, "b": np.int8})
    # a larger batch than the input is copied whole, tensorrt reads its start
    cal.set_input_shapes({"a": _A.shape, "b": (1, 8)})
    mock_batcher = MagicMock()
    mock_batcher.get_next_batch.return_value = batch
    cal.set_batcher(mock_batcher)

    first = cal.get_batch(names)
    assert first is not None
    assert len(set(first)) == len(names)
    for ptr, want in zip(first, expected):
        got = np.empty_like(want)
        memcpy_device_to_host(got, ptr)
        np.testing.assert_array_equal(got, want)
    assert cal.get_batch(names) == first
    cal.close()


@pytest.mark.parametrize(
    ("batch", "exc", "match"),
    [
        pytest.param(_A, ValueError, "returned one array", id="array-for-two-inputs"),
        pytest.param([_A], ValueError, r"holds 1 array\(s\)", id="short-list"),
        pytest.param((_A, _B, _A), ValueError, r"holds 3 array\(s\)", id="long-tuple"),
        pytest.param({"a": _A}, ValueError, r"missing \['b'\]", id="dict-missing-input"),
        pytest.param(
            {"a": _A, "b": _B, "c": _A}, ValueError, r"unknown \['c'\]", id="dict-unknown-input"
        ),
        pytest.param("a", TypeError, "must be a numpy array", id="not-a-batch"),
        # tensorrt would read the 48 values of a past the end of the 16 given
        pytest.param(
            [_A[:, :1], _B], ValueError, r"\(16 values\), but TensorRT reads 48", id="small-array"
        ),
    ],
)
def test_get_batch_mismatch_raises(batch, exc, match) -> None:
    """A batch that does not fit the inputs raises and is kept as the calibrator's error."""
    cal = EngineCalibrator()
    cal.set_input_shapes({"a": _A.shape, "b": _B.shape})
    mock_batcher = MagicMock()
    mock_batcher.get_next_batch.return_value = batch
    cal.set_batcher(mock_batcher)

    assert cal.error is None
    with pytest.raises(exc, match=match) as exc_info:
        cal.get_batch(["a", "b"])
    assert cal.error is exc_info.value
    cal.close()
