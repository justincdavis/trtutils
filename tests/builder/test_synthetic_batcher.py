# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for SyntheticBatcher -- random data batches for calibration."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.builder._batcher import SyntheticBatcher


@pytest.mark.cpu
def test_init_defaults() -> None:
    """Default initialization works."""
    batcher = SyntheticBatcher(
        shape=(224, 224, 3),
        dtype=np.float32,
    )
    assert batcher.num_batches == 10
    assert batcher.batch_size == 8


@pytest.mark.cpu
def test_custom_params() -> None:
    """Custom parameters are stored correctly."""
    batcher = SyntheticBatcher(
        shape=(640, 640, 3),
        dtype=np.float16,
        batch_size=4,
        num_batches=5,
        data_range=(-1.0, 1.0),
    )
    assert batcher.num_batches == 5
    assert batcher.batch_size == 4


@pytest.mark.cpu
def test_invalid_order() -> None:
    """ValueError for invalid order."""
    with pytest.raises(ValueError, match="Invalid order"):
        SyntheticBatcher(
            shape=(224, 224, 3),
            dtype=np.float32,
            order="INVALID",
        )


@pytest.mark.cpu
def test_num_batches_zero() -> None:
    """ValueError when num_batches < 1."""
    with pytest.raises(ValueError, match="num_batches must be at least 1"):
        SyntheticBatcher(
            shape=(224, 224, 3),
            dtype=np.float32,
            num_batches=0,
        )


@pytest.mark.cpu
def test_nhwc_order() -> None:
    """NHWC order sets correct data shape."""
    batcher = SyntheticBatcher(
        shape=(224, 224, 3),
        dtype=np.float32,
        order="NHWC",
    )
    assert batcher.num_batches == 10


@pytest.mark.cpu
@pytest.mark.parametrize("order", ["NCHW", "NHWC"], ids=["nchw", "nhwc"])
def test_batch_shape(order) -> None:
    """Shape matches config for both NCHW and NHWC."""
    batcher = SyntheticBatcher(
        shape=(224, 224, 3),
        dtype=np.float32,
        batch_size=2,
        num_batches=1,
        order=order,
    )
    batch = batcher.get_next_batch()
    if order == "NCHW":
        assert batch.shape == (2, 3, 224, 224)
    else:
        assert batch.shape == (2, 224, 224, 3)


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8], ids=["fp32", "fp16", "int8"])
def test_batch_dtype(dtype) -> None:
    """Dtype matches config."""
    batcher = SyntheticBatcher(
        shape=(32, 32, 3),
        dtype=dtype,
        num_batches=1,
    )
    batch = batcher.get_next_batch()
    assert batch.dtype == dtype


@pytest.mark.cpu
def test_data_range() -> None:
    """Values fall within configured data_range."""
    low, high = -1.0, 1.0
    batcher = SyntheticBatcher(
        shape=(32, 32, 3),
        dtype=np.float32,
        num_batches=1,
        data_range=(low, high),
    )
    batch = batcher.get_next_batch()
    assert batch.min() >= low
    assert batch.max() <= high


@pytest.mark.cpu
@pytest.mark.parametrize("num_batches", [1, 5, 10], ids=["1_batch", "5_batches", "10_batches"])
def test_correct_batch_count(num_batches) -> None:
    """Exact num_batches batches are produced."""
    batcher = SyntheticBatcher(
        shape=(32, 32, 3),
        dtype=np.float32,
        num_batches=num_batches,
    )
    count = 0
    while batcher.get_next_batch() is not None:
        count += 1
    assert count == num_batches


@pytest.mark.cpu
def test_exhausted_returns_none() -> None:
    """get_next_batch returns None after all batches consumed."""
    batcher = SyntheticBatcher(
        shape=(224, 224, 3),
        dtype=np.float32,
        num_batches=2,
    )
    batcher.get_next_batch()
    batcher.get_next_batch()
    assert batcher.get_next_batch() is None


@pytest.mark.cpu
def test_c_contiguous_output() -> None:
    """Output is C-contiguous."""
    batcher = SyntheticBatcher(
        shape=(224, 224, 3),
        dtype=np.float32,
        num_batches=1,
    )
    batch = batcher.get_next_batch()
    assert batch.flags["C_CONTIGUOUS"]
