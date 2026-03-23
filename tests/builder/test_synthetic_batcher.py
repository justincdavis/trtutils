# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for SyntheticBatcher -- random data batches for calibration."""

from __future__ import annotations

import numpy as np
import pytest

from tests.builder.conftest import drain_batches
from trtutils.builder._batcher import SyntheticBatcher


@pytest.mark.cpu
def test_invalid_order() -> None:
    """ValueError for invalid order."""
    with pytest.raises(ValueError, match="Invalid order"):
        SyntheticBatcher(
            shape=(8, 8, 3),
            dtype=np.float32,
            order="INVALID",
        )


@pytest.mark.cpu
def test_num_batches_zero() -> None:
    """ValueError when num_batches < 1."""
    with pytest.raises(ValueError, match="num_batches must be at least 1"):
        SyntheticBatcher(
            shape=(8, 8, 3),
            dtype=np.float32,
            num_batches=0,
        )


@pytest.mark.cpu
@pytest.mark.parametrize("order", ["NCHW", "NHWC"], ids=["nchw", "nhwc"])
def test_batch_shape(order) -> None:
    """Shape matches config and output is C-contiguous."""
    batcher = SyntheticBatcher(
        shape=(8, 8, 3),
        dtype=np.float32,
        batch_size=2,
        num_batches=1,
        order=order,
    )
    batch = batcher.get_next_batch()
    if order == "NCHW":
        assert batch.shape == (2, 3, 8, 8)
    else:
        assert batch.shape == (2, 8, 8, 3)
    assert batch.flags["C_CONTIGUOUS"]


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8], ids=["fp32", "fp16", "int8"])
def test_batch_dtype(dtype) -> None:
    """Dtype matches config."""
    batcher = SyntheticBatcher(
        shape=(4, 4, 3),
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
        shape=(4, 4, 3),
        dtype=np.float32,
        num_batches=1,
        data_range=(low, high),
    )
    batch = batcher.get_next_batch()
    assert batch.min() >= low
    assert batch.max() <= high


@pytest.mark.cpu
@pytest.mark.parametrize("num_batches", [1, 3], ids=["1-batch", "3-batches"])
def test_correct_batch_count(num_batches) -> None:
    """Exact num_batches batches are produced, then None."""
    batcher = SyntheticBatcher(
        shape=(4, 4, 3),
        dtype=np.float32,
        num_batches=num_batches,
    )
    assert drain_batches(batcher) == num_batches
    assert batcher.get_next_batch() is None
