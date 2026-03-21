# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for ImageBatcher -- threaded image loading for calibration."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.builder._batcher import ImageBatcher


@pytest.fixture
def close_batcher():
    """Ensure ImageBatcher._close() is called even on test failure."""
    batchers = []
    yield batchers
    for b in batchers:
        b._close()


@pytest.mark.cpu
def test_init_missing_dir(tmp_path) -> None:
    """FileNotFoundError for non-existent directory."""
    with pytest.raises(FileNotFoundError):
        ImageBatcher(
            tmp_path / "nonexistent",
            shape=(640, 640, 3),
            dtype=np.float32,
        )


@pytest.mark.cpu
def test_init_file_not_dir(tmp_path) -> None:
    """NotADirectoryError when path is a file."""
    f = tmp_path / "file.txt"
    f.write_text("test")
    with pytest.raises(NotADirectoryError):
        ImageBatcher(f, shape=(640, 640, 3), dtype=np.float32)


@pytest.mark.cpu
def test_init_empty_dir(empty_dir) -> None:
    """ValueError when no images found."""
    with pytest.raises(ValueError, match="Could not find any images"):
        ImageBatcher(empty_dir, shape=(640, 640, 3), dtype=np.float32)


@pytest.mark.cpu
def test_init_invalid_resize_method(test_image_dir) -> None:
    """ValueError for invalid resize method."""
    with pytest.raises(ValueError, match="Invalid resize method"):
        ImageBatcher(
            test_image_dir,
            shape=(640, 640, 3),
            dtype=np.float32,
            resize_method="invalid",
        )


@pytest.mark.cpu
def test_init_invalid_order(test_image_dir) -> None:
    """ValueError for invalid order."""
    with pytest.raises(ValueError, match="Invalid order"):
        ImageBatcher(
            test_image_dir,
            shape=(640, 640, 3),
            dtype=np.float32,
            order="INVALID",
        )


@pytest.mark.cpu
def test_init_max_images_zero(test_image_dir) -> None:
    """ValueError when max_images <= 1."""
    with pytest.raises(ValueError, match="max_images"):
        ImageBatcher(
            test_image_dir,
            shape=(640, 640, 3),
            dtype=np.float32,
            max_images=0,
        )


@pytest.mark.cpu
def test_no_valid_batches(single_image_dir) -> None:
    """ValueError when batch_size > num_images."""
    with pytest.raises(ValueError, match="Could not form any valid batches"):
        ImageBatcher(
            single_image_dir,
            shape=(32, 32, 3),
            dtype=np.float32,
            batch_size=8,
        )


@pytest.mark.cpu
def test_max_images_truncates(test_image_dir, close_batcher) -> None:
    """max_images limits total batches produced."""
    batcher = ImageBatcher(
        test_image_dir,
        shape=(32, 32, 3),
        dtype=np.float32,
        batch_size=2,
        max_images=4,
    )
    close_batcher.append(batcher)
    assert batcher.num_batches == 2
    count = 0
    while batcher.get_next_batch() is not None:
        count += 1
    assert count == 2


@pytest.mark.cpu
@pytest.mark.parametrize("order", ["NCHW", "NHWC"], ids=["nchw", "nhwc"])
@pytest.mark.parametrize("resize_method", ["letterbox", "linear"], ids=["letterbox", "linear"])
def test_batch_shape(test_image_dir, order, resize_method, close_batcher) -> None:
    """Output shape matches configured layout and is C-contiguous."""
    h, w, c = 32, 32, 3
    bs = 4
    batcher = ImageBatcher(
        test_image_dir,
        shape=(h, w, c),
        dtype=np.float32,
        batch_size=bs,
        order=order,
        resize_method=resize_method,
    )
    close_batcher.append(batcher)
    batch = batcher.get_next_batch()
    assert batch is not None
    if order == "NCHW":
        assert batch.shape == (bs, c, h, w)
    else:
        assert batch.shape == (bs, h, w, c)
    assert batch.flags["C_CONTIGUOUS"]


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
def test_batch_dtype(test_image_dir, dtype, close_batcher) -> None:
    """Batch dtype matches requested dtype."""
    batcher = ImageBatcher(
        test_image_dir,
        shape=(32, 32, 3),
        dtype=dtype,
        batch_size=4,
    )
    close_batcher.append(batcher)
    batch = batcher.get_next_batch()
    assert batch.dtype == dtype


@pytest.mark.cpu
def test_batch_count_and_lifecycle(test_image_dir, close_batcher) -> None:
    """Batch count matches, exhaustion returns None, thread stops on close."""
    batcher = ImageBatcher(
        test_image_dir,
        shape=(32, 32, 3),
        dtype=np.float32,
        batch_size=4,
    )
    close_batcher.append(batcher)
    count = 0
    while batcher.get_next_batch() is not None:
        count += 1
    assert count == batcher.num_batches
    assert batcher.get_next_batch() is None
    batcher._close()
    assert not batcher._thread.is_alive()
    assert batcher._event.is_set()
