# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for ImageBatcher -- threaded image loading for calibration."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.builder._batcher import ImageBatcher


@pytest.mark.cpu
def test_init_valid_dir(test_image_dir) -> None:
    """ImageBatcher initializes with a valid image directory."""
    batcher = ImageBatcher(
        test_image_dir,
        shape=(640, 640, 3),
        dtype=np.float32,
        batch_size=4,
    )
    assert batcher.num_batches > 0
    assert batcher.batch_size == 4
    batcher._close()


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
    """ValueError when batch_size > num_images (can't form a full batch)."""
    with pytest.raises(ValueError, match="Could not form any valid batches"):
        ImageBatcher(
            single_image_dir,
            shape=(224, 224, 3),
            dtype=np.float32,
            batch_size=8,  # Only 1 image, batch_size=8 → no valid batches
        )


@pytest.mark.cpu
def test_max_images_truncates(test_image_dir) -> None:
    """max_images limits total images processed."""
    batcher = ImageBatcher(
        test_image_dir,
        shape=(224, 224, 3),
        dtype=np.float32,
        batch_size=2,
        max_images=4,
    )
    assert batcher.num_batches == 2
    batcher._close()


@pytest.mark.cpu
@pytest.mark.parametrize("order", ["NCHW", "NHWC"], ids=["nchw", "nhwc"])
@pytest.mark.parametrize("resize_method", ["letterbox", "linear"], ids=["letterbox", "linear"])
def test_batch_shape(test_image_dir, order, resize_method) -> None:
    """Output shape matches configured (N,C,H,W) or (N,H,W,C)."""
    h, w, c = 224, 224, 3
    bs = 4
    batcher = ImageBatcher(
        test_image_dir,
        shape=(h, w, c),
        dtype=np.float32,
        batch_size=bs,
        order=order,
        resize_method=resize_method,
    )
    batch = batcher.get_next_batch()
    assert batch is not None
    if order == "NCHW":
        assert batch.shape == (bs, c, h, w)
    else:
        assert batch.shape == (bs, h, w, c)
    batcher._close()


@pytest.mark.cpu
def test_batch_dtype(test_image_dir) -> None:
    """Batch dtype matches requested dtype."""
    batcher = ImageBatcher(
        test_image_dir,
        shape=(224, 224, 3),
        dtype=np.float32,
        batch_size=4,
    )
    batch = batcher.get_next_batch()
    assert batch.dtype == np.float32
    batcher._close()


@pytest.mark.cpu
def test_batch_count(test_image_dir) -> None:
    """num_batches matches actual batch count from iteration."""
    batcher = ImageBatcher(
        test_image_dir,
        shape=(224, 224, 3),
        dtype=np.float32,
        batch_size=4,
    )
    count = 0
    while batcher.get_next_batch() is not None:
        count += 1
    assert count == batcher.num_batches
    batcher._close()


@pytest.mark.cpu
def test_get_next_batch_returns_array(test_image_dir) -> None:
    """get_next_batch returns a numpy array."""
    batcher = ImageBatcher(
        test_image_dir,
        shape=(224, 224, 3),
        dtype=np.float32,
        batch_size=4,
    )
    batch = batcher.get_next_batch()
    assert isinstance(batch, np.ndarray)
    assert batch.shape[0] == 4
    batcher._close()


@pytest.mark.cpu
def test_exhausted_returns_none(test_image_dir) -> None:
    """get_next_batch returns None after all batches consumed."""
    batcher = ImageBatcher(
        test_image_dir,
        shape=(224, 224, 3),
        dtype=np.float32,
        batch_size=4,
    )
    for _ in range(batcher.num_batches):
        batch = batcher.get_next_batch()
        assert batch is not None
    assert batcher.get_next_batch() is None
    batcher._close()


@pytest.mark.cpu
def test_prefetch_queue(test_image_dir) -> None:
    """Batches are prefetched in background thread."""
    batcher = ImageBatcher(
        test_image_dir,
        shape=(224, 224, 3),
        dtype=np.float32,
        batch_size=4,
    )
    # The thread should be alive and prefetching
    assert batcher._thread.is_alive() or batcher._queue.qsize() > 0
    # Get all batches
    for _ in range(batcher.num_batches):
        batcher.get_next_batch()
    batcher._close()


@pytest.mark.cpu
def test_cleanup_on_close(test_image_dir) -> None:
    """_close() stops the background thread."""
    batcher = ImageBatcher(
        test_image_dir,
        shape=(224, 224, 3),
        dtype=np.float32,
        batch_size=4,
    )
    batcher._close()
    assert not batcher._thread.is_alive()
    assert batcher._event.is_set()


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
def test_dtype_handling(test_image_dir, dtype) -> None:
    """Different dtypes work correctly."""
    batcher = ImageBatcher(
        test_image_dir,
        shape=(224, 224, 3),
        dtype=dtype,
        batch_size=4,
    )
    batch = batcher.get_next_batch()
    assert batch.dtype == dtype
    batcher._close()


@pytest.mark.cpu
def test_c_contiguous_output(test_image_dir) -> None:
    """Output is always C-contiguous."""
    batcher = ImageBatcher(
        test_image_dir,
        shape=(224, 224, 3),
        dtype=np.float32,
        batch_size=4,
    )
    batch = batcher.get_next_batch()
    assert batch.flags["C_CONTIGUOUS"]
    batcher._close()
