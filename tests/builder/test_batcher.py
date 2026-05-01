# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for SyntheticBatcher and ImageBatcher."""

from __future__ import annotations

import numpy as np
import pytest

from tests.builder.conftest import drain_batches
from trtutils.builder._batcher import ImageBatcher, SyntheticBatcher


@pytest.mark.cpu
class TestSyntheticBatcher:
    """Tests for SyntheticBatcher -- random data batches for calibration."""

    def test_invalid_order(self) -> None:
        """ValueError for invalid order."""
        with pytest.raises(ValueError, match="Invalid order"):
            SyntheticBatcher(
                shape=(8, 8, 3),
                dtype=np.float32,
                order="INVALID",
            )

    def test_num_batches_zero(self) -> None:
        """ValueError when num_batches < 1."""
        with pytest.raises(ValueError, match="num_batches must be at least 1"):
            SyntheticBatcher(
                shape=(8, 8, 3),
                dtype=np.float32,
                num_batches=0,
            )

    @pytest.mark.parametrize("order", ["NCHW", "NHWC"], ids=["nchw", "nhwc"])
    def test_batch_shape(self, order) -> None:
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

    @pytest.mark.parametrize(
        "dtype", [np.float32, np.float16, np.int8], ids=["fp32", "fp16", "int8"]
    )
    def test_batch_dtype(self, dtype) -> None:
        """Dtype matches config."""
        batcher = SyntheticBatcher(
            shape=(4, 4, 3),
            dtype=dtype,
            num_batches=1,
        )
        batch = batcher.get_next_batch()
        assert batch.dtype == dtype

    def test_data_range(self) -> None:
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

    @pytest.mark.parametrize("num_batches", [1, 3], ids=["1-batch", "3-batches"])
    def test_correct_batch_count(self, num_batches) -> None:
        """Exact num_batches batches are produced, then None."""
        batcher = SyntheticBatcher(
            shape=(4, 4, 3),
            dtype=np.float32,
            num_batches=num_batches,
        )
        assert drain_batches(batcher) == num_batches
        assert batcher.get_next_batch() is None


@pytest.mark.cpu
class TestImageBatcher:
    """Tests for ImageBatcher -- threaded image loading for calibration."""

    def test_init_missing_dir(self, tmp_path) -> None:
        """FileNotFoundError for non-existent directory."""
        with pytest.raises(FileNotFoundError):
            ImageBatcher(
                tmp_path / "nonexistent",
                shape=(640, 640, 3),
                dtype=np.float32,
            )

    def test_init_file_not_dir(self, tmp_path) -> None:
        """NotADirectoryError when path is a file."""
        f = tmp_path / "file.txt"
        f.write_text("test")
        with pytest.raises(NotADirectoryError):
            ImageBatcher(f, shape=(640, 640, 3), dtype=np.float32)

    def test_init_empty_dir(self, empty_dir) -> None:
        """ValueError when no images found."""
        with pytest.raises(ValueError, match="Could not find any images"):
            ImageBatcher(empty_dir, shape=(640, 640, 3), dtype=np.float32)

    def test_init_invalid_resize_method(self, test_image_dir) -> None:
        """ValueError for invalid resize method."""
        with pytest.raises(ValueError, match="Invalid resize method"):
            ImageBatcher(
                test_image_dir,
                shape=(640, 640, 3),
                dtype=np.float32,
                resize_method="invalid",
            )

    def test_init_invalid_order(self, test_image_dir) -> None:
        """ValueError for invalid order."""
        with pytest.raises(ValueError, match="Invalid order"):
            ImageBatcher(
                test_image_dir,
                shape=(640, 640, 3),
                dtype=np.float32,
                order="INVALID",
            )

    def test_init_max_images_zero(self, test_image_dir) -> None:
        """ValueError when max_images <= 1."""
        with pytest.raises(ValueError, match="max_images"):
            ImageBatcher(
                test_image_dir,
                shape=(640, 640, 3),
                dtype=np.float32,
                max_images=0,
            )

    def test_no_valid_batches(self, single_image_dir) -> None:
        """ValueError when batch_size > num_images."""
        with pytest.raises(ValueError, match="Could not form any valid batches"):
            ImageBatcher(
                single_image_dir,
                shape=(32, 32, 3),
                dtype=np.float32,
                batch_size=8,
            )

    def test_max_images_truncates(self, test_image_dir, close_batcher) -> None:
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
        assert drain_batches(batcher) == 2

    @pytest.mark.parametrize("order", ["NCHW", "NHWC"], ids=["nchw", "nhwc"])
    @pytest.mark.parametrize("resize_method", ["letterbox", "linear"], ids=["letterbox", "linear"])
    def test_batch_shape(self, test_image_dir, order, resize_method, close_batcher) -> None:
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

    @pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
    def test_batch_dtype(self, test_image_dir, dtype, close_batcher) -> None:
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

    def test_batch_count_and_lifecycle(self, test_image_dir, close_batcher) -> None:
        """Batch count matches, exhaustion returns None, thread stops on close."""
        batcher = ImageBatcher(
            test_image_dir,
            shape=(32, 32, 3),
            dtype=np.float32,
            batch_size=4,
        )
        close_batcher.append(batcher)
        assert drain_batches(batcher) == batcher.num_batches
        assert batcher.get_next_batch() is None
        batcher._close()
        assert not batcher._thread.is_alive()
        assert batcher._event.is_set()
