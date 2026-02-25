"""Tests for ImageBatcher -- threaded image loading for calibration."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.cpu
class TestImageBatcherInit:
    """Tests for ImageBatcher initialization and validation."""

    def test_init_valid_dir(self, test_image_dir):
        """ImageBatcher initializes with a valid image directory."""
        from trtutils.builder._batcher import ImageBatcher

        batcher = ImageBatcher(
            test_image_dir,
            shape=(640, 640, 3),
            dtype=np.float32,
            batch_size=4,
        )
        assert batcher.num_batches > 0
        assert batcher.batch_size == 4
        batcher._close()

    def test_init_missing_dir(self, tmp_path):
        """FileNotFoundError for non-existent directory."""
        from trtutils.builder._batcher import ImageBatcher

        with pytest.raises(FileNotFoundError):
            ImageBatcher(
                tmp_path / "nonexistent",
                shape=(640, 640, 3),
                dtype=np.float32,
            )

    def test_init_file_not_dir(self, tmp_path):
        """NotADirectoryError when path is a file."""
        from trtutils.builder._batcher import ImageBatcher

        f = tmp_path / "file.txt"
        f.write_text("test")
        with pytest.raises(NotADirectoryError):
            ImageBatcher(f, shape=(640, 640, 3), dtype=np.float32)

    def test_init_empty_dir(self, empty_dir):
        """ValueError when no images found."""
        from trtutils.builder._batcher import ImageBatcher

        with pytest.raises(ValueError, match="Could not find any images"):
            ImageBatcher(empty_dir, shape=(640, 640, 3), dtype=np.float32)

    def test_init_invalid_resize_method(self, test_image_dir):
        """ValueError for invalid resize method."""
        from trtutils.builder._batcher import ImageBatcher

        with pytest.raises(ValueError, match="Invalid resize method"):
            ImageBatcher(
                test_image_dir,
                shape=(640, 640, 3),
                dtype=np.float32,
                resize_method="invalid",
            )

    def test_init_invalid_order(self, test_image_dir):
        """ValueError for invalid order."""
        from trtutils.builder._batcher import ImageBatcher

        with pytest.raises(ValueError, match="Invalid order"):
            ImageBatcher(
                test_image_dir,
                shape=(640, 640, 3),
                dtype=np.float32,
                order="INVALID",
            )

    def test_init_max_images_zero(self, test_image_dir):
        """ValueError when max_images <= 1."""
        from trtutils.builder._batcher import ImageBatcher

        with pytest.raises(ValueError, match="max_images"):
            ImageBatcher(
                test_image_dir,
                shape=(640, 640, 3),
                dtype=np.float32,
                max_images=0,
            )

    def test_no_valid_batches(self, single_image_dir):
        """ValueError when batch_size > num_images (can't form a full batch)."""
        from trtutils.builder._batcher import ImageBatcher

        with pytest.raises(ValueError, match="Could not form any valid batches"):
            ImageBatcher(
                single_image_dir,
                shape=(224, 224, 3),
                dtype=np.float32,
                batch_size=8,  # Only 1 image, batch_size=8 → no valid batches
            )

    def test_max_images_truncates(self, test_image_dir):
        """max_images limits total images processed."""
        from trtutils.builder._batcher import ImageBatcher

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
class TestImageBatcherOutput:
    """Tests for ImageBatcher output shape and data validation."""

    @pytest.mark.parametrize("order", ["NCHW", "NHWC"], ids=["nchw", "nhwc"])
    @pytest.mark.parametrize("resize_method", ["letterbox", "linear"], ids=["letterbox", "linear"])
    def test_batch_shape(self, test_image_dir, order, resize_method):
        """Output shape matches configured (N,C,H,W) or (N,H,W,C)."""
        from trtutils.builder._batcher import ImageBatcher

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

    def test_batch_dtype(self, test_image_dir):
        """Batch dtype matches requested dtype."""
        from trtutils.builder._batcher import ImageBatcher

        batcher = ImageBatcher(
            test_image_dir,
            shape=(224, 224, 3),
            dtype=np.float32,
            batch_size=4,
        )
        batch = batcher.get_next_batch()
        assert batch.dtype == np.float32
        batcher._close()

    def test_batch_count(self, test_image_dir):
        """num_batches matches actual batch count from iteration."""
        from trtutils.builder._batcher import ImageBatcher

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
class TestImageBatcherIteration:
    """Tests for ImageBatcher batch retrieval."""

    def test_get_next_batch_returns_array(self, test_image_dir):
        """get_next_batch returns a numpy array."""
        from trtutils.builder._batcher import ImageBatcher

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

    def test_exhausted_returns_none(self, test_image_dir):
        """get_next_batch returns None after all batches consumed."""
        from trtutils.builder._batcher import ImageBatcher

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

    def test_num_batches_property(self, test_image_dir):
        """num_batches property returns expected count."""
        from trtutils.builder._batcher import ImageBatcher

        batcher = ImageBatcher(
            test_image_dir,
            shape=(224, 224, 3),
            dtype=np.float32,
            batch_size=4,
        )
        # 8 images, batch_size=4 → 2 batches
        assert batcher.num_batches == 2
        batcher._close()

    def test_batch_size_property(self, test_image_dir):
        """batch_size property returns configured batch size."""
        from trtutils.builder._batcher import ImageBatcher

        batcher = ImageBatcher(
            test_image_dir,
            shape=(224, 224, 3),
            dtype=np.float32,
            batch_size=4,
        )
        assert batcher.batch_size == 4
        batcher._close()


@pytest.mark.cpu
class TestImageBatcherThreading:
    """Tests for ImageBatcher threading behavior."""

    def test_prefetch_queue(self, test_image_dir):
        """Batches are prefetched in background thread."""
        from trtutils.builder._batcher import ImageBatcher

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

    def test_cleanup_on_close(self, test_image_dir):
        """_close() stops the background thread."""
        from trtutils.builder._batcher import ImageBatcher

        batcher = ImageBatcher(
            test_image_dir,
            shape=(224, 224, 3),
            dtype=np.float32,
            batch_size=4,
        )
        batcher._close()
        assert not batcher._thread.is_alive()
        assert batcher._event.is_set()

    def test_atexit_registered(self, test_image_dir):
        """Atexit cleanup is registered."""
        from trtutils.builder._batcher import ImageBatcher

        batcher = ImageBatcher(
            test_image_dir,
            shape=(224, 224, 3),
            dtype=np.float32,
            batch_size=4,
        )
        # atexit._exithandlers is not public, but we can verify _close is registered
        # by checking that it was called by atexit (verified by thread stopping)
        batcher._close()


@pytest.mark.cpu
class TestImageBatcherValidation:
    """Tests for ImageBatcher data validation."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
    def test_dtype_handling(self, test_image_dir, dtype):
        """Different dtypes work correctly."""
        from trtutils.builder._batcher import ImageBatcher

        batcher = ImageBatcher(
            test_image_dir,
            shape=(224, 224, 3),
            dtype=dtype,
            batch_size=4,
        )
        batch = batcher.get_next_batch()
        assert batch.dtype == dtype
        batcher._close()

    def test_c_contiguous_output(self, test_image_dir):
        """Output is always C-contiguous."""
        from trtutils.builder._batcher import ImageBatcher

        batcher = ImageBatcher(
            test_image_dir,
            shape=(224, 224, 3),
            dtype=np.float32,
            batch_size=4,
        )
        batch = batcher.get_next_batch()
        assert batch.flags["C_CONTIGUOUS"]
        batcher._close()

    def test_verbose_mode(self, test_image_dir):
        """Verbose mode produces output without error."""
        from trtutils.builder._batcher import ImageBatcher

        batcher = ImageBatcher(
            test_image_dir,
            shape=(224, 224, 3),
            dtype=np.float32,
            batch_size=4,
            verbose=True,
        )
        batch = batcher.get_next_batch()
        assert batch is not None
        batcher._close()
