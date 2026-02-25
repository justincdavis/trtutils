"""Tests for SyntheticBatcher -- random data batches for calibration."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.cpu
class TestSyntheticBatcherInit:
    """Tests for SyntheticBatcher initialization."""

    def test_init_defaults(self):
        """Default initialization works."""
        from trtutils.builder._batcher import SyntheticBatcher

        batcher = SyntheticBatcher(
            shape=(224, 224, 3),
            dtype=np.float32,
        )
        assert batcher.num_batches == 10
        assert batcher.batch_size == 8

    def test_custom_params(self):
        """Custom parameters are stored correctly."""
        from trtutils.builder._batcher import SyntheticBatcher

        batcher = SyntheticBatcher(
            shape=(640, 640, 3),
            dtype=np.float16,
            batch_size=4,
            num_batches=5,
            data_range=(-1.0, 1.0),
        )
        assert batcher.num_batches == 5
        assert batcher.batch_size == 4

    def test_invalid_order(self):
        """ValueError for invalid order."""
        from trtutils.builder._batcher import SyntheticBatcher

        with pytest.raises(ValueError, match="Invalid order"):
            SyntheticBatcher(
                shape=(224, 224, 3),
                dtype=np.float32,
                order="INVALID",
            )

    def test_num_batches_zero(self):
        """ValueError when num_batches < 1."""
        from trtutils.builder._batcher import SyntheticBatcher

        with pytest.raises(ValueError, match="num_batches must be at least 1"):
            SyntheticBatcher(
                shape=(224, 224, 3),
                dtype=np.float32,
                num_batches=0,
            )

    def test_nhwc_order(self):
        """NHWC order sets correct data shape."""
        from trtutils.builder._batcher import SyntheticBatcher

        batcher = SyntheticBatcher(
            shape=(224, 224, 3),
            dtype=np.float32,
            order="NHWC",
        )
        assert batcher.num_batches == 10

    def test_verbose(self):
        """Verbose mode does not raise."""
        from trtutils.builder._batcher import SyntheticBatcher

        batcher = SyntheticBatcher(
            shape=(224, 224, 3),
            dtype=np.float32,
            verbose=True,
        )
        assert batcher.num_batches == 10


@pytest.mark.cpu
class TestSyntheticBatcherOutput:
    """Tests for SyntheticBatcher output shape, dtype, and data range."""

    @pytest.mark.parametrize("order", ["NCHW", "NHWC"], ids=["nchw", "nhwc"])
    def test_batch_shape(self, order):
        """Shape matches config for both NCHW and NHWC."""
        from trtutils.builder._batcher import SyntheticBatcher

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

    @pytest.mark.parametrize(
        "dtype", [np.float32, np.float16, np.int8], ids=["fp32", "fp16", "int8"]
    )
    def test_batch_dtype(self, dtype):
        """Dtype matches config."""
        from trtutils.builder._batcher import SyntheticBatcher

        batcher = SyntheticBatcher(
            shape=(32, 32, 3),
            dtype=dtype,
            num_batches=1,
        )
        batch = batcher.get_next_batch()
        assert batch.dtype == dtype

    def test_data_range(self):
        """Values fall within configured data_range."""
        from trtutils.builder._batcher import SyntheticBatcher

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
class TestSyntheticBatcherIteration:
    """Tests for SyntheticBatcher batch retrieval."""

    @pytest.mark.parametrize("num_batches", [1, 5, 10], ids=["1_batch", "5_batches", "10_batches"])
    def test_correct_batch_count(self, num_batches):
        """Exact num_batches batches are produced."""
        from trtutils.builder._batcher import SyntheticBatcher

        batcher = SyntheticBatcher(
            shape=(32, 32, 3),
            dtype=np.float32,
            num_batches=num_batches,
        )
        count = 0
        while batcher.get_next_batch() is not None:
            count += 1
        assert count == num_batches

    def test_exhausted_returns_none(self):
        """get_next_batch returns None after all batches consumed."""
        from trtutils.builder._batcher import SyntheticBatcher

        batcher = SyntheticBatcher(
            shape=(224, 224, 3),
            dtype=np.float32,
            num_batches=2,
        )
        batcher.get_next_batch()
        batcher.get_next_batch()
        assert batcher.get_next_batch() is None

    def test_num_batches_property(self):
        """Returns configured count."""
        from trtutils.builder._batcher import SyntheticBatcher

        batcher = SyntheticBatcher(
            shape=(32, 32, 3),
            dtype=np.float32,
            num_batches=7,
        )
        assert batcher.num_batches == 7

    def test_batch_size_property(self):
        """Returns configured batch size."""
        from trtutils.builder._batcher import SyntheticBatcher

        batcher = SyntheticBatcher(
            shape=(32, 32, 3),
            dtype=np.float32,
            batch_size=16,
        )
        assert batcher.batch_size == 16

    def test_c_contiguous_output(self):
        """Output is C-contiguous."""
        from trtutils.builder._batcher import SyntheticBatcher

        batcher = SyntheticBatcher(
            shape=(224, 224, 3),
            dtype=np.float32,
            num_batches=1,
        )
        batch = batcher.get_next_batch()
        assert batch.flags["C_CONTIGUOUS"]

    def test_dtype_conversion(self):
        """Data is converted to requested dtype."""
        from trtutils.builder._batcher import SyntheticBatcher

        batcher = SyntheticBatcher(
            shape=(224, 224, 3),
            dtype=np.float16,
            num_batches=1,
        )
        batch = batcher.get_next_batch()
        assert batch.dtype == np.float16
