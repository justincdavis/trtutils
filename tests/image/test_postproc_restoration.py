# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/postprocessors/_cpu_restoration.py and the restorer padding."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image._image_restorer import pad_to_input
from trtutils.image.postprocessors import get_restored_images, postprocess_restoration

pytestmark = pytest.mark.cpu


def _reference(chw: np.ndarray) -> np.ndarray:
    """Real-ESRGAN's own conversion: clamp, RGB to BGR, HWC, * 255, round, uint8."""
    image = np.clip(chw.astype(np.float32), 0, 1)[[2, 1, 0]].transpose(1, 2, 0)
    return (image * 255.0).round().astype(np.uint8)


@pytest.mark.parametrize(
    ("batch", "dtype"),
    [
        pytest.param(1, np.float32, id="batch1-fp32"),
        pytest.param(2, np.float32, id="batch2-fp32"),
        pytest.param(2, np.float16, id="batch2-fp16"),
    ],
)
@pytest.mark.parametrize(
    ("sizes", "scale"),
    [
        pytest.param(None, 1, id="whole"),
        pytest.param([(6, 9), (12, 16)], 2, id="crop-x2"),
        pytest.param([(3, 4), (2, 1)], 4, id="crop-x4"),
    ],
)
def test_postprocess_restoration_matches_reference(batch, dtype, sizes, scale) -> None:
    """Images equal upstream's conversion, cropped to scale * size, as new contiguous arrays."""
    rng = np.random.default_rng(0)
    out = rng.uniform(-0.2, 1.2, (batch, 3, 24, 32)).astype(dtype)
    # exact half levels, where rounding mode matters
    out[:, :, 0, :4] = np.array([0.5, 1.5, 2.5, 254.5], dtype=np.float32) / 255.0
    sizes = sizes[:batch] if sizes is not None else None
    results = postprocess_restoration([out], sizes, scale)
    assert len(results) == batch
    for i, result in enumerate(results):
        height, width = (24, 32) if sizes is None else (sizes[i][0] * scale, sizes[i][1] * scale)
        image = result[0]
        np.testing.assert_array_equal(image, _reference(out[i])[:height, :width])
        assert image.shape == (height, width, 3)
        assert image.dtype == np.uint8
        assert image.flags.c_contiguous
        assert not np.shares_memory(image, out)
    images = get_restored_images(results)
    assert all(image is result[0] for image, result in zip(images, results))


def test_postprocess_restoration_rejects_size_count() -> None:
    """Sizes must give one (height, width) per image of the batch."""
    out = np.zeros((2, 3, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected 2 image sizes, got 1"):
        postprocess_restoration([out], [(4, 4)], 2)


@pytest.mark.parametrize(
    ("shape", "pad_hw"),
    [
        pytest.param((20, 30, 3), (0, 0), id="exact"),
        pytest.param((15, 30, 3), (5, 0), id="bottom"),
        pytest.param((20, 22, 3), (0, 8), id="right"),
        pytest.param((12, 17, 3), (8, 13), id="both"),
    ],
)
def test_pad_to_input_mirrors_bottom_right(shape, pad_hw) -> None:
    """Smaller images are padded at the bottom and right by reflection, never scaled."""
    image = np.random.default_rng(1).integers(0, 256, shape, dtype=np.uint8)
    padded = pad_to_input(image, (30, 20))
    assert padded.shape == (20, 30, 3)
    np.testing.assert_array_equal(padded[: shape[0], : shape[1]], image)
    expected = np.pad(image, ((0, pad_hw[0]), (0, pad_hw[1]), (0, 0)), mode="reflect")
    np.testing.assert_array_equal(padded, expected)
    if pad_hw == (0, 0):
        assert padded is image


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((21, 30, 3), id="taller"),
        pytest.param((20, 31, 3), id="wider"),
    ],
)
def test_pad_to_input_rejects_larger_image(shape) -> None:
    """An image larger than the engine input raises, pointing at building at the image size."""
    image = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="build the engine at the image size"):
        pad_to_input(image, (30, 20))
