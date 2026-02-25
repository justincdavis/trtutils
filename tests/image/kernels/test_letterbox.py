# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc"
"""Tests for the letterbox resize CUDA kernel."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from trtutils.core import (
    Kernel,
    create_binding,
    create_stream,
    destroy_stream,
    memcpy_device_to_host_async,
    memcpy_host_to_device_async,
    stream_synchronize,
)
from trtutils.image import kernels

try:
    from cv2ext.image import letterbox as cv2ext_letterbox  # type: ignore[import-untyped]

    _CV2EXT_AVAILABLE = True
except ImportError:
    cv2ext_letterbox = None  # type: ignore[assignment]
    _CV2EXT_AVAILABLE = False

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
_HORSE_IMAGE_PATH = _DATA_DIR / "horse.jpg"


def _run_letterbox_kernel(
    img: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Run the letterbox kernel and return result."""
    o_width, o_height = output_shape
    height, width = img.shape[:2]

    stream = create_stream()

    num_threads: tuple[int, int, int] = (32, 32, 1)
    num_blocks: tuple[int, int, int] = (
        math.ceil(o_width / num_threads[1]),
        math.ceil(o_height / num_threads[0]),
        1,
    )

    input_binding = create_binding(img, is_input=True)
    dummy_output = np.zeros((o_height, o_width, 3), dtype=np.uint8)
    output_binding = create_binding(dummy_output, pagelocked_mem=True)

    scale_x = o_width / width
    scale_y = o_height / height
    scale = min(scale_x, scale_y)
    new_width = int(width * scale)
    new_height = int(height * scale)
    pad_x = int((o_width - new_width) / 2)
    pad_y = int((o_height - new_height) / 2)

    kernel = Kernel(kernels.LETTERBOX_RESIZE[0], kernels.LETTERBOX_RESIZE[1])
    args = kernel.create_args(
        input_binding.allocation,
        output_binding.allocation,
        width,
        height,
        o_width,
        o_height,
        pad_x,
        pad_y,
        new_width,
        new_height,
    )

    memcpy_host_to_device_async(input_binding.allocation, img, stream)
    kernel.call(num_blocks, num_threads, stream, args)
    memcpy_device_to_host_async(output_binding.host_allocation, output_binding.allocation, stream)
    stream_synchronize(stream)

    result = output_binding.host_allocation.copy()

    destroy_stream(stream)
    input_binding.free()
    output_binding.free()
    kernel.free()

    return result


@pytest.mark.gpu
class TestLetterboxKernel:
    """Tests for the letterbox CUDA kernel."""

    def test_compiles(self) -> None:
        """Letterbox kernel compiles without error."""
        stream = create_stream()
        compiled = Kernel(kernels.LETTERBOX_RESIZE[0], kernels.LETTERBOX_RESIZE[1])
        assert compiled is not None
        destroy_stream(stream)

    @pytest.mark.skipif(not _CV2EXT_AVAILABLE, reason="cv2ext not installed")
    def test_correctness_against_cv2ext(self) -> None:
        """GPU letterbox result matches cv2ext.letterbox()."""
        if not _HORSE_IMAGE_PATH.exists():
            pytest.skip("Horse test image not found")
        img = cv2.imread(str(_HORSE_IMAGE_PATH))
        if img is None:
            pytest.skip("Failed to read test image")

        output_shape = (640, 480)
        assert cv2ext_letterbox is not None
        resized_img, _, _ = cv2ext_letterbox(img, output_shape)  # type: ignore[misc]
        cuda_result = _run_letterbox_kernel(img, output_shape)

        assert cuda_result.shape == resized_img.shape
        cpu_mean = np.mean(resized_img)
        assert cpu_mean - 0.5 <= np.mean(cuda_result) <= cpu_mean + 0.5
        diff_mask = np.any(resized_img != cuda_result, axis=-1)
        avg_diff = np.mean(np.abs(resized_img[diff_mask] - cuda_result[diff_mask]))
        assert avg_diff < 1.0

    @pytest.mark.parametrize(
        "output_shape",
        [(640, 640), (416, 416), (320, 320)],
        ids=["640x640", "416x416", "320x320"],
    )
    def test_various_target_sizes(self, output_shape: tuple[int, int]) -> None:
        """Letterbox kernel works with various target sizes."""
        if not _HORSE_IMAGE_PATH.exists():
            pytest.skip("Horse test image not found")
        img = cv2.imread(str(_HORSE_IMAGE_PATH))
        if img is None:
            pytest.skip("Failed to read test image")

        o_width, o_height = output_shape
        result = _run_letterbox_kernel(img, output_shape)
        assert result.shape == (o_height, o_width, 3)
