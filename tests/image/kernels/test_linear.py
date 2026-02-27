# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc"
"""Tests for the linear resize CUDA kernel."""

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

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
_HORSE_IMAGE_PATH = _DATA_DIR / "horse.jpg"


def _run_linear_kernel(
    img: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Run the linear resize kernel and return result."""
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

    kernel = Kernel(kernels.LINEAR_RESIZE[0], kernels.LINEAR_RESIZE[1])
    args = kernel.create_args(
        input_binding.allocation,
        output_binding.allocation,
        width,
        height,
        o_width,
        o_height,
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
class TestLinearResizeKernel:
    """Tests for the linear resize CUDA kernel."""

    def test_compiles(self) -> None:
        """Linear resize kernel compiles without error."""
        stream = create_stream()
        compiled = Kernel(kernels.LINEAR_RESIZE[0], kernels.LINEAR_RESIZE[1])
        assert compiled is not None
        destroy_stream(stream)

    def test_correctness_against_cv2(self) -> None:
        """GPU linear resize matches cv2.resize(INTER_LINEAR)."""
        if not _HORSE_IMAGE_PATH.exists():
            pytest.skip("Horse test image not found")
        img = cv2.imread(str(_HORSE_IMAGE_PATH))
        if img is None:
            pytest.skip("Failed to read test image")

        output_shape = (640, 480)
        o_width, o_height = output_shape
        resized_img = np.asarray(
            cv2.resize(img, (o_width, o_height), interpolation=cv2.INTER_LINEAR)
        )
        cuda_result = _run_linear_kernel(img, output_shape)

        assert cuda_result.shape == resized_img.shape
        cpu_mean = float(resized_img.mean())
        assert cpu_mean - 0.5 <= np.mean(cuda_result) <= cpu_mean + 0.5
        diff_mask = np.any(resized_img != cuda_result, axis=-1)
        avg_diff = np.mean(np.abs(resized_img[diff_mask] - cuda_result[diff_mask]))
        assert avg_diff < 1.0

    @pytest.mark.parametrize(
        "output_shape",
        [(640, 640), (416, 416)],
        ids=["640x640", "416x416"],
    )
    def test_various_target_sizes(self, output_shape: tuple[int, int]) -> None:
        """Linear resize kernel works with various target sizes."""
        if not _HORSE_IMAGE_PATH.exists():
            pytest.skip("Horse test image not found")
        img = cv2.imread(str(_HORSE_IMAGE_PATH))
        if img is None:
            pytest.skip("Failed to read test image")

        o_width, o_height = output_shape
        result = _run_linear_kernel(img, output_shape)
        assert result.shape == (o_height, o_width, 3)
