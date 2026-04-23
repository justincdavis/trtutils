# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc"
"""
Performance benchmarks for image preprocessing CUDA kernels.

Port from: tests/legacy/image/kernels/test_sst_performance.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

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


def _get_kernel_timings(kernel_data: tuple[Any, Any], n_iter: int = 100) -> list[float]:
    """Measure kernel execution timings."""
    output_height = 640
    output_width = 640
    batch_size = 1
    scale = 1.0 / 255.0
    offset = 0.0

    if not _HORSE_IMAGE_PATH.exists():
        pytest.skip("Horse test image not found")
    img = cv2.imread(str(_HORSE_IMAGE_PATH))
    if img is None:
        pytest.skip("Failed to read test image")
    img = cv2.resize(img, (output_width, output_height))  # type: ignore[arg-type]

    stream = create_stream()

    num_threads: tuple[int, int, int] = (32, 32, 1)
    num_blocks: tuple[int, int, int] = (
        math.ceil(output_width / num_threads[0]),
        math.ceil(output_height / num_threads[1]),
        batch_size,
    )

    dummy_input = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    input_binding = create_binding(dummy_input, is_input=True)
    dummy_output = np.zeros((1, 3, output_height, output_width), dtype=np.float32)
    output_binding = create_binding(dummy_output, pagelocked_mem=True)

    kernel_obj = Kernel(kernel_data[0], kernel_data[1])
    args = kernel_obj.create_args(
        input_binding.allocation,
        output_binding.allocation,
        scale,
        offset,
        output_height,
        output_width,
        batch_size,
    )

    memcpy_host_to_device_async(input_binding.allocation, img, stream)
    kernel_obj.call(num_blocks, num_threads, stream, args)
    memcpy_device_to_host_async(output_binding.host_allocation, output_binding.allocation, stream)
    stream_synchronize(stream)

    timings: list[float] = []
    for _ in range(n_iter):
        t0 = time.time()
        kernel_obj.call(num_blocks, num_threads, stream, args)
        stream_synchronize(stream)
        t1 = time.time()
        timings.append(t1 - t0)

    destroy_stream(stream)
    input_binding.free()
    output_binding.free()
    kernel_obj.free()

    return timings


@pytest.mark.performance
class TestKernelPerformance:
    """Performance benchmarks for CUDA image kernels."""

    def test_sst_fast_faster_than_sst(self) -> None:
        """SST_FAST kernel is faster than standard SCALE_SWAP_TRANSPOSE."""
        sst_fast_timings = _get_kernel_timings(kernels.SST_FAST)
        sst_timings = _get_kernel_timings(kernels.SCALE_SWAP_TRANSPOSE)

        sst_fast_mean = float(np.mean(sst_fast_timings))
        sst_mean = float(np.mean(sst_timings))

        print(
            f"SST_FAST mean: {sst_fast_mean:.6f}s, SST mean: {sst_mean:.6f}s,"
            f" speedup: {sst_mean / sst_fast_mean:.2f}x"
        )
        assert sst_fast_mean < sst_mean

    def test_sst_benchmark(self) -> None:
        """Benchmark standard SST kernel timing."""
        timings = _get_kernel_timings(kernels.SCALE_SWAP_TRANSPOSE)
        mean_time = float(np.mean(timings))
        print(f"SST kernel: {mean_time * 1000:.3f}ms avg over {len(timings)} iterations")
        assert mean_time < 0.1  # should be well under 100ms per iteration

    def test_sst_fast_benchmark(self) -> None:
        """Benchmark SST_FAST kernel timing."""
        timings = _get_kernel_timings(kernels.SST_FAST)
        mean_time = float(np.mean(timings))
        print(f"SST_FAST kernel: {mean_time * 1000:.3f}ms avg over {len(timings)} iterations")
        assert mean_time < 0.1  # should be well under 100ms per iteration
