# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="unused-ignore"
from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

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
    from .common import IMG_PATH
except ImportError:
    from common import (
        IMG_PATH,
    )

if TYPE_CHECKING:
    from pathlib import Path


def _get_kernel_timings(kernel_data: tuple[Path, str]) -> list[float]:
    output_height = 640
    output_width = 640
    batch_size = 1
    scale = 1.0 / 255.0
    offset = 0.0

    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (output_width, output_height))  # type: ignore[arg-type]

    stream = create_stream()

    # block and thread info
    num_threads: tuple[int, int, int] = (32, 32, 1)
    num_blocks: tuple[int, int, int] = (
        math.ceil(output_width / num_threads[0]),
        math.ceil(output_height / num_threads[1]),
        batch_size,
    )

    # allocate input/output binding
    dummy_input: np.ndarray = np.zeros(
        (output_height, output_width, 3),
        dtype=np.uint8,
    )
    # set is_input since we do not use the host_allocation here
    input_binding = create_binding(
        dummy_input,
        is_input=True,
    )
    dummy_output: np.ndarray = np.zeros(
        (1, 3, output_height, output_width),
        dtype=np.float32,
    )
    # set pagelocked memory since we read from the host allocation
    output_binding = create_binding(
        dummy_output,
        pagelocked_mem=True,
    )

    # load the kernel
    kernel = Kernel(
        kernel_data[0],
        kernel_data[1],
    )

    # do entire normal setup for the kernel
    args = kernel.create_args(
        input_binding.allocation,
        output_binding.allocation,
        scale,
        offset,
        output_height,
        output_width,
        batch_size,
    )

    memcpy_host_to_device_async(
        input_binding.allocation,
        img,
        stream,
    )

    kernel.call(num_blocks, num_threads, stream, args)

    memcpy_device_to_host_async(
        output_binding.host_allocation,
        output_binding.allocation,
        stream,
    )

    stream_synchronize(stream)

    # do the big timing loop
    timings: list[float] = []
    for _ in range(1000):
        t0 = time.time()
        kernel.call(num_blocks, num_threads, stream, args)
        stream_synchronize(stream)
        t1 = time.time()
        timings.append(t1 - t0)

    destroy_stream(stream)
    input_binding.free()
    output_binding.free()
    kernel.free()

    return timings


def test_sst_performance() -> None:
    """Test the performance of the scale-swap-transpose kernel."""
    sst_fast = _get_kernel_timings(kernels.SST_FAST)
    sst = _get_kernel_timings(kernels.SCALE_SWAP_TRANSPOSE)

    sst_fast_mean = np.mean(sst_fast)
    sst_mean = np.mean(sst)

    print(f"sst_fast_mean: {sst_fast_mean}, sst_mean: {sst_mean}")
    print(f"speedup: {sst_mean / sst_fast_mean}")

    assert sst_fast_mean < sst_mean


if __name__ == "__main__":
    test_sst_performance()
