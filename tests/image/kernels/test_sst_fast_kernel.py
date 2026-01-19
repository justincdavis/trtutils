# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import math

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
from trtutils.image.preprocessors import preprocess

try:
    from .common import IMG_PATH, kernel_compile
except ImportError:
    from common import (  # type: ignore[no-redef, import-not-found]
        IMG_PATH,
        kernel_compile,
    )


def test_sst_fast_compile() -> None:
    """Test compilation of the scale-swap-transpose kernel."""
    kernel_compile(kernels.SST_FAST)


def test_sst_fast_f16_compile() -> None:
    """Test compilation of the scale-swap-transpose fp16 kernel."""
    kernel_compile(kernels.SST_FAST_F16)


def test_sst_fast_results() -> None:
    """Test scale-swap-transpose kernel results against CPU implementation."""
    output_height = 640
    output_width = 640
    batch_size = 1
    scale = 1.0 / 255.0
    offset = 0.0

    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (output_width, output_height))

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
        *kernels.SST_FAST,
    )

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

    cuda_result = output_binding.host_allocation

    cpu_result, _, _ = preprocess([img], (output_width, output_height), dummy_output.dtype)

    assert cuda_result.shape == cpu_result.shape
    assert np.mean(cuda_result) == np.mean(cpu_result)
    assert np.min(cuda_result) == np.min(cpu_result)  # type: ignore[operator]
    assert np.max(cuda_result) == np.max(cpu_result)  # type: ignore[operator]
    assert np.allclose(cuda_result, cpu_result)

    destroy_stream(stream)
    input_binding.free()
    output_binding.free()
    kernel.free()


def test_sst_fast_batch_results() -> None:
    """Test scale-swap-transpose kernel with batch processing."""
    output_height = 640
    output_width = 640
    batch_size = 4
    scale = 1.0 / 255.0
    offset = 0.0

    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (output_width, output_height))

    # create batch input by stacking the same image
    batch_input = np.stack([img] * batch_size, axis=0)  # (N, H, W, 3)

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
        (batch_size, output_height, output_width, 3),
        dtype=np.uint8,
    )
    input_binding = create_binding(
        dummy_input,
        is_input=True,
    )
    dummy_output: np.ndarray = np.zeros(
        (batch_size, 3, output_height, output_width),
        dtype=np.float32,
    )
    output_binding = create_binding(
        dummy_output,
        pagelocked_mem=True,
    )

    kernel = Kernel(*kernels.SST_FAST)

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
        batch_input,
        stream,
    )

    kernel.call(num_blocks, num_threads, stream, args)

    memcpy_device_to_host_async(
        output_binding.host_allocation,
        output_binding.allocation,
        stream,
    )

    stream_synchronize(stream)

    cuda_result = output_binding.host_allocation

    # get CPU result for single image
    cpu_result, _, _ = preprocess([img], (output_width, output_height), dummy_output.dtype)

    # verify each batch element matches the CPU result
    for i in range(batch_size):
        assert cuda_result[i].shape == cpu_result[0].shape
        assert np.allclose(cuda_result[i], cpu_result[0])

    destroy_stream(stream)
    input_binding.free()
    output_binding.free()
    kernel.free()


def test_sst_fast_f16_results() -> None:
    """Test scale-swap-transpose fp16 kernel results against CPU implementation."""
    output_height = 640
    output_width = 640
    batch_size = 1
    scale = 1.0 / 255.0
    offset = 0.0

    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (output_width, output_height))

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
    input_binding = create_binding(
        dummy_input,
        is_input=True,
    )
    dummy_output: np.ndarray = np.zeros(
        (1, 3, output_height, output_width),
        dtype=np.float16,
    )
    output_binding = create_binding(
        dummy_output,
        pagelocked_mem=True,
    )

    kernel = Kernel(*kernels.SST_FAST_F16)

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

    cuda_result = output_binding.host_allocation

    cpu_result, _, _ = preprocess([img], (output_width, output_height), np.float32)

    assert cuda_result.shape == cpu_result.shape
    # use relaxed tolerances for fp16
    assert np.isclose(np.mean(cuda_result), np.mean(cpu_result), rtol=1e-3, atol=1e-3)
    assert np.allclose(cuda_result, cpu_result, rtol=1e-3, atol=1e-3)

    destroy_stream(stream)
    input_binding.free()
    output_binding.free()
    kernel.free()


if __name__ == "__main__":
    test_sst_fast_results()
    test_sst_fast_batch_results()
    test_sst_fast_f16_results()
