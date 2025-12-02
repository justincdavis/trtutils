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


def test_sst_imagenet_compile() -> None:
    """Test compilation of the scale-swap-transpose imagenet kernel."""
    kernel_compile(kernels.IMAGENET_SST)


def test_sst_imagenet_f16_compile() -> None:
    """Test compilation of the scale-swap-transpose imagenet fp16 kernel."""
    kernel_compile(kernels.IMAGENET_SST_F16)


def test_sst_imagenet_results() -> None:
    """Test scale-swap-transpose imagenet kernel results against CPU implementation."""
    output_height = 640
    output_width = 640
    batch_size = 1
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

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

    # allocate mean and std bindings (as (1, 3, 1, 1) arrays to match preprocessor)
    mean_array = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_array = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
    mean_binding = create_binding(mean_array)
    std_binding = create_binding(std_array)

    # load the kernel
    kernel = Kernel(
        *kernels.IMAGENET_SST,
    )

    # copy mean and std to device
    memcpy_host_to_device_async(
        mean_binding.allocation,
        mean_array,
        stream,
    )
    memcpy_host_to_device_async(
        std_binding.allocation,
        std_array,
        stream,
    )

    args = kernel.create_args(
        input_binding.allocation,
        output_binding.allocation,
        mean_binding.allocation,
        std_binding.allocation,
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

    cpu_result, _, _ = preprocess(
        img,
        (output_width, output_height),
        dummy_output.dtype,
        mean=mean,
        std=std,
    )

    assert cuda_result.shape == cpu_result.shape
    # use approximate equality for floating point comparisons
    # small differences can occur due to GPU vs CPU floating point operations
    assert np.isclose(np.mean(cuda_result), np.mean(cpu_result), rtol=1e-6, atol=1e-6)
    assert np.isclose(np.min(cuda_result), np.min(cpu_result), rtol=1e-6, atol=1e-6)  # type: ignore[operator]
    assert np.isclose(np.max(cuda_result), np.max(cpu_result), rtol=1e-6, atol=1e-6)  # type: ignore[operator]
    assert np.allclose(cuda_result, cpu_result, rtol=1e-6, atol=1e-6)

    destroy_stream(stream)
    input_binding.free()
    output_binding.free()
    mean_binding.free()
    std_binding.free()
    kernel.free()


def test_sst_imagenet_batch_results() -> None:
    """Test scale-swap-transpose imagenet kernel with batch processing."""
    output_height = 640
    output_width = 640
    batch_size = 4
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

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

    # allocate mean and std bindings
    mean_array = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_array = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
    mean_binding = create_binding(mean_array)
    std_binding = create_binding(std_array)

    kernel = Kernel(*kernels.IMAGENET_SST)

    # copy mean and std to device
    memcpy_host_to_device_async(
        mean_binding.allocation,
        mean_array,
        stream,
    )
    memcpy_host_to_device_async(
        std_binding.allocation,
        std_array,
        stream,
    )

    args = kernel.create_args(
        input_binding.allocation,
        output_binding.allocation,
        mean_binding.allocation,
        std_binding.allocation,
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
    cpu_result, _, _ = preprocess(
        img,
        (output_width, output_height),
        dummy_output.dtype,
        mean=mean,
        std=std,
    )

    # verify each batch element matches the CPU result
    for i in range(batch_size):
        assert cuda_result[i].shape == cpu_result[0].shape
        assert np.allclose(cuda_result[i], cpu_result[0], rtol=1e-6, atol=1e-6)

    destroy_stream(stream)
    input_binding.free()
    output_binding.free()
    mean_binding.free()
    std_binding.free()
    kernel.free()


def test_sst_imagenet_f16_results() -> None:
    """Test scale-swap-transpose imagenet fp16 kernel results against CPU implementation."""
    output_height = 640
    output_width = 640
    batch_size = 1
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

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

    # allocate mean and std bindings
    mean_array = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_array = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
    mean_binding = create_binding(mean_array)
    std_binding = create_binding(std_array)

    kernel = Kernel(*kernels.IMAGENET_SST_F16)

    # copy mean and std to device
    memcpy_host_to_device_async(
        mean_binding.allocation,
        mean_array,
        stream,
    )
    memcpy_host_to_device_async(
        std_binding.allocation,
        std_array,
        stream,
    )

    args = kernel.create_args(
        input_binding.allocation,
        output_binding.allocation,
        mean_binding.allocation,
        std_binding.allocation,
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

    cpu_result, _, _ = preprocess(
        img,
        (output_width, output_height),
        np.float32,
        mean=mean,
        std=std,
    )

    assert cuda_result.shape == cpu_result.shape
    # use relaxed tolerances for fp16
    assert np.isclose(np.mean(cuda_result), np.mean(cpu_result), rtol=1e-3, atol=1e-3)
    assert np.allclose(cuda_result, cpu_result, rtol=1e-3, atol=1e-3)

    destroy_stream(stream)
    input_binding.free()
    output_binding.free()
    mean_binding.free()
    std_binding.free()
    kernel.free()


if __name__ == "__main__":
    test_sst_imagenet_results()
    test_sst_imagenet_batch_results()
    test_sst_imagenet_f16_results()
