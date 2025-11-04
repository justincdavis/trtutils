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


def test_sst_imagenet_results() -> None:
    """Test scale-swap-transpose imagenet kernel results against CPU implementation."""
    output_shape = 640
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (output_shape, output_shape))

    stream = create_stream()

    # block and thread info
    num_threads: tuple[int, int, int] = (32, 32, 1)
    num_blocks: tuple[int, int, int] = (
        math.ceil(output_shape / num_threads[0]),
        math.ceil(output_shape / num_threads[1]),
        1,
    )

    # allocate input/output binding
    dummy_input: np.ndarray = np.zeros(
        (output_shape, output_shape, 3),
        dtype=np.uint8,
    )
    # set is_input since we do not use the host_allocation here
    input_binding = create_binding(
        dummy_input,
        is_input=True,
    )
    dummy_output: np.ndarray = np.zeros(
        (1, 3, output_shape, output_shape),
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
        output_shape,
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
        (output_shape, output_shape),
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


if __name__ == "__main__":
    test_sst_imagenet_results()
