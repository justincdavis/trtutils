# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from trtutils.core import Kernel, create_stream, destroy_stream, create_binding, memcpy_host_to_device_async, memcpy_device_to_host_async, stream_synchronize
from trtutils.impls import kernels
from trtutils.impls.yolo import preprocess


def test_scale_swap_transpose_compile():
    sst = Kernel(*kernels.SCALE_SWAP_TRANSPOSE)
    assert sst is not None


def test_sst_results():
    output_shape = 640
    scale = 1.0 / 255.0
    offset = 0.0

    img = cv2.imread(
        str(Path(__file__).parent.parent.parent / "data" / "horse.jpg")
    )
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

    # load the kernel
    kernel = Kernel(
        *kernels.SCALE_SWAP_TRANSPOSE,
    )

    args = kernel.create_args(
        input_binding.allocation,
        output_binding.allocation,
        scale,
        offset,
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

    cpu_result, _, _ = preprocess(img, (output_shape, output_shape), dummy_output.dtype)

    assert cuda_result.shape == cpu_result.shape
    assert np.allclose(cuda_result, cpu_result)

    destroy_stream(stream)
    input_binding.free()
    output_binding.free()
    kernel.free()
