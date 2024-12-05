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


def test_linear_compile():
    linear = Kernel(*kernels.LINEAR_RESIZE)
    assert linear is not None


def test_linear_results():
    output_shape = (640, 480)

    img = cv2.imread(
        str(Path(__file__).parent.parent.parent / "data" / "horse.jpg")
    )
    resized_img = cv2.resize(img, output_shape, interpolation=cv2.INTER_LINEAR)

    height, width = img.shape[:2]
    o_width, o_height = output_shape

    stream = create_stream()

    # block and thread info
    num_threads: tuple[int, int, int] = (32, 32, 1)
    num_blocks: tuple[int, int, int] = (
        math.ceil(o_width / num_threads[1]),  # X-axis (width)
        math.ceil(o_height / num_threads[0]),  # Y-axis (height)
        1,
    )

    # set is_input since we do not use the host_allocation here
    input_binding = create_binding(
        img,
        is_input=True,
    )
    dummy_output: np.ndarray = np.zeros(
        (o_height, o_width, 3),
        dtype=np.uint8,
    )
    # set pagelocked memory since we read from the host allocation
    output_binding = create_binding(
        dummy_output,
        pagelocked_mem=True,
    )

    # load the kernel
    kernel = Kernel(
        *kernels.LINEAR_RESIZE,
    )

    args = kernel.create_args(
        input_binding.allocation,
        output_binding.allocation,
        width,
        height,
        o_width,
        o_height,
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

    assert cuda_result.shape == resized_img.shape
    cpu_mean = np.mean(resized_img)
    # allow up to an overall 0.5 out of 255.0 drift (1.0 abs)
    assert cpu_mean - 0.5 <= np.mean(cuda_result) <= cpu_mean + 0.5

    # cv2.imshow("CPU", resized_img)
    # cv2.imshow("CUDA", cuda_result)
    # cv2.waitKey(0)

    destroy_stream(stream)
    input_binding.free()
    output_binding.free()
    kernel.free()
