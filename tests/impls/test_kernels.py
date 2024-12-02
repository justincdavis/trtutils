# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import math
from pathlib import Path

import cv2
from cv2ext.image import letterbox
import numpy as np

from trtutils.core import Kernel, create_stream, destroy_stream, create_binding, create_kernel_args, memcpy_host_to_device_async, memcpy_device_to_host_async, stream_synchronize
from trtutils.impls import kernels
from trtutils.impls.yolo import preprocess


def test_scale_swap_transpose_compile():
    sst = Kernel(*kernels.SCALE_SWAP_TRANSPOSE)
    assert sst is not None

def test_letterbox_compile():
    letterbox = Kernel(*kernels.LETTERBOX_RESIZE)
    assert letterbox is not None

def test_linear_compile():
    linear = Kernel(*kernels.LINEAR_RESIZE)
    assert linear is not None


def test_sst_results():
    output_shape = (640, 640)
    scale = 1.0 / 255.0
    offset = 0.0

    img = cv2.imread(
        str(Path(__file__).parent.parent.parent / "data" / "horse.jpg")
    )
    img = cv2.resize(img, (640, 640))

    stream = create_stream()

    # block and thread info
    num_threads: tuple[int, int, int] = (32, 32, 1)
    num_blocks: tuple[int, int, int] = (
        math.ceil(output_shape[1] / num_threads[0]),
        math.ceil(output_shape[0] / num_threads[1]),
        1,
    )

    # allocate input/output binding
    dummy_input: np.ndarray = np.zeros(
        (output_shape[1], output_shape[0], 3),
        dtype=np.uint8,
    )
    # set is_input since we do not use the host_allocation here
    input_binding = create_binding(
        dummy_input,
        is_input=True,
    )
    dummy_output: np.ndarray = np.zeros(
        (1, 3, output_shape[1], output_shape[0]),
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

    # args = create_sst_args()
    args = create_kernel_args(
        input_binding.allocation,
        output_binding.allocation,
        scale,
        offset,
        640,
        640,
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
    destroy_stream(stream)

    cuda_result = output_binding.host_allocation

    cpu_result, _, _ = preprocess(img, (640, 640), np.float32)

    assert cuda_result.shape == cpu_result.shape
    assert np.allclose(cuda_result, cpu_result)


# def test_letterbox_results():
#     output_shape = (640, 640)

#     img = cv2.imread(
#         str(Path(__file__).parent.parent.parent / "data" / "horse.jpg")
#     )
#     letterboxed_img, _, _ = letterbox(img, (640, 640))

#     stream = create_stream()

#     # block and thread info
#     num_threads: tuple[int, int, int] = (32, 32, 1)
#     num_blocks: tuple[int, int, int] = (
#         math.ceil(output_shape[1] / num_threads[0]),
#         math.ceil(output_shape[0] / num_threads[1]),
#         1,
#     )

#     # set is_input since we do not use the host_allocation here
#     input_binding = create_binding(
#         img,
#         is_input=True,
#     )
#     dummy_output: np.ndarray = np.zeros(
#         (output_shape[1], output_shape[0], 3),
#         dtype=np.float32,
#     )
#     # set pagelocked memory since we read from the host allocation
#     output_binding = create_binding(
#         dummy_output,
#         pagelocked_mem=True,
#     )

#     # load the kernel
#     kernel = Kernel(
#         *kernels.LETTERBOX_RESIZE,
#     )

#     height, width = img.shape[:2]
#     o_width, o_height = output_shape
#     scale_x = o_width / width
#     scale_y = o_height / height
#     scale = min(scale_x, scale_y)
#     padding_x = (width - (o_width * scale)) / 2
#     padding_y = (height - (o_height * scale)) / 2
#     new_width = int(width * scale)
#     new_height = int(height * scale)

#     # args = create_sst_args()
#     args = create_kernel_args(
#         input_binding.allocation,
#         output_binding.allocation,
#         width,
#         height,
#         640,
#         640,
#         scale_x,
#         scale_y,
#         padding_x,
#         padding_y,
#         new_width,
#         new_height,
#     )

#     memcpy_host_to_device_async(
#         input_binding.allocation,
#         img,
#         stream,
#     )

#     kernel.call(num_blocks, num_threads, stream, args)

#     memcpy_device_to_host_async(
#         output_binding.host_allocation,
#         output_binding.allocation,
#         stream,
#     )

#     stream_synchronize(stream)
#     destroy_stream(stream)

#     cuda_result = output_binding.host_allocation

#     assert cuda_result.shape == letterboxed_img.shape

#     # cv2.imshow("original", img)
#     # cv2.imshow("CPU-letterbox", letterboxed_img)
#     # cv2.imshow("CUDA-letterbox", cuda_result)
#     # cv2.waitKey(0)

#     assert False  # the output is broken


def test_linear_results():
    output_shape = (1920, 1080)

    img = cv2.imread(
        str(Path(__file__).parent.parent.parent / "data" / "horse.jpg")
    )
    resized_img = cv2.resize(img, output_shape, interpolation=cv2.INTER_NEAREST)

    height, width = img.shape[:2]
    o_width, o_height = output_shape

    stream = create_stream()

    # block and thread info
    num_threads: tuple[int, int, int] = (32, 32, 1)
    num_blocks: tuple[int, int, int] = (
        int(o_width / num_threads[0]),
        int(o_height / num_threads[1]),
        1,
    )

    # set is_input since we do not use the host_allocation here
    input_binding = create_binding(
        img,
        is_input=True,
    )
    dummy_output: np.ndarray = np.zeros(
        (o_height, o_width, 3),
        dtype=np.float32,
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

    # args = create_sst_args()
    args = create_kernel_args(
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
    destroy_stream(stream)

    cuda_result = output_binding.host_allocation

    assert cuda_result.shape == resized_img.shape

    cv2.imshow("original", img)
    cv2.imshow("CPU-resize", resized_img)
    cv2.imshow("CUDA-resize", cuda_result)
    cv2.waitKey(0)
