# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import math

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
from trtutils.image.postprocessors.detection import postprocess_detections

try:
    from .common import kernel_compile
except ImportError:
    from common import (  # type: ignore[no-redef, import-not-found]
        kernel_compile,
    )


def test_rescale_compile() -> None:
    """Test compilation of the rescale detections kernel."""
    kernel_compile(kernels.RESCALE_DETECTIONS)


def test_rescale_v10_results() -> None:
    """
    Test rescale detections kernel results for YOLOv10.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.

    """
    topk = 300
    input_shape = (1, topk, 6)
    ratios = (1.0, 1.0)
    padding = (32.0, 32.0)
    conf_thres = 0.25

    stream = create_stream()

    # block and thread info
    num_threads: tuple[int, int, int] = (256, 1, 1)
    num_blocks: tuple[int, int, int] = (
        math.ceil(topk / num_threads[0]),
        1,
        1,
    )

    dummy_yolo_v10_boxes: np.ndarray = np.zeros(
        input_shape,
        dtype=np.float32,
    )
    box_mask: np.ndarray = np.zeros(
        (topk,),
        dtype=bool,
    )
    input_binding = create_binding(
        dummy_yolo_v10_boxes,
        pagelocked_mem=True,
    )

    # fill input with deterministic random dummy data
    rng = np.random.default_rng(0)
    host_boxes = input_binding.host_allocation
    host_boxes[..., 0:4] = rng.uniform(0, 1024, size=host_boxes[..., 0:4].shape).astype(np.float32)
    host_boxes[..., 4] = rng.uniform(0, 1, size=host_boxes[..., 4].shape).astype(np.float32)
    host_boxes[..., 5] = rng.integers(0, 80, size=host_boxes[..., 5].shape).astype(np.float32)
    memcpy_host_to_device_async(
        input_binding.allocation,
        input_binding.host_allocation,
        stream=stream,
    )
    dummy_output: np.ndarray = np.zeros(input_shape, dtype=np.float32)
    output_binding = create_binding(
        dummy_output,
        pagelocked_mem=True,
    )
    mask_binding = create_binding(
        box_mask,
        pagelocked_mem=True,
    )

    # process detections on the host side
    num_dets = np.array((1, 1), dtype=np.int32)
    num_dets[0] = 10
    scores = np.array((1, topk), dtype=np.float32)
    classes = np.array((1, topk), dtype=np.int32)
    host_outputs = postprocess_detections(
        [
            num_dets,
            host_boxes,
            scores,
            classes,
        ],
        ratios=ratios,
        padding=padding,
        conf_thres=conf_thres,
    )

    # load the kernel
    kernel = Kernel(
        *kernels.RESCALE_DETECTIONS,
    )

    args = kernel.create_args(
        input_binding.allocation,
        output_binding.allocation,
        mask_binding.allocation,
        topk,
        6,
        conf_thres,
        ratios[0],
        ratios[1],
        padding[0],
        padding[1],
    )

    kernel.call(
        num_blocks=num_blocks,
        num_threads=num_threads,
        stream=stream,
        args=args,
    )

    memcpy_device_to_host_async(
        output_binding.host_allocation,
        output_binding.allocation,
        stream=stream,
    )
    memcpy_device_to_host_async(
        mask_binding.host_allocation,
        mask_binding.allocation,
        stream=stream,
    )
    stream_synchronize(stream)

    output = output_binding.host_allocation
    mask = mask_binding.host_allocation

    assert output.shape == input_shape
    assert mask.shape == (topk,)

    

    input_binding.free()
    output_binding.free()
    mask_binding.free()
    kernel.free()
    destroy_stream(stream)


if __name__ == "__main__":
    test_rescale_compile()
    test_rescale_v10_results()
