# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import math
import time

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


def _efficient_nms_outputs(topk: int = 100) -> list[np.ndarray]:
    np.random.seed(0)
    num_dets = np.zeros((1, 1), dtype=np.int32)
    bboxes = np.zeros((1, topk, 4), dtype=np.float32)
    scores = np.zeros((1, topk), dtype=np.float32)
    classes = np.zeros((1, topk), dtype=np.int32)
    num_dets[0] = 10
    bboxes[..., 0:4] = np.random.uniform(0, 1024, size=bboxes[..., 0:4].shape).astype(
        np.float32
    )
    scores[..., 0:10] = np.random.uniform(0.5, 1, size=scores[..., 0:10].shape).astype(
        np.float32
    )
    classes[..., 0:10] = np.random.randint(0, 80, size=classes[..., 0:10].shape).astype(
        np.int32
    )
    return [
        num_dets,
        bboxes,
        scores,
        classes,
    ]


def _v10_outputs(topk: int = 300) -> list[np.ndarray]:
    np.random.seed(0)
    boxes = np.zeros((1, topk, 6), dtype=np.float32)
    boxes[..., 0:4] = np.random.uniform(0, 1024, size=boxes[..., 0:4].shape).astype(
        np.float32
    )
    boxes[..., 4] = np.random.uniform(0, 1, size=boxes[..., 4].shape).astype(np.float32)
    boxes[..., 5] = np.random.randint(0, 80, size=boxes[..., 5].shape).astype(
        np.float32
    )
    return [
        boxes,
    ]


def test_rescale_v10_compile() -> None:
    """Test compilation of the rescale detections kernel."""
    kernel_compile(kernels.RESCALE_V10_DETECTIONS)


def test_rescale_eff_nms_compile() -> None:
    """Test compilation of the rescale detections kernel."""
    kernel_compile(kernels.RESCALE_EFF_NMS_DETECTIONS)


def test_rescale_v10_performance() -> None:
    """
    Test rescale detections kernel performance for YOLOv10.

    Compares to CPU side postprocessing performance.
    """
    topk = 300
    ratios = (1.0, 1.0)
    padding = (32.0, 32.0)
    conf_thres = 0.25
    iterations = 10000

    stream = create_stream()

    # block and thread info
    num_threads: tuple[int, int, int] = (256, 1, 1)
    num_blocks: tuple[int, int, int] = (
        math.ceil(topk / num_threads[0]),
        1,
        1,
    )

    mock_outputs = _v10_outputs(topk)

    box_mask: np.ndarray = np.zeros(
        (1, topk),
        dtype=bool,
    )
    input_binding = create_binding(
        mock_outputs[0],
        pagelocked_mem=True,
    )

    memcpy_host_to_device_async(
        input_binding.allocation,
        mock_outputs[0],
        stream=stream,
    )

    mask_binding = create_binding(
        box_mask,
        pagelocked_mem=True,
    )

    # load the kernel
    kernel = Kernel(
        *kernels.RESCALE_V10_DETECTIONS,
    )

    args = kernel.create_args(
        input_binding.allocation,
        mask_binding.allocation,
        topk,
        6,
        conf_thres,
        ratios[0],
        ratios[1],
        padding[0],
        padding[1],
    )

    gpu_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        kernel.call(
            num_blocks=num_blocks,
            num_threads=num_threads,
            stream=stream,
            args=args,
        )

        memcpy_device_to_host_async(
            input_binding.host_allocation,
            input_binding.allocation,
            stream=stream,
        )
        memcpy_device_to_host_async(
            mask_binding.host_allocation,
            mask_binding.allocation,
            stream=stream,
        )
        stream_synchronize(stream)
        t1 = time.perf_counter()
        gpu_times.append(t1 - t0)

    # handle host side postprocessing
    host_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        postprocess_detections(
            mock_outputs,
            ratios=ratios,
            padding=padding,
            conf_thres=conf_thres,
        )
        t1 = time.perf_counter()
        host_times.append(t1 - t0)

    mean_gpu_time = np.mean(gpu_times) * 1000.0
    mean_host_time = np.mean(host_times) * 1000.0

    assert mean_gpu_time <= mean_host_time

    input_binding.free()
    mask_binding.free()
    kernel.free()
    destroy_stream(stream)


def test_rescale_v10_results() -> None:
    """Test rescale detections kernel results for YOLOv10."""
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

    mock_outputs = _v10_outputs(topk)

    box_mask: np.ndarray = np.zeros(
        (1, topk),
        dtype=bool,
    )
    input_binding = create_binding(
        mock_outputs[0],
        pagelocked_mem=True,
    )

    memcpy_host_to_device_async(
        input_binding.allocation,
        mock_outputs[0],
        stream=stream,
    )

    mask_binding = create_binding(
        box_mask,
        pagelocked_mem=True,
    )

    # load the kernel
    kernel = Kernel(
        *kernels.RESCALE_V10_DETECTIONS,
    )

    args = kernel.create_args(
        input_binding.allocation,
        mask_binding.allocation,
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
        input_binding.host_allocation,
        input_binding.allocation,
        stream=stream,
    )
    memcpy_device_to_host_async(
        mask_binding.host_allocation,
        mask_binding.allocation,
        stream=stream,
    )
    stream_synchronize(stream)

    output = input_binding.host_allocation.copy()
    mask = mask_binding.host_allocation.copy()

    assert output.shape == input_shape
    assert mask.shape == (1, topk)

    # perform the mask operation
    output = output[mask]

    # CPU side creates scores and classes separately
    bboxes = output[:, :4]
    scores = output[:, 4]
    classes = output[:, 5].astype(int)

    # process detections on the host side
    host_outputs = postprocess_detections(
        mock_outputs,
        ratios=ratios,
        padding=padding,
        conf_thres=conf_thres,
    )

    # separate from the list from CPU
    host_bboxes = host_outputs[0]
    host_scores = host_outputs[1]
    host_classes = host_outputs[2]

    assert bboxes.shape == host_bboxes.shape
    assert np.allclose(bboxes, host_bboxes)

    assert scores.shape == host_scores.shape
    assert np.allclose(scores, host_scores)

    assert classes.shape == host_classes.shape
    assert np.allclose(classes, host_classes)

    input_binding.free()
    mask_binding.free()
    kernel.free()
    destroy_stream(stream)


def test_rescale_efficient_nms_results() -> None:
    """Test rescale detections kernel results for EfficientNMS."""
    topk = 100
    input_shape = (1, topk, 4)
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

    mock_outputs = _efficient_nms_outputs(topk)

    box_mask: np.ndarray = np.zeros(
        (1, topk),
        dtype=bool,
    )
    input_binding = create_binding(
        mock_outputs[1],
        pagelocked_mem=True,
    )
    score_binding = create_binding(
        mock_outputs[2],
        pagelocked_mem=True,
    )
    mask_binding = create_binding(
        box_mask,
        pagelocked_mem=True,
    )

    memcpy_host_to_device_async(
        input_binding.allocation,
        mock_outputs[1],
        stream=stream,
    )
    memcpy_host_to_device_async(
        score_binding.allocation,
        mock_outputs[2],
        stream=stream,
    )

    # load the kernel
    kernel = Kernel(
        *kernels.RESCALE_EFF_NMS_DETECTIONS,
    )

    args = kernel.create_args(
        input_binding.allocation,
        score_binding.allocation,
        mask_binding.allocation,
        topk,
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
        input_binding.host_allocation,
        input_binding.allocation,
        stream=stream,
    )
    memcpy_device_to_host_async(
        mask_binding.host_allocation,
        mask_binding.allocation,
        stream=stream,
    )
    stream_synchronize(stream)

    output = input_binding.host_allocation.copy()
    mask = mask_binding.host_allocation.copy()

    assert output.shape == input_shape
    assert mask.shape == (1, topk)

    # perform the mask operation
    # index with zero to preserve batch dimension
    bboxes = output[:, mask[0]]

    # process detections on the host side
    host_outputs = postprocess_detections(
        mock_outputs,
        ratios=ratios,
        padding=padding,
        conf_thres=conf_thres,
    )

    # separate from the list from CPU
    host_bboxes = host_outputs[1]

    assert bboxes.shape == host_bboxes.shape
    assert np.allclose(bboxes, host_bboxes)

    input_binding.free()
    mask_binding.free()
    kernel.free()
    destroy_stream(stream)


def test_rescale_efficient_nms_performance() -> None:
    """Test rescale detections kernel performance for EfficientNMS."""
    topk = 100
    ratios = (1.0, 1.0)
    padding = (32.0, 32.0)
    conf_thres = 0.25
    iterations = 10000

    stream = create_stream()

    # block and thread info
    num_threads: tuple[int, int, int] = (256, 1, 1)
    num_blocks: tuple[int, int, int] = (
        math.ceil(topk / num_threads[0]),
        1,
        1,
    )

    mock_outputs = _efficient_nms_outputs(topk)

    box_mask: np.ndarray = np.zeros(
        (1, topk),
        dtype=bool,
    )
    input_binding = create_binding(
        mock_outputs[1],
        pagelocked_mem=True,
    )
    score_binding = create_binding(
        mock_outputs[2],
        pagelocked_mem=True,
    )
    mask_binding = create_binding(
        box_mask,
        pagelocked_mem=True,
    )

    memcpy_host_to_device_async(
        input_binding.allocation,
        mock_outputs[1],
        stream=stream,
    )
    memcpy_host_to_device_async(
        score_binding.allocation,
        mock_outputs[2],
        stream=stream,
    )

    # load the kernel
    kernel = Kernel(
        *kernels.RESCALE_EFF_NMS_DETECTIONS,
    )

    args = kernel.create_args(
        input_binding.allocation,
        score_binding.allocation,
        mask_binding.allocation,
        topk,
        conf_thres,
        ratios[0],
        ratios[1],
        padding[0],
        padding[1],
    )

    gpu_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        kernel.call(
            num_blocks=num_blocks,
            num_threads=num_threads,
            stream=stream,
            args=args,
        )

        memcpy_device_to_host_async(
            input_binding.host_allocation,
            input_binding.allocation,
            stream=stream,
        )
        memcpy_device_to_host_async(
            mask_binding.host_allocation,
            mask_binding.allocation,
            stream=stream,
        )
        stream_synchronize(stream)
        t1 = time.perf_counter()
        gpu_times.append(t1 - t0)

    host_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        # process detections on the host side
        postprocess_detections(
            mock_outputs,
            ratios=ratios,
            padding=padding,
            conf_thres=conf_thres,
        )
        t1 = time.perf_counter()
        host_times.append(t1 - t0)

    mean_gpu_time = np.mean(gpu_times) * 1000.0
    mean_host_time = np.mean(host_times) * 1000.0

    assert mean_gpu_time <= mean_host_time

    input_binding.free()
    mask_binding.free()
    kernel.free()
    destroy_stream(stream)


if __name__ == "__main__":
    test_rescale_v10_compile()
    test_rescale_eff_nms_compile()
