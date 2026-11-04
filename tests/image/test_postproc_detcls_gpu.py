# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CUDA detection, classification, and NMS postprocessors match the CPU path."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.core import create_stream, destroy_stream
from trtutils.core._buffer import Buffer, MemoryLocation
from trtutils.core._stream import stream_synchronize
from trtutils.image.postprocessors._classifier import postprocess_classifications
from trtutils.image.postprocessors._detection import postprocess_efficient_nms, postprocess_yolov10
from trtutils.image.postprocessors._gpu_detcls import CudaClsPostprocessor, CudaDetPostprocessor
from trtutils.image.postprocessors._gpu_nms import YoloNms

_RATIOS = [(0.5, 0.5), (0.25, 0.5)]
_PADDING = [(4.0, 8.0), (2.0, 6.0)]
_INPUT = (64, 48)


@pytest.fixture
def stream():
    """A CUDA stream for the postprocessor kernels."""
    handle = create_stream()
    yield handle
    destroy_stream(handle)


def _device(array: np.ndarray) -> Buffer:
    return Buffer.from_array(np.ascontiguousarray(array), MemoryLocation.DEVICE)


def test_cuda_yolov10_matches_cpu(stream) -> None:
    """Unletterbox and the confidence filter keep the same rows as the CPU path."""
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((2, 5, 6)).astype(np.float32)
    rows[:, :, 4] = np.array([0.9, 0.05, 0.4, 0.0, 0.2], np.float32)
    rows[:, :, 5] = np.array([1, 2, 3, 0, 4], np.float32)
    cpu = postprocess_yolov10([rows.copy()], _RATIOS, _PADDING, 0.3, _INPUT)
    gpu = CudaDetPostprocessor().yolov10(_device(rows), _RATIOS, _PADDING, 0.3, _INPUT, stream)
    assert len(gpu) == len(cpu)
    for got, exp in zip(gpu, cpu):
        assert got[2].tolist() == exp[2].tolist()
        np.testing.assert_allclose(got[0], exp[0], atol=1e-4)
        np.testing.assert_allclose(got[1], exp[1], atol=1e-5)


def test_cuda_efficient_nms_matches_cpu(stream) -> None:
    """EfficientNMS keeps the first num_dets rows and the same unletterboxed boxes."""
    num = np.array([[2], [1]], np.int32)
    boxes = np.zeros((2, 4, 4), np.float32)
    boxes[0, 0] = (10, 12, 20, 22)
    boxes[0, 1] = (14, 16, 30, 40)
    boxes[1, 0] = (8, 10, 18, 20)
    scores = np.array([[0.8, 0.2, 0.9, 0.1], [0.05, 0.7, 0.4, 0.3]], np.float32)
    classes = np.array([[3, 1, 0, 2], [4, 5, 6, 7]], np.float32)
    outputs = [num, boxes, scores, classes]
    cpu = postprocess_efficient_nms(
        [part.copy() for part in outputs], _RATIOS, _PADDING, None, _INPUT
    )
    gpu = CudaDetPostprocessor().efficient_nms(
        [_device(part) for part in outputs], _RATIOS, _PADDING, None, _INPUT, stream
    )
    for got, exp in zip(gpu, cpu):
        assert got[2].tolist() == exp[2].tolist()
        np.testing.assert_allclose(got[0], exp[0], atol=1e-4)
        np.testing.assert_allclose(got[1], exp[1], atol=1e-5)


def test_cuda_nms_batched_over_row_cap(stream) -> None:
    """Batched NMS suppresses overlaps when the head is longer than the mask rows."""
    num = 4104
    head = np.zeros((2, 5, num), np.float32)
    head[0, 0] = np.arange(num, dtype=np.float32) * 100.0
    head[0, 2] = 10.0
    head[0, 3] = 10.0
    head[0, 4] = 0.5
    head[1, 0, :2] = 5.0
    head[1, 1, :2] = 5.0
    head[1, 2, :2] = 10.0
    head[1, 3, :2] = 10.0
    head[1, 4, 0] = 0.9
    head[1, 4, 1] = 0.8
    nms = YoloNms()
    nms(_device(head), 4, 1, 0.3, 0.5, 10, stream)
    stream_synchronize(stream)
    assert nms.keep_count.numpy().tolist() == [-1, 1]
    assert int(nms.keep.numpy()[1, 0]) == 0


def test_cuda_softmax_matches_cpu(stream) -> None:
    """The class vector matches a host softmax, and softmax=False copies the logits."""
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((3, 17)).astype(np.float32)
    cuda = CudaClsPostprocessor()
    for softmax in (True, False):
        cpu = postprocess_classifications([logits.copy()], softmax=softmax)
        gpu = cuda(_device(logits), softmax=softmax, stream=stream)
        for got, exp in zip(gpu, cpu):
            np.testing.assert_allclose(got[0], exp[0], rtol=1e-5, atol=1e-5)
