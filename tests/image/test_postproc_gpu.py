# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/postprocessors/_gpu.py -- CUDA postprocessing parity with the CPU path."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils import Buffer, MemoryLocation
from trtutils.core._stream import create_stream, destroy_stream
from trtutils.image.postprocessors._gpu import (
    CudaObbPostprocessor,
    CudaPosePostprocessor,
    CudaSegPostprocessor,
)
from trtutils.image.postprocessors._obb import postprocess_yolo_obb
from trtutils.image.postprocessors._pose import postprocess_yolo_pose
from trtutils.image.postprocessors._segmentation import MaskCanvas, postprocess_yolo_seg

INPUT = (64, 64)


def _seg_head(rng: np.random.Generator, batch: int, num: int, nc: int, nm: int) -> list:
    """Random clustered boxes with quantized scores, so NMS overlaps and score ties happen."""
    centers = rng.uniform(8, 56, size=(batch, 2, 6))
    pick = rng.integers(0, 6, size=(batch, num))
    xy = np.take_along_axis(centers, pick[:, None], axis=2) + rng.normal(0, 2, (batch, 2, num))
    wh = rng.uniform(4, 30, size=(batch, 2, num))
    scores = np.round(rng.uniform(0, 1, size=(batch, nc, num)), 2)
    coefs = rng.normal(0, 1, size=(batch, nm, num))
    head = np.concatenate([xy, wh, scores, coefs], axis=1).astype(np.float32)
    proto = rng.normal(0, 2, size=(batch, nm, 16, 16)).astype(np.float32)
    return [head, proto]


@pytest.mark.parametrize(
    ("batch", "conf_thres", "nms_iou_thres", "max_det"),
    [
        pytest.param(1, 0.25, 0.5, 300, id="default"),
        pytest.param(1, None, 0.5, 300, id="no-conf"),
        pytest.param(1, 0.99, 0.5, 300, id="nothing-passes"),
        pytest.param(1, 0.1, 0.0, 300, id="iou-zero"),
        pytest.param(1, 0.1, 0.9, 7, id="max-det"),
        pytest.param(2, 0.25, 0.5, 300, id="batch2"),
    ],
)
def test_cuda_seg_matches_cpu(batch, conf_thres, nms_iou_thres, max_det) -> None:
    """Boxes, scores, classes and every mask pixel match postprocess_yolo_seg."""
    rng = np.random.default_rng(0)
    head, proto = _seg_head(rng, batch, num=500, nc=3, nm=8)
    ratios = [(0.6, 0.6), (0.35, 0.35)][:batch]
    padding = [(0.0, 6.0), (10.0, 0.0)][:batch]
    kwargs = {"max_det": max_det, "input_size": INPUT}
    ref = postprocess_yolo_seg([head, proto], ratios, padding, conf_thres, nms_iou_thres, **kwargs)

    stream = create_stream()
    try:
        outputs = [Buffer.from_array(a, MemoryLocation.DEVICE) for a in (head, proto)]
        canvases = [MaskCanvas(mapped=True) for _ in range(batch)]
        post = CudaSegPostprocessor()
        # twice, so the second run reuses the canvases and clears the first run's boxes
        for _ in range(2):
            ours = post(
                outputs,
                ratios,
                padding,
                conf_thres,
                nms_iou_thres,
                0.5,
                canvases=canvases,
                stream=stream,
                **kwargs,
            )
            for got, want in zip(ours, ref):
                assert len(got[1]) == len(want[1])
                for field, expected in zip(got, want):
                    np.testing.assert_array_equal(field, expected)
    finally:
        destroy_stream(stream)


def _pose_head(rng: np.random.Generator, batch: int, num: int, nk: int) -> np.ndarray:
    """Clustered boxes with quantized scores, keypoints spilling past the input for clipping."""
    seg = _seg_head(rng, batch, num, nc=1, nm=3 * nk)[0]
    seg[:, 5:] = rng.uniform(-10, 74, size=seg[:, 5:].shape)
    seg[:, 7::3] = rng.uniform(0, 1, size=seg[:, 7::3].shape)
    return seg


@pytest.mark.parametrize(
    ("batch", "conf_thres", "nms_iou_thres", "max_det"),
    [
        pytest.param(1, 0.25, 0.5, 300, id="default"),
        pytest.param(1, None, 0.5, 300, id="no-conf"),
        pytest.param(1, 0.99, 0.5, 300, id="nothing-passes"),
        pytest.param(1, 0.1, 0.0, 300, id="iou-zero"),
        pytest.param(1, 0.1, 0.9, 7, id="max-det"),
        pytest.param(2, 0.25, 0.5, 300, id="batch2"),
    ],
)
def test_cuda_pose_matches_cpu(batch, conf_thres, nms_iou_thres, max_det) -> None:
    """Boxes, scores, classes and every keypoint match postprocess_yolo_pose."""
    rng = np.random.default_rng(0)
    head = _pose_head(rng, batch, num=500, nk=17)
    ratios = [(0.6, 0.6), (0.35, 0.35)][:batch]
    padding = [(0.0, 6.0), (10.0, 0.0)][:batch]
    kwargs = {"max_det": max_det, "input_size": INPUT}
    ref = postprocess_yolo_pose([head], ratios, padding, conf_thres, nms_iou_thres, **kwargs)

    stream = create_stream()
    try:
        outputs = [Buffer.from_array(head, MemoryLocation.DEVICE)]
        post = CudaPosePostprocessor()
        # twice, so the second run reuses the result buffers
        for _ in range(2):
            ours = post(outputs, ratios, padding, conf_thres, nms_iou_thres, stream=stream, **kwargs)
            for got, want in zip(ours, ref):
                assert len(got[1]) == len(want[1])
                for field, expected in zip(got, want):
                    np.testing.assert_array_equal(field, expected)
    finally:
        destroy_stream(stream)


def _obb_head(rng: np.random.Generator, batch: int, num: int, nc: int) -> np.ndarray:
    """Clustered rotated boxes with quantized scores, so probiou overlaps and score ties happen."""
    centers = rng.uniform(100, 900, size=(batch, 2, 8))
    pick = rng.integers(0, 8, size=(batch, num))
    xy = np.take_along_axis(centers, pick[:, None], axis=2) + rng.normal(0, 25, (batch, 2, num))
    wh = rng.uniform(30, 300, size=(batch, 2, num))
    scores = np.round(rng.uniform(0, 1, size=(batch, nc, num)), 2)
    angle = rng.uniform(-np.pi / 4, 3 * np.pi / 4, size=(batch, 1, num))
    return np.concatenate([xy, wh, scores, angle], axis=1).astype(np.float32)


# scores, classes and rotated boxes are exact, and so are the kept sets because the synthetic
# pairs sit well away from the probiou threshold (logf / expf / cosf can differ from numpy's
# float32 in the last ulp, which only matters for a pair within ~1e-6 of nms_iou_thres).
# The enclosing boxes use cos / sin of the angle, where numpy's float32 version is not correctly
# rounded (16% of inputs differ from cosf), so they match to a few ulp (~1e-4 px at 1e3 px)
@pytest.mark.parametrize(
    ("batch", "conf_thres", "nms_iou_thres", "max_det"),
    [
        pytest.param(1, 0.25, 0.5, 300, id="default"),
        pytest.param(1, None, 0.5, 300, id="no-conf"),
        pytest.param(1, 1.01, 0.5, 300, id="nothing-passes"),
        pytest.param(1, 0.1, 0.0, 300, id="iou-zero"),
        pytest.param(1, 0.1, 0.9, 7, id="max-det"),
        pytest.param(1, 0.1, 0.3, 2000, id="many-candidates"),
        pytest.param(2, 0.25, 0.5, 300, id="batch2"),
    ],
)
def test_cuda_obb_matches_cpu(batch, conf_thres, nms_iou_thres, max_det) -> None:
    """Enclosing boxes, scores, classes and rotated boxes match postprocess_yolo_obb."""
    rng = np.random.default_rng(0)
    head = _obb_head(rng, batch, num=2000, nc=3)
    ratios = [(0.6, 0.6), (0.35, 0.35)][:batch]
    padding = [(0.0, 6.0), (10.0, 0.0)][:batch]
    ref = postprocess_yolo_obb([head], ratios, padding, conf_thres, nms_iou_thres, max_det=max_det)

    stream = create_stream()
    try:
        outputs = [Buffer.from_array(head, MemoryLocation.DEVICE)]
        post = CudaObbPostprocessor()
        # twice, so the second run reuses the buffers
        for _ in range(2):
            ours = post(
                outputs, ratios, padding, conf_thres, nms_iou_thres, max_det=max_det, stream=stream
            )
            for got, want in zip(ours, ref):
                assert len(got[1]) == len(want[1])
                for k, (field, expected) in enumerate(zip(got, want)):
                    assert field.dtype == expected.dtype
                    if k == 0:
                        np.testing.assert_allclose(field, expected, rtol=1e-5, atol=1e-3)
                    else:
                        np.testing.assert_array_equal(field, expected)
    finally:
        destroy_stream(stream)


def test_cuda_obb_rejects_nonuniform_scale() -> None:
    """Like the CPU path, a non-uniform resize ratio raises."""
    head = _obb_head(np.random.default_rng(0), 1, num=10, nc=3)
    stream = create_stream()
    try:
        outputs = [Buffer.from_array(head, MemoryLocation.DEVICE)]
        with pytest.raises(ValueError, match="uniform"):
            CudaObbPostprocessor()(
                outputs, [(0.5, 0.25)], [(0.0, 0.0)], 0.1, 0.5, max_det=10, stream=stream
            )
    finally:
        destroy_stream(stream)
