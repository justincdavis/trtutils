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


@pytest.mark.parametrize(
    "ratio",
    [
        pytest.param(5.0, id="13px-wide"),
        pytest.param(1.7, id="38px-wide"),
        pytest.param(0.21, id="305px-wide"),
    ],
)
def test_cuda_seg_masks_on_unaligned_frames(ratio: float) -> None:
    """Frames whose rows and mask planes are not 16 byte aligned get the exact masks."""
    rng = np.random.default_rng(1)
    head, proto = _seg_head(rng, 1, num=500, nc=3, nm=8)
    kwargs = {"max_det": 300, "input_size": INPUT}
    ref = postprocess_yolo_seg([head, proto], [(ratio, ratio)], [(0.0, 0.0)], 0.1, 0.5, **kwargs)

    stream = create_stream()
    try:
        outputs = [Buffer.from_array(a, MemoryLocation.DEVICE) for a in (head, proto)]
        canvases = [MaskCanvas(mapped=True)]
        post = CudaSegPostprocessor()
        for _ in range(2):
            ours = post(
                outputs,
                [(ratio, ratio)],
                [(0.0, 0.0)],
                0.1,
                0.5,
                0.5,
                canvases=canvases,
                stream=stream,
                **kwargs,
            )
            assert ours[0][3].any()
            np.testing.assert_array_equal(ours[0][3], ref[0][3])
    finally:
        destroy_stream(stream)


@pytest.mark.parametrize(
    ("nms_iou_thres", "shift", "ulps"),
    [
        pytest.param(0.5, 10 / 3, 1, id="iou-at-threshold"),
        pytest.param(0.5, 10 / 3, 37, id="iou-near-threshold"),
        pytest.param(0.0, 10.0, 1, id="barely-touching"),
        pytest.param(0.0, 10.0, 293, id="nearly-touching"),
    ],
)
def test_cuda_seg_nms_at_the_iou_threshold(nms_iou_thres, shift, ulps) -> None:
    """Pairs whose IoU walks across the threshold in float32 steps get the CPU NMS decisions."""
    side, pitch = 10.0, 24.0
    grid = 26
    centers = np.stack(np.meshgrid(np.arange(grid), np.arange(grid)), -1).reshape(-1, 2)
    centers = centers * pitch + 12.0
    # the second box's center, walked ulps float32 steps at a time through the threshold shift
    steps = (np.arange(grid * grid) - grid * grid // 2) * ulps
    second = ((centers[:, 0] + shift).astype(np.float32).view(np.int32) + steps).view(np.float32)
    head = np.zeros((1, 4 + 1 + 4, 2 * grid * grid), dtype=np.float32)
    for k, ((cx, cy), sx) in enumerate(zip(centers, second)):
        head[0, :5, 2 * k] = [cx, cy, side, side, 0.9]
        head[0, :5, 2 * k + 1] = [sx, cy, side, side, 0.8]
    proto = np.zeros((1, 4, 16, 16), dtype=np.float32)
    kwargs = {"max_det": 2 * grid * grid, "input_size": (640, 640)}
    ref = postprocess_yolo_seg(
        [head, proto], [(1.0, 1.0)], [(0.0, 0.0)], 0.25, nms_iou_thres, 0.6, **kwargs
    )
    assert 0 < len(ref[0][1]) - grid * grid < grid * grid  # some second boxes survive, some do not

    stream = create_stream()
    try:
        outputs = [Buffer.from_array(a, MemoryLocation.DEVICE) for a in (head, proto)]
        ours = CudaSegPostprocessor()(
            outputs,
            [(1.0, 1.0)],
            [(0.0, 0.0)],
            0.25,
            nms_iou_thres,
            0.6,
            canvases=[MaskCanvas(mapped=True)],
            stream=stream,
            **kwargs,
        )
        for field in range(3):
            np.testing.assert_array_equal(ours[0][field], ref[0][field])
    finally:
        destroy_stream(stream)


def _rect_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU as cv2.dnn.NMSBoxes computes it: double rects from float32 corners."""
    ax, ay, aw, ah = float(a[0]), float(a[1]), float(a[2] - a[0]), float(a[3] - a[1])
    bx, by, bw, bh = float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter)


@pytest.mark.parametrize("pick", range(8))
def test_cuda_seg_nms_with_a_pair_iou_as_threshold(pick: int) -> None:
    """A threshold equal to a pair's IoU in float32 is decided like the CPU, beyond float32."""
    rng = np.random.default_rng(pick)
    num = 64
    head = np.zeros((1, 4 + 1 + 4, 2 * num), dtype=np.float32)
    centers = np.stack([np.arange(num) % 8 * 70 + 30, np.arange(num) // 8 * 70 + 30], -1)
    for k, (cx, cy) in enumerate(centers):
        w, h = rng.uniform(10, 40, 2)
        head[0, :5, 2 * k] = [cx, cy, w, h, 0.9]
        head[0, :5, 2 * k + 1] = [cx + rng.uniform(0, w / 2), cy + rng.uniform(0, h / 2), w, h, 0.8]
    xyxy = np.concatenate([head[0, :2] - head[0, 2:4] / 2, head[0, :2] + head[0, 2:4] / 2]).T
    thres = float(np.float32(_rect_iou(xyxy[2 * pick], xyxy[2 * pick + 1])))
    proto = np.zeros((1, 4, 16, 16), dtype=np.float32)
    kwargs = {"max_det": 2 * num, "input_size": (640, 640)}
    ref = postprocess_yolo_seg([head, proto], [(1.0, 1.0)], [(0.0, 0.0)], 0.25, thres, 0.6, **kwargs)

    stream = create_stream()
    try:
        outputs = [Buffer.from_array(a, MemoryLocation.DEVICE) for a in (head, proto)]
        ours = CudaSegPostprocessor()(
            outputs,
            [(1.0, 1.0)],
            [(0.0, 0.0)],
            0.25,
            thres,
            0.6,
            canvases=[MaskCanvas(mapped=True)],
            stream=stream,
            **kwargs,
        )
        for field in range(3):
            np.testing.assert_array_equal(ours[0][field], ref[0][field])
    finally:
        destroy_stream(stream)


def test_cuda_seg_reused_canvases_grow_and_shrink() -> None:
    """Frames with more, fewer, then more masks than the canvas holds, and a new frame size."""
    rng = np.random.default_rng(1)
    head, proto = _seg_head(rng, 2, num=500, nc=3, nm=8)
    outputs = [Buffer.from_array(a, MemoryLocation.DEVICE) for a in (head, proto)]
    padding = [(0.0, 6.0), (10.0, 0.0)]
    kwargs = {"max_det": 300, "input_size": INPUT}
    stream = create_stream()
    try:
        canvases = [MaskCanvas(mapped=True) for _ in range(2)]
        post = CudaSegPostprocessor()
        for conf_thres, ratio in [(0.9, 0.6), (0.25, 0.6), (0.6, 0.6), (0.1, 0.6), (0.25, 0.5)]:
            ratios = [(ratio, ratio), (0.35, 0.35)]
            args = ([head, proto], ratios, padding, conf_thres, 0.5)
            ref = postprocess_yolo_seg(*args, **kwargs)
            args = (outputs, ratios, padding, conf_thres, 0.5, 0.5)
            ours = post(*args, canvases=canvases, stream=stream, **kwargs)
            for got, want in zip(ours, ref):
                for field, expected in zip(got, want):
                    np.testing.assert_array_equal(field, expected)
    finally:
        destroy_stream(stream)


def _one_box_seg(x2: float) -> list:
    """One detection spanning x 4..x2 whose mask fills the whole box."""
    head = np.zeros((1, 4 + 1 + 4, 1), dtype=np.float32)
    head[0, :5, 0] = [(4 + x2) / 2, 32, x2 - 4, 40, 0.9]
    head[0, 5:, 0] = 1.0
    return [head, np.full((1, 4, 16, 16), 5.0, dtype=np.float32)]


def test_cuda_seg_clears_stale_columns_past_a_shrunk_box() -> None:
    """A reused canvas drops the stale pixels just right of a narrower box at every alignment."""
    stream = create_stream()
    try:
        canvases = [MaskCanvas(mapped=True)]
        post = CudaSegPostprocessor()
        kwargs = {"max_det": 300, "input_size": INPUT, "canvases": canvases, "stream": stream}
        for x2 in range(44, 60):
            head = [Buffer.from_array(a, MemoryLocation.DEVICE) for a in _one_box_seg(60.0)]
            post(head, [(1.0, 1.0)], [(0.0, 0.0)], 0.5, 0.5, 0.5, **kwargs)
            head = [Buffer.from_array(a, MemoryLocation.DEVICE) for a in _one_box_seg(x2)]
            ours = post(head, [(1.0, 1.0)], [(0.0, 0.0)], 0.5, 0.5, 0.5, **kwargs)
            assert ours[0][3][0, 12:52, 4:x2].all()
            assert not ours[0][3][0, :, x2:].any()
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
