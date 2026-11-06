# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/postprocessors/_cuda_segmentation.py -- CUDA segmentation matches the CPU."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.postprocessors._cuda_segmentation import CUDASegPostprocessor
from trtutils.image.postprocessors._cpu_segmentation import MaskCanvas, postprocess_yolo_seg

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


def _assert_seg_equal(ours: list, ref: list, fields: int = 4) -> None:
    assert len(ours) == len(ref)
    for got, want in zip(ours, ref):
        assert len(got[1]) == len(want[1])
        for field in range(fields):
            np.testing.assert_array_equal(got[field], want[field])


@pytest.mark.parametrize(
    ("batch", "conf_thres", "nms_iou_thres", "max_det", "num"),
    [
        pytest.param(1, 0.25, 0.5, 300, 500, id="default"),
        pytest.param(1, None, 0.5, 300, 500, id="no-conf"),
        pytest.param(1, 0.99, 0.5, 300, 500, id="nothing-passes"),
        pytest.param(1, 0.1, 0.0, 300, 500, id="iou-zero"),
        pytest.param(1, 0.1, 0.9, 7, 500, id="max-det"),
        pytest.param(2, 0.25, 0.5, 300, 500, id="batch2"),
        pytest.param(1, None, 0.5, 300, 5000, id="over-nms-rows"),
        pytest.param(2, None, 0.5, 300, 5000, id="over-nms-rows-batch2"),
    ],
)
def test_cuda_seg_matches_cpu(
    cuda_stream, to_device, batch, conf_thres, nms_iou_thres, max_det, num
) -> None:
    """Boxes, scores, classes and every mask pixel match, also past 4096 NMS candidates."""
    rng = np.random.default_rng(0)
    head, proto = _seg_head(rng, batch, num=num, nc=3, nm=8)
    ratios = [(0.6, 0.6), (0.35, 0.35)][:batch]
    padding = [(0.0, 6.0), (10.0, 0.0)][:batch]
    kwargs = {"max_det": max_det, "input_size": INPUT}
    ref = postprocess_yolo_seg([head, proto], ratios, padding, conf_thres, nms_iou_thres, **kwargs)

    outputs = [to_device(head), to_device(proto)]
    canvases = [MaskCanvas(mapped=True) for _ in range(batch)]
    post = CUDASegPostprocessor(cuda_stream)
    # twice, so the second run reuses the canvases and clears the first run's boxes
    for _ in range(2):
        ours = post(
            outputs, ratios, padding, conf_thres, nms_iou_thres, 0.5, canvases=canvases, **kwargs
        )
        _assert_seg_equal(ours, ref)


@pytest.mark.parametrize(
    "ratio",
    [
        pytest.param(5.0, id="13px-wide"),
        pytest.param(1.7, id="38px-wide"),
        pytest.param(0.21, id="305px-wide"),
    ],
)
def test_cuda_seg_masks_on_unaligned_frames(cuda_stream, to_device, ratio: float) -> None:
    """Frames whose rows and mask planes are not 16 byte aligned get the exact masks."""
    rng = np.random.default_rng(1)
    head, proto = _seg_head(rng, 1, num=500, nc=3, nm=8)
    kwargs = {"max_det": 300, "input_size": INPUT}
    ref = postprocess_yolo_seg([head, proto], [(ratio, ratio)], [(0.0, 0.0)], 0.1, 0.5, **kwargs)

    outputs = [to_device(head), to_device(proto)]
    canvases = [MaskCanvas(mapped=True)]
    post = CUDASegPostprocessor(cuda_stream)
    for _ in range(2):
        ours = post(
            outputs, [(ratio, ratio)], [(0.0, 0.0)], 0.1, 0.5, 0.5, canvases=canvases, **kwargs
        )
        assert ours[0][3].any()
        np.testing.assert_array_equal(ours[0][3], ref[0][3])


@pytest.mark.parametrize(
    ("nms_iou_thres", "shift", "ulps"),
    [
        pytest.param(0.5, 10 / 3, 1, id="iou-at-threshold"),
        pytest.param(0.5, 10 / 3, 37, id="iou-near-threshold"),
        pytest.param(0.0, 10.0, 1, id="barely-touching"),
        pytest.param(0.0, 10.0, 293, id="nearly-touching"),
    ],
)
def test_cuda_seg_nms_at_the_iou_threshold(
    cuda_stream, to_device, nms_iou_thres, shift, ulps
) -> None:
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

    ours = CUDASegPostprocessor(cuda_stream)(
        [to_device(head), to_device(proto)],
        [(1.0, 1.0)],
        [(0.0, 0.0)],
        0.25,
        nms_iou_thres,
        0.6,
        canvases=[MaskCanvas(mapped=True)],
        **kwargs,
    )
    _assert_seg_equal(ours, ref, fields=3)


def _rect_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU as cv2.dnn.NMSBoxes computes it: double rects from float32 corners."""
    ax, ay, aw, ah = float(a[0]), float(a[1]), float(a[2] - a[0]), float(a[3] - a[1])
    bx, by, bw, bh = float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter)


@pytest.mark.parametrize("pick", range(8))
def test_cuda_seg_nms_with_a_pair_iou_as_threshold(cuda_stream, to_device, pick: int) -> None:
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

    ours = CUDASegPostprocessor(cuda_stream)(
        [to_device(head), to_device(proto)],
        [(1.0, 1.0)],
        [(0.0, 0.0)],
        0.25,
        thres,
        0.6,
        canvases=[MaskCanvas(mapped=True)],
        **kwargs,
    )
    _assert_seg_equal(ours, ref, fields=3)


def test_seg_canvas_clears_when_nothing_is_kept() -> None:
    """A reused canvas drops the previous masks when the next frame keeps nothing."""
    rng = np.random.default_rng(0)
    head, proto = _seg_head(rng, 1, num=500, nc=3, nm=8)
    ratios, padding = [(1.0, 1.0)], [(0.0, 0.0)]
    base = {"max_det": 300, "input_size": INPUT}
    kwargs = {**base, "canvases": [MaskCanvas()]}
    first = postprocess_yolo_seg([head, proto], ratios, padding, 0.25, 0.5, **kwargs)
    assert first[0][3].any()
    empty = postprocess_yolo_seg([head, proto], ratios, padding, 1.1, 0.5, **kwargs)
    assert empty[0][3].shape[0] == 0
    assert not first[0][3].any()
    again = postprocess_yolo_seg([head, proto], ratios, padding, 0.25, 0.5, **kwargs)
    fresh = postprocess_yolo_seg([head, proto], ratios, padding, 0.25, 0.5, **base)
    np.testing.assert_array_equal(again[0][3], fresh[0][3])


def test_cuda_seg_reused_canvases_grow_and_shrink(cuda_stream, to_device) -> None:
    """Frames with more, fewer, then more masks than the canvas holds, and a new frame size."""
    rng = np.random.default_rng(1)
    head, proto = _seg_head(rng, 2, num=500, nc=3, nm=8)
    outputs = [to_device(head), to_device(proto)]
    padding = [(0.0, 6.0), (10.0, 0.0)]
    kwargs = {"max_det": 300, "input_size": INPUT}
    canvases = [MaskCanvas(mapped=True) for _ in range(2)]
    post = CUDASegPostprocessor(cuda_stream)
    for conf_thres, ratio in [(0.9, 0.6), (0.25, 0.6), (0.6, 0.6), (0.1, 0.6), (0.25, 0.5)]:
        ratios = [(ratio, ratio), (0.35, 0.35)]
        ref = postprocess_yolo_seg([head, proto], ratios, padding, conf_thres, 0.5, **kwargs)
        ours = post(outputs, ratios, padding, conf_thres, 0.5, 0.5, canvases=canvases, **kwargs)
        _assert_seg_equal(ours, ref)


def _one_box_seg(x2: float) -> list:
    """One detection spanning x 4..x2 whose mask fills the whole box."""
    head = np.zeros((1, 4 + 1 + 4, 1), dtype=np.float32)
    head[0, :5, 0] = [(4 + x2) / 2, 32, x2 - 4, 40, 0.9]
    head[0, 5:, 0] = 1.0
    return [head, np.full((1, 4, 16, 16), 5.0, dtype=np.float32)]


def test_cuda_seg_clears_stale_columns_past_a_shrunk_box(cuda_stream, to_device) -> None:
    """A reused canvas drops the stale pixels just right of a narrower box at every alignment."""
    post = CUDASegPostprocessor(cuda_stream)
    kwargs = {"max_det": 300, "input_size": INPUT, "canvases": [MaskCanvas(mapped=True)]}
    for x2 in range(44, 60):
        wide = [to_device(a) for a in _one_box_seg(60.0)]
        post(wide, [(1.0, 1.0)], [(0.0, 0.0)], 0.5, 0.5, 0.5, **kwargs)
        narrow = [to_device(a) for a in _one_box_seg(x2)]
        ours = post(narrow, [(1.0, 1.0)], [(0.0, 0.0)], 0.5, 0.5, 0.5, **kwargs)
        assert ours[0][3][0, 12:52, 4:x2].all()
        assert not ours[0][3][0, :, x2:].any()
