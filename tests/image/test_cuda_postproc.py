# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/postprocessors/_cuda.py -- GPU-vs-CPU postprocessing parity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from trtutils.core import (
    create_binding,
    create_stream,
    destroy_stream,
    memcpy_host_to_device_async,
    stream_synchronize,
)
from trtutils.image.postprocessors import (
    postprocess_classifications,
    postprocess_depth,
    postprocess_detr,
    postprocess_detr_lbs,
    postprocess_efficient_nms,
    postprocess_hand_interactions,
    postprocess_rfdetr,
    postprocess_rtdetrv3,
    postprocess_yolov10,
)
from trtutils.image.postprocessors._cuda import CUDAPostprocessor

if TYPE_CHECKING:
    from trtutils.compat._libs import cudart

pytestmark = pytest.mark.gpu

RNG = np.random.default_rng(0)
B = 4
RATIOS = [(0.5, 0.5), (1.0, 0.8), (0.3, 0.9), (0.7, 0.7)]
PADS = [(10.0, 5.0), (0.0, 20.0), (3.0, 1.0), (100.0, 50.0)]

# keep device bindings alive for the test session (pinned host memory)
_BINDINGS: list = []


@pytest.fixture(scope="module")
def stream() -> cudart.cudaStream_t:
    """Module-scoped CUDA stream for the parity tests."""
    s = create_stream()
    yield s
    destroy_stream(s)


def to_dev(stream: cudart.cudaStream_t, arr: np.ndarray) -> int:
    """Copy ``arr`` to a device binding and return the device pointer."""
    arr = np.ascontiguousarray(arr)
    binding = create_binding(arr)
    _BINDINGS.append(binding)
    memcpy_host_to_device_async(binding.allocation, arr, stream)
    return binding.allocation


def _dets_matrix(res: list[np.ndarray]) -> np.ndarray:
    """Boxes + scores + class ids as one (N, 6) float array."""
    if len(res[0]) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return np.hstack([res[0], res[1][:, None], res[2][:, None].astype(res[0].dtype)])


def _sorted_dets(t: np.ndarray) -> np.ndarray:
    """Sort (N, 6) detection rows (y1, x1, score, class) for order-insensitive comparison."""
    if len(t) == 0:
        return t
    order = np.lexsort((t[:, 1], t[:, 0], t[:, 4], t[:, 5]))
    return t[order]


def check_dets(name: str, cpu_res: list[list[np.ndarray]], gpu_res: list[list[np.ndarray]]) -> None:
    """Assert GPU and CPU detection results match (order-insensitive)."""
    assert len(cpu_res) == len(gpu_res)
    for b, (c, g) in enumerate(zip(cpu_res, gpu_res)):
        assert len(c[0]) == len(g[0]), f"{name} b{b}: count {len(c[0])} vs {len(g[0])}"
        cs = _sorted_dets(_dets_matrix(c))
        gs = _sorted_dets(_dets_matrix(g))
        if len(c[0]) > 0:
            np.testing.assert_allclose(cs[:, :4], gs[:, :4], rtol=2e-5, atol=1e-4)
            np.testing.assert_allclose(cs[:, 4], gs[:, 4], rtol=2e-4, atol=1e-5)
            assert np.array_equal(cs[:, 5].astype(int), gs[:, 5].astype(int))


def test_efficient_nms_parity(stream: cudart.cudaStream_t) -> None:
    """efficient_nms matches the CPU postprocessor across num_dets/class dtypes."""
    k = 50
    num_dets = np.array([k, 3, 12, 0], dtype=np.int32)
    boxes = RNG.uniform(-20, 1000, (B, k, 4)).astype(np.float32)
    scores = RNG.uniform(0, 1, (B, k)).astype(np.float32)
    classes = RNG.integers(0, 80, (B, k)).astype(np.int32)

    for cls_dtype in (np.int32, np.float32):
        for nd_dtype in (np.int32, np.float32):
            nd = num_dets.astype(nd_dtype)
            cl = classes.astype(cls_dtype)
            spec = [
                (list(nd.shape), nd.dtype),
                (list(boxes.shape), boxes.dtype),
                (list(scores.shape), scores.dtype),
                (list(cl.shape), cl.dtype),
            ]
            cp = CUDAPostprocessor(spec, "efficient_nms", k, stream=stream)
            ptrs = [to_dev(stream, a) for a in (nd, boxes, scores, cl)]
            stream_synchronize(stream)
            gpu = cp.postprocess_efficient_nms(B, ptrs, RATIOS, PADS, conf_thres=0.3)
            cpu = postprocess_efficient_nms(
                [nd.copy(), boxes.copy(), scores.copy(), cl.copy()], RATIOS, PADS, conf_thres=0.3
            )
            check_dets(f"efficient_nms(nd={nd_dtype.__name__}, cls={cls_dtype.__name__})", cpu, gpu)


def test_efficient_nms_batch_x_k_gt_256(stream: cudart.cudaStream_t) -> None:
    """Compact kernel grid must cover batch*k > 256 candidates (regression: grid was ceil(k/256))."""
    batch, k = 6, 50  # 6 * 50 = 300 > 256
    num_dets = np.full(batch, k, dtype=np.int32)
    boxes = RNG.uniform(0, 1000, (batch, k, 4)).astype(np.float32)
    scores = RNG.uniform(0.4, 1, (batch, k)).astype(np.float32)
    classes = RNG.integers(0, 80, (batch, k)).astype(np.int32)
    ratios = [(0.5, 0.5)] * batch
    pads = [(10.0, 5.0)] * batch

    spec = [
        (list(num_dets.shape), num_dets.dtype),
        (list(boxes.shape), boxes.dtype),
        (list(scores.shape), scores.dtype),
        (list(classes.shape), classes.dtype),
    ]
    cp = CUDAPostprocessor(spec, "efficient_nms", k, stream=stream)
    ptrs = [to_dev(stream, a) for a in (num_dets, boxes, scores, classes)]
    stream_synchronize(stream)
    gpu = cp.postprocess_efficient_nms(batch, ptrs, ratios, pads, conf_thres=0.3)
    cpu = postprocess_efficient_nms(
        [num_dets.copy(), boxes.copy(), scores.copy(), classes.copy()], ratios, pads, conf_thres=0.3
    )
    check_dets("efficient_nms(batch_x_k>256)", cpu, gpu)
    # all scores > 0.4: every candidate must survive
    assert all(len(g[0]) == k for g in gpu)


def test_yolov10_parity(stream: cudart.cudaStream_t) -> None:
    """yolov10 GPU results match the CPU postprocessor."""
    n = 300
    v10 = np.zeros((B, n, 6), dtype=np.float32)
    v10[:, :, :4] = RNG.uniform(-20, 1000, (B, n, 4))
    v10[:, :, 4] = RNG.uniform(0, 1, (B, n))
    v10[:, :, 5] = RNG.integers(0, 80, (B, n))

    cp = CUDAPostprocessor([(list(v10.shape), v10.dtype)], "yolov10", n, stream=stream)
    stream_synchronize(stream)
    gpu = cp.postprocess_yolov10(B, [to_dev(stream, v10)], RATIOS, PADS, conf_thres=0.5)
    cpu = postprocess_yolov10([v10.copy()], RATIOS, PADS, conf_thres=0.5)
    check_dets("yolov10", cpu, gpu)


def _detr_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = 300
    d_scores = RNG.uniform(0, 1, (B, q)).astype(np.float32)
    d_labels = RNG.integers(0, 91, (B, q)).astype(np.int32)
    d_boxes = RNG.uniform(0, 800, (B, q, 4)).astype(np.float32)
    # inject some non-finite boxes
    d_boxes[0, 5] = [np.nan, 1, 2, 3]
    d_boxes[1, 7, 2] = np.inf
    return d_scores, d_labels, d_boxes


def test_detr_parity(stream: cudart.cudaStream_t) -> None:
    """Detr GPU results match the CPU postprocessor."""
    d_scores, d_labels, d_boxes = _detr_data()
    spec = [
        (list(d_scores.shape), d_scores.dtype),
        (list(d_labels.shape), d_labels.dtype),
        (list(d_boxes.shape), d_boxes.dtype),
    ]
    cp = CUDAPostprocessor(spec, "detr", 300, stream=stream)
    ptrs = [to_dev(stream, a) for a in (d_scores, d_labels, d_boxes)]
    stream_synchronize(stream)
    gpu = cp.postprocess_detr(B, ptrs, RATIOS, PADS, conf_thres=0.4)
    cpu = postprocess_detr(
        [d_scores.copy(), d_labels.copy(), d_boxes.copy()], RATIOS, PADS, conf_thres=0.4
    )
    check_dets("detr", cpu, gpu)


def test_detr_lbs_parity(stream: cudart.cudaStream_t) -> None:
    """LBS output order (labels, boxes, scores) reordered like Detector does."""
    d_scores, d_labels, d_boxes = _detr_data()
    spec = [
        (list(d_labels.shape), d_labels.dtype),
        (list(d_boxes.shape), d_boxes.dtype),
        (list(d_scores.shape), d_scores.dtype),
    ]
    cp = CUDAPostprocessor([spec[2], spec[0], spec[1]], "detr", 300, stream=stream)
    ptrs = [to_dev(stream, a) for a in (d_scores, d_labels, d_boxes)]
    stream_synchronize(stream)
    gpu = cp.postprocess_detr(B, ptrs, RATIOS, PADS, conf_thres=0.4)
    cpu = postprocess_detr_lbs(
        [d_labels.copy(), d_boxes.copy(), d_scores.copy()], RATIOS, PADS, conf_thres=0.4
    )
    check_dets("detr_lbs", cpu, gpu)


def test_rtdetrv3_parity(stream: cudart.cudaStream_t) -> None:
    """rtdetrv3 GPU results match the CPU postprocessor."""
    counts_img = [5, 0, 9, 3]
    t = sum(counts_img)
    rd = np.zeros((t, 6), dtype=np.float32)
    rd[:, 0] = RNG.integers(0, 80, t)
    rd[:, 1] = RNG.uniform(0, 1, t)
    rd[:, 2:6] = RNG.uniform(-5, 1000, (t, 4))
    rd[2, 2] = np.nan  # one non-finite
    ndets = np.array(counts_img, dtype=np.int32)

    cp = CUDAPostprocessor(
        [(list(rd.shape), rd.dtype), (list(ndets.shape), ndets.dtype)],
        "rtdetrv3",
        max(counts_img),
        total_dets=t,
        stream=stream,
    )
    ptrs = [to_dev(stream, a) for a in (rd, ndets)]
    stream_synchronize(stream)
    gpu = cp.postprocess_rtdetrv3(B, ptrs, RATIOS, PADS, conf_thres=0.3)
    cpu = postprocess_rtdetrv3([rd.copy(), ndets.copy()], RATIOS, PADS, conf_thres=0.3)
    check_dets("rtdetrv3", cpu, gpu)


def test_rfdetr_parity(stream: cudart.cudaStream_t) -> None:
    """Rfdetr GPU results match the CPU postprocessor."""
    q, c = 250, 81
    dets = RNG.uniform(0, 1, (B, q, 4)).astype(np.float32)
    logits = RNG.uniform(-5, 5, (B, q, c)).astype(np.float32)

    cp = CUDAPostprocessor(
        [(list(dets.shape), dets.dtype), (list(logits.shape), logits.dtype)],
        "rfdetr",
        q,
        stream=stream,
    )
    ptrs = [to_dev(stream, a) for a in (dets, logits)]
    stream_synchronize(stream)
    gpu = cp.postprocess_rfdetr(B, ptrs, RATIOS, PADS, conf_thres=0.5, input_size=(640, 640))
    cpu = postprocess_rfdetr(
        [dets.copy(), logits.copy()], RATIOS, PADS, conf_thres=0.5, input_size=(640, 640)
    )
    check_dets("rfdetr", cpu, gpu)


def test_classification_parity(stream: cudart.cudaStream_t) -> None:
    """Classification GPU results match the CPU postprocessor."""
    c = 1000
    logits = RNG.uniform(-10, 10, (B, c)).astype(np.float32)

    cp = CUDAPostprocessor([(list(logits.shape), logits.dtype)], "classification", 0, stream=stream)
    stream_synchronize(stream)
    gpu = cp.postprocess_classifications(B, [to_dev(stream, logits)])
    cpu = postprocess_classifications([logits.copy()])
    for b in range(B):
        np.testing.assert_allclose(gpu[b][0], cpu[b][0], rtol=2e-4, atol=1e-6)


def test_depth_parity(stream: cudart.cudaStream_t) -> None:
    """Depth GPU results match the CPU postprocessor."""
    hw = (64, 64)
    depth = RNG.uniform(0, 10, (B, 1, hw[0], hw[1])).astype(np.float32)

    cp = CUDAPostprocessor([(list(depth.shape), depth.dtype)], "depth", 0, stream=stream)
    stream_synchronize(stream)
    gpu = cp.postprocess_depth(B, [to_dev(stream, depth)])
    cpu = postprocess_depth([depth.copy()])
    for b in range(B):
        np.testing.assert_allclose(gpu[b][0], cpu[b][0], rtol=1e-4, atol=1e-6)


def _hand_data() -> list[np.ndarray]:
    k = 30
    h_boxes = RNG.uniform(0, 640, (B, k, 4)).astype(np.float32)
    h_boxes[:, :, 2] = h_boxes[:, :, 0] + RNG.uniform(5, 100, (B, k)).astype(np.float32)
    h_boxes[:, :, 3] = h_boxes[:, :, 1] + RNG.uniform(5, 100, (B, k)).astype(np.float32)
    h_scores = RNG.uniform(0, 1, (B, k)).astype(np.float32)
    h_labels = RNG.integers(0, 3, (B, k)).astype(np.int32)
    h_pairs = RNG.uniform(0, 1, (B, k, k, 2)).astype(np.float32)
    h_side = RNG.integers(0, 2, (B, k)).astype(np.int32)
    return [h_boxes, h_scores, h_labels, h_pairs, h_side]


def test_hand_interactions_parity(stream: cudart.cudaStream_t) -> None:
    """Hand NMS survivor set, boxes, and gathered pairs/side match the CPU (cv2) path."""
    data = _hand_data()
    spec = [(list(a.shape), a.dtype) for a in data]
    cp = CUDAPostprocessor(spec, "hand", 30, stream=stream)
    ptrs = [to_dev(stream, a) for a in data]
    stream_synchronize(stream)
    gpu = cp.postprocess_hand_interactions(B, ptrs, RATIOS, PADS, 0.3, 0.5)
    cpu = postprocess_hand_interactions([a.copy() for a in data], RATIOS, PADS, 0.3, 0.5)

    for b, (c, g) in enumerate(zip(cpu, gpu)):
        # NMS order may differ; match rows by (score, label) key
        ckey = {(round(float(sc), 5), int(lab)) for sc, lab in zip(c[1], c[2])}
        gkey = {(round(float(sc), 5), int(lab)) for sc, lab in zip(g[1], g[2])}
        assert ckey == gkey, f"b{b}: survivors differ\n{sorted(ckey - gkey)}\n{sorted(gkey - ckey)}"
        cmap = {(round(float(sc), 5), int(lab)): i for i, (sc, lab) in enumerate(zip(c[1], c[2]))}
        for i, (sc, lab) in enumerate(zip(g[1], g[2])):
            j = cmap[(round(float(sc), 5), int(lab))]
            np.testing.assert_allclose(g[0][i], c[0][j], rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(g[3], c[3], rtol=1e-5, atol=1e-6)
        if c[4].shape[1] > 0:
            assert np.array_equal(g[4], c[4])


def test_invalid_family_raises() -> None:
    """Unknown families are rejected with a ValueError."""
    spec = [([4, 4], np.dtype(np.float32))]
    with pytest.raises(ValueError, match="Invalid postprocessor family"):
        CUDAPostprocessor(spec, "bogus", 4, stream=None)  # type: ignore[arg-type]


def test_dtype_validation_raises() -> None:
    """Families reject non-matching output dtypes at construction time."""
    # yolov10 expects float32; int32 must be rejected
    spec = [([4, 300, 6], np.dtype(np.int32))]
    with pytest.raises(ValueError, match="expects"):
        CUDAPostprocessor(spec, "yolov10", 300, stream=None)  # type: ignore[arg-type]
