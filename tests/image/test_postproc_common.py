# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/postprocessors/_common.py -- box helpers shared by postprocessors."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.postprocessors._common import (
    cxcywh_to_xyxy,
    nms,
    nms_indices,
    unletterbox_boxes,
)

pytestmark = pytest.mark.cpu

_OVERLAP = np.array([[10, 10, 50, 50], [11, 11, 51, 51]], dtype=np.float32)
_SCORES = np.array([0.9, 0.6], dtype=np.float32)


@pytest.mark.parametrize(
    ("boxes", "scores", "labels", "conf_thres", "iou_thres", "kept"),
    [
        pytest.param(_OVERLAP, _SCORES, [1, 1], 0.1, 0.5, [0], id="same-class-deduped"),
        pytest.param(_OVERLAP, _SCORES, [0, 1], 0.1, 0.5, [0, 1], id="other-class-kept"),
        pytest.param(_OVERLAP, _SCORES, [0, 0], None, 0.5, [0], id="zero-labels-deduped"),
        pytest.param(_OVERLAP, _SCORES, [1, 1], 0.1, 0.99, [0, 1], id="loose-iou-kept"),
        pytest.param(np.zeros((0, 4)), np.zeros(0), [], 0.1, 0.5, [], id="empty-input"),
    ],
)
def test_nms_indices(boxes, scores, labels, conf_thres, iou_thres, kept: list[int]) -> None:
    """Class-aware NMS keeps the expected indices."""
    keep = nms_indices(boxes, scores, np.array(labels, dtype=int), conf_thres, iou_thres)
    assert sorted(keep.tolist()) == kept


@pytest.mark.parametrize(
    "max_det",
    [
        pytest.param(1, id="max-det-1"),
        pytest.param(3, id="max-det-3"),
        pytest.param(10, id="max-det-10"),
    ],
)
def test_nms_max_det_keeps_top_scores(max_det: int) -> None:
    """Disjoint boxes are capped at max_det, highest scores first."""
    n = 10
    boxes = np.array([[i * 100, 0, i * 100 + 10, 10] for i in range(n)], dtype=np.float32)
    scores = np.random.default_rng(0).permutation(n).astype(np.float32) / n + 0.05
    keep = nms_indices(boxes, scores, np.zeros(n, dtype=int), 0.0, 0.5, max_det)
    assert keep.tolist() == np.argsort(-scores)[:max_det].tolist()


def test_cxcywh_to_xyxy_exact() -> None:
    """Center boxes convert to corners, extra columns are ignored."""
    rows = np.array([[10.0, 20.0, 4.0, 6.0, 0.9], [5.0, 5.0, 2.0, 2.0, 0.1]])
    np.testing.assert_array_equal(
        cxcywh_to_xyxy(rows), [[8.0, 17.0, 12.0, 23.0], [4.0, 4.0, 6.0, 6.0]]
    )


def test_unletterbox_boxes_exact_and_clipped() -> None:
    """Padding is removed, ratios undone, and negatives clip to 0."""
    boxes = np.array([[14.0, 24.0, 34.0, 44.0], [0.0, 0.0, 14.0, 24.0]], dtype=np.float32)
    out = unletterbox_boxes(boxes, (0.5, 0.25), (4.0, 4.0))
    np.testing.assert_allclose(out, [[20.0, 80.0, 60.0, 160.0], [0.0, 0.0, 20.0, 80.0]])


def _reference_nms(dets, iou_threshold, agnostic):
    """Greedy NMS over (bbox, score, class) in pure python, integer IoU as cv2ext computed it."""
    dets = sorted(dets, key=lambda d: d[1], reverse=True)
    keep = [True] * len(dets)
    for i, (a, _, ca) in enumerate(dets):
        if not keep[i]:
            continue
        for j in range(i + 1, len(dets)):
            b, _, cb = dets[j]
            if not keep[j] or (not agnostic and ca != cb):
                continue
            inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1])
            )
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            if union and inter / union > iou_threshold:
                keep[j] = False
    return [d for d, k in zip(dets, keep) if k]


_DETS = [
    ((0, 0, 100, 100), 0.8, 0),
    ((5, 5, 105, 105), 0.9, 0),
    ((0, 0, 100, 100), 0.7, 1),
    ((200, 200, 300, 300), 0.6, 0),
]


@pytest.mark.parametrize(
    ("dets", "agnostic", "expected"),
    [
        pytest.param(_DETS, None, [_DETS[1], _DETS[2], _DETS[3]], id="class-aware"),
        pytest.param(_DETS, True, [_DETS[1], _DETS[3]], id="agnostic"),
        pytest.param([], None, [], id="empty"),
    ],
)
def test_nms(dets, agnostic, expected) -> None:
    """Nms keeps the highest scoring box per overlap group, sorted by score, same objects."""
    assert nms(dets, 0.5, agnostic=agnostic) == expected


def test_nms_matches_reference() -> None:
    """Nms agrees with the pure python greedy reference on random detections."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        xy = rng.integers(0, 200, (30, 2))
        wh = rng.integers(1, 100, (30, 2))
        dets = [
            ((int(x), int(y), int(x + w), int(y + h)), float(s), int(c))
            for (x, y), (w, h), s, c in zip(xy, wh, rng.uniform(0.01, 1, 30), rng.integers(0, 3, 30))
        ]
        for agnostic in (None, True):
            for thres in (0.3, 0.5, 0.7):
                assert nms(dets, thres, agnostic=agnostic) == _reference_nms(dets, thres, agnostic)
