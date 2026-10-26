# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/postprocessors/_common.py -- box helpers shared by postprocessors."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.postprocessors._common import (
    cxcywh_to_xyxy,
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
