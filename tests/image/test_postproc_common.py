# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the box helpers shared by the postprocessors."""

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


def test_nms_dedupes_same_class() -> None:
    """Overlapping same-class boxes keep only the higher score."""
    keep = nms_indices(_OVERLAP, _SCORES, np.array([1, 1]), 0.1, 0.5)
    assert keep.tolist() == [0]


def test_nms_keeps_different_classes() -> None:
    """Overlapping boxes of different classes are both kept."""
    keep = nms_indices(_OVERLAP, _SCORES, np.array([0, 1]), 0.1, 0.5)
    assert sorted(keep.tolist()) == [0, 1]


def test_nms_zero_labels_dedupes_everything() -> None:
    """All-zero labels (pose usage) behave as plain NMS."""
    keep = nms_indices(_OVERLAP, _SCORES, np.zeros(2, dtype=int), None, 0.5)
    assert keep.tolist() == [0]


def test_nms_empty_input() -> None:
    """No boxes returns no indices."""
    keep = nms_indices(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int), 0.1, 0.5)
    assert keep.shape == (0,)


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
