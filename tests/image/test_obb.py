# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the YOLO-style OBB (oriented bounding box) postprocessor."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trtutils.image.postprocessors._obb import get_obb_detections, postprocess_yolo_obb

pytestmark = pytest.mark.cpu


def _make_output0(
    dets: list[tuple[float, float, float, float, float, int, float]],
    nc: int,
) -> np.ndarray:
    """
    Build a synthetic ``output0`` tensor of shape (1, 4+nc+1, N) from detections.

    Each detection is (cx, cy, w, h, angle, class_id, score).
    """
    n = len(dets)
    pred = np.zeros((n, 4 + nc + 1), dtype=np.float32)
    for i, (cx, cy, w, h, angle, class_id, score) in enumerate(dets):
        pred[i, 0] = cx
        pred[i, 1] = cy
        pred[i, 2] = w
        pred[i, 3] = h
        pred[i, 4 + class_id] = score
        pred[i, -1] = angle
    return pred.T[None]  # (1, 4+nc+1, N)


def test_round_trip_postprocess_to_get_obb_detections() -> None:
    """postprocess_yolo_obb -> get_obb_detections round trips a single detection."""
    dets = [(50.0, 50.0, 20.0, 10.0, 0.0, 0, 0.9)]
    output0 = _make_output0(dets, nc=3)

    postprocessed = postprocess_yolo_obb(
        [output0],
        ratios=[(1.0, 1.0)],
        padding=[(0.0, 0.0)],
        conf_thres=0.5,
    )
    assert len(postprocessed) == 1

    bboxes, scores, class_ids, rboxes = postprocessed[0]
    assert bboxes.shape == (1, 4)
    assert rboxes.shape == (1, 5)
    np.testing.assert_allclose(rboxes[0], [50.0, 50.0, 20.0, 10.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(scores, [0.9], atol=1e-6)
    np.testing.assert_array_equal(class_ids, [0])

    detections = get_obb_detections(postprocessed, conf_thres=0.5)
    assert len(detections) == 1
    assert len(detections[0]) == 1
    rbox, score, class_id = detections[0][0]
    np.testing.assert_allclose(rbox, (50.0, 50.0, 20.0, 10.0, 0.0), atol=1e-4)
    assert score == pytest.approx(0.9, abs=1e-6)
    assert class_id == 0


def test_unletterboxing_scales_center_and_size_but_not_angle() -> None:
    """cx/cy/w/h are unletterboxed with ratio/padding; the angle passes through unchanged."""
    angle = math.pi / 6  # 30 degrees
    dets = [(110.0, 60.0, 40.0, 20.0, angle, 0, 0.9)]
    output0 = _make_output0(dets, nc=1)

    postprocessed = postprocess_yolo_obb(
        [output0],
        ratios=[(2.0, 2.0)],
        padding=[(10.0, 5.0)],
        conf_thres=0.5,
    )
    _bboxes, _scores, _class_ids, rboxes = postprocessed[0]

    expected_cx = (110.0 - 10.0) / 2.0
    expected_cy = (60.0 - 5.0) / 2.0
    expected_w = 40.0 / 2.0
    expected_h = 20.0 / 2.0
    np.testing.assert_allclose(
        rboxes[0],
        [expected_cx, expected_cy, expected_w, expected_h, angle],
        atol=1e-4,
    )


def test_enclosing_bbox_matches_cxcywh_when_angle_is_zero() -> None:
    """With angle 0 the derived enclosing box equals the plain cxcywh box."""
    dets = [(50.0, 50.0, 20.0, 10.0, 0.0, 0, 0.9)]
    output0 = _make_output0(dets, nc=1)

    postprocessed = postprocess_yolo_obb(
        [output0],
        ratios=[(1.0, 1.0)],
        padding=[(0.0, 0.0)],
        conf_thres=0.5,
    )
    bboxes, *_rest = postprocessed[0]

    np.testing.assert_allclose(bboxes[0], [40.0, 45.0, 60.0, 55.0], atol=1e-4)


def test_enclosing_bbox_scales_by_sqrt2_for_45_degree_square() -> None:
    """A 45-degree-rotated square's enclosing box is scaled by sqrt(2)."""
    side = 10.0
    dets = [(50.0, 50.0, side, side, math.pi / 4, 0, 0.9)]
    output0 = _make_output0(dets, nc=1)

    postprocessed = postprocess_yolo_obb(
        [output0],
        ratios=[(1.0, 1.0)],
        padding=[(0.0, 0.0)],
        conf_thres=0.5,
    )
    bboxes, *_rest = postprocessed[0]

    half_extent = side * math.sqrt(2) / 2.0
    expected = [50.0 - half_extent, 50.0 - half_extent, 50.0 + half_extent, 50.0 + half_extent]
    np.testing.assert_allclose(bboxes[0], expected, atol=1e-4)


def test_rotated_nms_dedupes_same_class_but_keeps_different_classes() -> None:
    """Overlapping same-class rboxes are deduped; overlapping different-class ones are kept."""
    dets = [
        (50.0, 50.0, 20.0, 20.0, 0.0, 0, 0.9),  # class 0, kept
        (52.0, 52.0, 20.0, 20.0, 0.0, 0, 0.6),  # class 0, suppressed (heavy overlap)
        (50.0, 50.0, 20.0, 20.0, 0.0, 1, 0.8),  # class 1, kept (different class)
    ]
    output0 = _make_output0(dets, nc=2)

    postprocessed = postprocess_yolo_obb(
        [output0],
        ratios=[(1.0, 1.0)],
        padding=[(0.0, 0.0)],
        conf_thres=0.1,
        nms_iou_thres=0.5,
    )
    _bboxes, scores, class_ids, _rboxes = postprocessed[0]

    assert len(scores) == 2
    kept = sorted(zip(class_ids.tolist(), scores.tolist()))
    assert kept == [(0, pytest.approx(0.9)), (1, pytest.approx(0.8))]


def test_batch_of_two_images_returns_two_entries() -> None:
    """A batch of 2 images produces one postprocessed entry per image."""
    dets0 = [(50.0, 50.0, 20.0, 10.0, 0.0, 0, 0.9)]
    dets1 = [(30.0, 30.0, 10.0, 10.0, 0.0, 0, 0.8)]
    output0 = np.concatenate(
        [_make_output0(dets0, nc=1), _make_output0(dets1, nc=1)],
        axis=0,
    )

    postprocessed = postprocess_yolo_obb(
        [output0],
        ratios=[(1.0, 1.0), (1.0, 1.0)],
        padding=[(0.0, 0.0), (0.0, 0.0)],
        conf_thres=0.5,
    )
    assert len(postprocessed) == 2

    detections = get_obb_detections(postprocessed, conf_thres=0.5)
    assert len(detections) == 2
    assert len(detections[0]) == 1
    assert len(detections[1]) == 1


def test_non_standard_class_count_is_derived_not_hardcoded() -> None:
    """Nc is derived from the tensor shape, proven with a non-15/80 class count."""
    nc = 3
    dets = [(50.0, 50.0, 20.0, 10.0, 0.0, 2, 0.9)]  # class index 2 of 3
    output0 = _make_output0(dets, nc=nc)
    assert output0.shape == (1, 4 + nc + 1, 1)

    postprocessed = postprocess_yolo_obb(
        [output0],
        ratios=[(1.0, 1.0)],
        padding=[(0.0, 0.0)],
        conf_thres=0.5,
    )
    _bboxes, _scores, class_ids, _rboxes = postprocessed[0]
    np.testing.assert_array_equal(class_ids, [2])
