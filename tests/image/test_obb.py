# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/postprocessors/_obb.py -- YOLO-style OBB decoding and NMS."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trtutils.image.postprocessors._obb import (
    _probiou,
    _rotated_nms_indices,
    get_obb_detections,
    postprocess_yolo_obb,
)

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


def _post(
    dets: list[tuple[float, float, float, float, float, int, float]],
    nc: int = 1,
    ratio: float = 1.0,
    padding: tuple[float, float] = (0.0, 0.0),
) -> list[np.ndarray]:
    return postprocess_yolo_obb(
        [_make_output0(dets, nc)], [(ratio, ratio)], [padding], conf_thres=0.1
    )[0]


_HALF = 10.0 * math.sqrt(2) / 2.0


@pytest.mark.parametrize(
    ("det", "nc", "ratio", "padding", "bbox", "rbox"),
    [
        pytest.param(
            (50.0, 50.0, 20.0, 10.0, 0.0, 2, 0.9),
            3,
            1.0,
            (0.0, 0.0),
            [40.0, 45.0, 60.0, 55.0],
            [50.0, 50.0, 20.0, 10.0, 0.0],
            id="axis-aligned-class-2",
        ),
        # cx/cy/w/h are unletterboxed, the angle passes through unchanged
        pytest.param(
            (110.0, 60.0, 40.0, 20.0, math.pi / 2, 0, 0.9),
            1,
            2.0,
            (10.0, 5.0),
            [45.0, 17.5, 55.0, 37.5],
            [50.0, 27.5, 20.0, 10.0, math.pi / 2],
            id="letterboxed-rotated",
        ),
        # a 45 degree square's enclosing box grows by sqrt(2)
        pytest.param(
            (50.0, 50.0, 10.0, 10.0, math.pi / 4, 0, 0.9),
            1,
            1.0,
            (0.0, 0.0),
            [50 - _HALF, 50 - _HALF, 50 + _HALF, 50 + _HALF],
            [50.0, 50.0, 10.0, 10.0, math.pi / 4],
            id="square-45-degrees",
        ),
    ],
)
def test_round_trip_postprocess_to_get_obb_detections(
    det, nc: int, ratio: float, padding, bbox, rbox
) -> None:
    """A detection is unletterboxed with its enclosing box, then flows through get_obb_detections."""
    postprocessed = _post([det], nc=nc, ratio=ratio, padding=padding)
    bboxes, scores, class_ids, rboxes = postprocessed
    np.testing.assert_allclose(bboxes, [bbox], atol=1e-4)
    np.testing.assert_allclose(rboxes, [rbox], atol=1e-4)
    np.testing.assert_allclose(scores, [det[6]], atol=1e-6)
    np.testing.assert_array_equal(class_ids, [det[5]])

    detections = get_obb_detections([postprocessed], conf_thres=0.5)
    assert len(detections) == 1
    assert len(detections[0]) == 1
    out_rbox, score, class_id = detections[0][0]
    np.testing.assert_allclose(out_rbox, rbox, atol=1e-4)
    assert score == pytest.approx(det[6], abs=1e-6)
    assert class_id == det[5]


def test_batch_of_two_images_returns_two_entries() -> None:
    """A batch of 2 images produces one postprocessed entry per image."""
    output0 = np.concatenate(
        [
            _make_output0([(50.0, 50.0, 20.0, 10.0, 0.0, 0, 0.9)], nc=1),
            _make_output0([(30.0, 30.0, 10.0, 10.0, 0.0, 0, 0.8)], nc=1),
        ],
        axis=0,
    )
    postprocessed = postprocess_yolo_obb(
        [output0], [(1.0, 1.0)] * 2, [(0.0, 0.0)] * 2, conf_thres=0.5
    )
    detections = get_obb_detections(postprocessed, conf_thres=0.5)
    assert [len(d) for d in detections] == [1, 1]


def test_probiou_identical_disjoint_and_symmetric() -> None:
    """Probiou is ~1 for identical boxes, ~0 for far apart ones, and symmetric."""
    rboxes = np.array(
        [
            [50.0, 50.0, 20.0, 10.0, 0.3],
            [50.0, 50.0, 20.0, 10.0, 0.3],
            [500.0, 500.0, 20.0, 10.0, 0.3],
        ],
        dtype=np.float32,
    )
    iou = _probiou(rboxes, rboxes)
    assert iou.shape == (3, 3)
    assert iou[0, 1] == pytest.approx(1.0, abs=1e-3)
    assert iou[0, 2] == pytest.approx(0.0, abs=1e-3)
    np.testing.assert_allclose(iou, iou.T, atol=1e-6)


@pytest.mark.parametrize(
    ("second", "kept"),
    [
        pytest.param((51.0, 50.0, 40.0, 10.0, 0.1, 0), [0], id="rotated-overlap-suppressed"),
        pytest.param((51.0, 50.0, 40.0, 10.0, 0.1, 1), [0, 1], id="other-class-kept"),
        pytest.param((50.0, 50.0, 40.0, 10.0, math.pi / 2, 0), [0, 1], id="crossed-kept"),
        pytest.param((150.0, 50.0, 40.0, 10.0, 0.0, 0), [0, 1], id="disjoint-kept"),
    ],
)
def test_rotated_nms_pairs(second: tuple, kept: list[int]) -> None:
    """Only a heavily overlapping same-class rotated box is suppressed."""
    rboxes = np.array([[50.0, 50.0, 40.0, 10.0, 0.0], second[:5]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    class_ids = np.array([0, second[5]])
    assert _rotated_nms_indices(rboxes, scores, class_ids, 0.5, 300).tolist() == kept


def test_rotated_nms_is_ultralytics_fast_nms() -> None:
    """A box overlapping only an already suppressed box is still dropped, as in ultralytics."""
    rboxes = np.array(
        [
            [50.0, 50.0, 40.0, 10.0, 0.0],
            [60.0, 50.0, 40.0, 10.0, 0.0],
            [70.0, 50.0, 40.0, 10.0, 0.0],
        ],
        dtype=np.float32,
    )
    iou = _probiou(rboxes, rboxes)
    assert iou[0, 1] >= 0.5
    assert iou[1, 2] >= 0.5
    assert iou[0, 2] < 0.5
    scores = np.array([0.7, 0.8, 0.9], dtype=np.float32)
    assert _rotated_nms_indices(rboxes, scores, np.zeros(3), 0.5, 300).tolist() == [2]


@pytest.mark.parametrize(
    "max_det",
    [
        pytest.param(1, id="max-det-1"),
        pytest.param(3, id="max-det-3"),
        pytest.param(10, id="max-det-10"),
    ],
)
def test_rotated_nms_max_det_score_sorted(max_det: int) -> None:
    """Disjoint boxes come back sorted by score and capped at max_det, across blocks."""
    n = 10
    rboxes = np.zeros((n, 5), dtype=np.float32)
    rboxes[:, 0] = np.arange(n) * 100.0
    rboxes[:, 2:4] = 10.0
    scores = np.random.default_rng(0).permutation(n).astype(np.float32) / n
    keep = _rotated_nms_indices(rboxes, scores, np.zeros(n), 0.5, max_det, block=4)
    assert len(keep) == max_det
    assert keep.tolist() == np.argsort(-scores)[:max_det].tolist()


def test_postprocess_obb_sorted_by_score() -> None:
    """Postprocessed OBB results are ordered by descending score."""
    dets = [(x * 50.0, 50.0, 20.0, 10.0, 0.0, 0, s) for x, s in enumerate([0.3, 0.9, 0.6], 1)]
    _bboxes, scores, _class_ids, rboxes = _post(dets)
    np.testing.assert_allclose(scores, [0.9, 0.6, 0.3], atol=1e-6)
    np.testing.assert_allclose(rboxes[:, 0], [100.0, 150.0, 50.0])
