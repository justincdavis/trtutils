# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for nms_indices in postprocessors/_common.py -- numpy path parity with cv2."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils._flags import FLAGS
from trtutils.image.postprocessors._common import nms_indices

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(not FLAGS.FOUND_CV2, reason="the reference is the cv2 branch"),
]


@pytest.fixture(autouse=True)
def _numpy_nms(monkeypatch) -> None:
    """Run the numpy path so every test compares it against cv2."""
    monkeypatch.setattr(FLAGS, "FOUND_CV2", False)


def _cv2_nms(*args: object) -> np.ndarray:
    """Reference result: nms_indices' own cv2.dnn.NMSBoxesBatched branch."""
    FLAGS.FOUND_CV2 = True
    expected = nms_indices(*args)
    FLAGS.FOUND_CV2 = False
    return expected


def _make(
    n: int,
    num_classes: int,
    seed: int,
    *,
    extent: float = 640.0,
    box_scale: float = 0.15,
    origin: float = 0.0,
    quantize: int = 0,
    zero_frac: float = 0.0,
    duplicates: bool = False,
    clustered: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if clustered:
        centers = rng.random((max(n // 10, 1), 2)) * extent
        xy = centers[rng.integers(0, centers.shape[0], n)] + rng.normal(0, extent * 0.02, (n, 2))
    else:
        xy = rng.random((n, 2)) * extent * 0.8 + origin
    wh = rng.random((n, 2)) * extent * box_scale
    wh[rng.random(n) < zero_frac] = 0
    boxes = np.concatenate([xy, xy + wh], axis=1).astype(np.float32)
    if duplicates and n > 1:
        boxes[1::2] = boxes[0::2][: boxes[1::2].shape[0]]
    scores = rng.random(n).astype(np.float32)
    if quantize:
        scores = (np.round(scores * quantize) / quantize).astype(np.float32)
    labels = rng.integers(0, num_classes, n)
    return boxes, scores, labels


_SCENES = [
    pytest.param({"n": 0, "num_classes": 3}, id="size-0"),
    pytest.param({"n": 1, "num_classes": 3}, id="size-1"),
    pytest.param({"n": 2, "num_classes": 1}, id="size-2-one-class"),
    pytest.param({"n": 2, "num_classes": 3, "clustered": True}, id="size-2-clustered"),
    pytest.param({"n": 50, "num_classes": 1}, id="50-boxes-1-class"),
    pytest.param({"n": 50, "num_classes": 3}, id="50-boxes-3-classes"),
    pytest.param({"n": 50, "num_classes": 80}, id="50-boxes-80-classes"),
    pytest.param({"n": 50, "num_classes": 3, "box_scale": 0.6}, id="50-boxes-large"),
    pytest.param({"n": 2000, "num_classes": 1}, id="2000-boxes-1-class"),
    pytest.param({"n": 2000, "num_classes": 80}, id="2000-boxes-80-classes"),
    pytest.param({"n": 2000, "num_classes": 3, "clustered": True}, id="2000-boxes-clustered"),
    pytest.param({"n": 300, "num_classes": 80, "clustered": True}, id="300-boxes-clustered-80"),
    pytest.param({"n": 50, "num_classes": 3, "quantize": 4}, id="ties-coarse"),
    pytest.param({"n": 2000, "num_classes": 80, "quantize": 10}, id="ties-2000"),
    pytest.param({"n": 50, "num_classes": 3, "duplicates": True}, id="duplicate-boxes"),
    pytest.param(
        {"n": 50, "num_classes": 3, "duplicates": True, "quantize": 2}, id="duplicates-and-ties"
    ),
    pytest.param({"n": 50, "num_classes": 3, "zero_frac": 0.3}, id="zero-area-boxes"),
    pytest.param({"n": 50, "num_classes": 1, "zero_frac": 1.0}, id="all-zero-area"),
    pytest.param({"n": 300, "num_classes": 80, "zero_frac": 0.2}, id="zero-area-80-classes"),
    pytest.param(
        {"n": 50, "num_classes": 3, "origin": -300.0, "box_scale": 0.3}, id="negative-coords"
    ),
    pytest.param(
        {"n": 60, "num_classes": 2, "origin": -300.0, "box_scale": 1.5}, id="negative-spill-2"
    ),
    pytest.param(
        {"n": 300, "num_classes": 80, "origin": -300.0, "box_scale": 0.5},
        id="negative-coords-80-classes",
    ),
    pytest.param({"n": 50, "num_classes": 3, "extent": 4000.0}, id="large-coords"),
    pytest.param(
        {"n": 300, "num_classes": 80, "extent": 4000.0, "clustered": True}, id="large-clustered"
    ),
]
_SEEDS = [pytest.param(seed, id=f"seed{seed}") for seed in (0, 1, 2)]
_CONF = [
    pytest.param(None, id="conf-none"),
    pytest.param(0.25, id="conf-0.25"),
    pytest.param("score", id="conf-equals-a-score"),
]
_IOU = [pytest.param(t, id=f"iou-{t}") for t in (0.0, 0.45, 0.7, 1.0)]
_MAX_DET = [
    pytest.param(None, id="maxdet-none"),
    pytest.param(1, id="maxdet-1"),
    pytest.param(10, id="maxdet-10"),
]


@pytest.mark.parametrize("max_det", _MAX_DET)
@pytest.mark.parametrize("iou_thres", _IOU)
@pytest.mark.parametrize("conf", _CONF)
@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("scene", _SCENES)
def test_nms_indices_matches_cv2(scene, seed, conf, iou_thres, max_det) -> None:
    """Kept indices equal cv2.dnn.NMSBoxesBatched exactly."""
    boxes, scores, labels = _make(seed=seed, **scene)
    first = float(scores[0]) if scores.size else 0.5
    conf_thres = first if conf == "score" else conf
    expected = _cv2_nms(boxes, scores, labels, conf_thres, iou_thres, max_det)
    actual = nms_indices(boxes, scores, labels, conf_thres, iou_thres, max_det)
    assert np.array_equal(actual, expected)


def test_nms_indices_offset_uses_filtered_boxes() -> None:
    """The class offset comes from every box, including ones under the score threshold."""
    boxes = np.array([[0, 0, 10, 10], [-6, -6, 4, 4], [0, 0, 500, 500]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.1], dtype=np.float32)
    labels = np.array([0, 1, 1])
    expected = _cv2_nms(boxes, scores, labels, 0.5, 0.0, None)
    assert np.array_equal(nms_indices(boxes, scores, labels, 0.5, 0.0, None), expected)


def test_nms_indices_offset_is_max_coordinate_plus_one() -> None:
    """A class-1 box shifted to exactly touch the class-0 box is suppressed."""
    boxes = np.array([[0, 0, 10, 10], [-11, -11, -10.5, -10.5]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    labels = np.array([0, 1])
    expected = _cv2_nms(boxes, scores, labels, 0.1, 0.0, None)
    assert expected.tolist() == [0]
    assert np.array_equal(nms_indices(boxes, scores, labels, 0.1, 0.0, None), expected)
