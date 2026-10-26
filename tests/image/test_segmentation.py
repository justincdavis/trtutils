# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/postprocessors/_segmentation.py -- YOLO-style mask decoding."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from trtutils.image.postprocessors._segmentation import (
    _decode_masks,
    get_segmentations,
    postprocess_yolo_seg,
)

pytestmark = pytest.mark.cpu


def _make_output0(pred: np.ndarray) -> np.ndarray:
    """Wrap a (C, N) prediction tensor into the (1, C, N) engine output shape."""
    return pred[None].astype(np.float32)


def _make_proto(nm: int, mh: int, mw: int, fill: float = -10.0) -> np.ndarray:
    """Build a (1, nm, mh, mw) prototype tensor filled with a constant logit."""
    return np.full((1, nm, mh, mw), fill, dtype=np.float32)


def _single_det(nm: int, cls_scores: list[float], logit: float, mh: int = 8, mw: int = 8) -> list:
    """One (12, 12, 20, 20) detection whose mask is the constant sigmoid(logit)."""
    cxcywh = np.array([[16.0], [16.0], [8.0], [8.0]], dtype=np.float32)
    scores = np.array(cls_scores, dtype=np.float32)[:, None]
    coeffs = np.zeros((nm, 1), dtype=np.float32)
    coeffs[0] = 1.0
    proto = _make_proto(nm, mh, mw, fill=0.0)
    proto[0, 0] = logit
    return [_make_output0(np.concatenate([cxcywh, scores, coeffs], axis=0)), proto]


@pytest.mark.parametrize(
    ("ratio", "padding", "input_size", "bbox"),
    [
        pytest.param(0.5, 4.0, (32, 32), [16.0, 16.0, 32.0, 32.0], id="letterboxed"),
        pytest.param(1.0, 0.0, (32, 32), [12.0, 12.0, 20.0, 20.0], id="identity"),
    ],
)
def test_postprocess_round_trip_to_get_segmentations(ratio, padding, input_size, bbox) -> None:
    """A detection is unletterboxed exactly and flows through get_segmentations."""
    outputs = _single_det(4, [0.1, 0.05, 0.9], logit=10.0)  # class 2 wins
    postprocessed = postprocess_yolo_seg(
        outputs, [(ratio, ratio)], [(padding, padding)], input_size=input_size
    )
    assert len(postprocessed) == 1
    bboxes, scores, class_ids, masks = postprocessed[0]
    np.testing.assert_allclose(bboxes[0], bbox)
    assert scores[0] == pytest.approx(0.9)
    assert class_ids[0] == 2
    orig = round((input_size[0] - 2 * padding) / ratio)
    assert masks.shape == (1, orig, orig)
    assert masks.dtype == np.uint8

    segmentations = get_segmentations(postprocessed)
    assert len(segmentations) == 1
    assert len(segmentations[0]) == 1
    seg_bbox, score, class_id, mask = segmentations[0][0]
    assert seg_bbox == tuple(int(v) for v in bbox)
    assert isinstance(score, float)
    assert class_id == 2
    np.testing.assert_array_equal(mask, masks[0])


@pytest.mark.parametrize(
    ("logit", "mask_thres", "filled"),
    [
        pytest.param(10.0, 0.5, True, id="default-thres-filled"),
        pytest.param(1.0, 0.9, False, id="above-thres-empty"),
        pytest.param(-10.0, 0.0, True, id="explicit-zero-filled"),
    ],
)
def test_mask_thres_binarizes_inside_bbox(logit: float, mask_thres: float, filled: bool) -> None:
    """Masks are {0, 1}, zero outside their bbox, and thresholded by mask_thres inside it."""
    outputs = _single_det(2, [0.9, 0.1], logit=logit)
    result = postprocess_yolo_seg(
        outputs, [(1.0, 1.0)], [(0.0, 0.0)], mask_thres=mask_thres, input_size=(32, 32)
    )
    bboxes, _scores, _class_ids, masks = result[0]
    mask = masks[0]

    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1}
    x1, y1, x2, y2 = (round(v) for v in bboxes[0])
    outside = mask.copy()
    outside[y1:y2, x1:x2] = 0
    assert not outside.any()
    assert bool(mask[y1:y2, x1:x2].all()) is filled
    assert bool(mask[y1:y2, x1:x2].any()) is filled


def test_batch_of_two_images_returns_two_entries() -> None:
    """A batch of 2 images produces 2 postprocessed entries with independent detections."""
    nm, mh, mw = 2, 4, 4
    cxcywh0 = np.array([[4.0, 12.0], [4.0, 12.0], [4.0, 4.0], [4.0, 4.0]], dtype=np.float32)
    cls_scores0 = np.array([[0.9, 0.05], [0.05, 0.02]], dtype=np.float32)
    coeffs0 = np.zeros((nm, 2), dtype=np.float32)
    pred0 = np.concatenate([cxcywh0, cls_scores0, coeffs0], axis=0)

    cxcywh1 = np.array([[4.0, 12.0], [4.0, 12.0], [4.0, 4.0], [4.0, 4.0]], dtype=np.float32)
    cls_scores1 = np.array([[0.8, 0.1], [0.1, 0.7]], dtype=np.float32)
    coeffs1 = np.zeros((nm, 2), dtype=np.float32)
    pred1 = np.concatenate([cxcywh1, cls_scores1, coeffs1], axis=0)

    output0 = np.stack([pred0, pred1], axis=0).astype(np.float32)
    proto = _make_proto(nm, mh, mw, fill=-10.0)
    proto = np.repeat(proto, 2, axis=0)

    result = postprocess_yolo_seg(
        [output0, proto],
        ratios=[(1.0, 1.0), (1.0, 1.0)],
        padding=[(0.0, 0.0), (0.0, 0.0)],
        conf_thres=0.5,
        input_size=(16, 16),
    )
    assert len(result) == 2
    assert len(result[0][0]) == 1
    assert len(result[1][0]) == 2


@pytest.mark.parametrize(
    ("conf_thres", "kept_classes"),
    [
        pytest.param(0.5, [0], id="filtered"),
        pytest.param(0.0, [0, 1], id="explicit-zero"),
        pytest.param(None, [0, 1], id="none"),
    ],
)
def test_get_segmentations_conf_thres(conf_thres, kept_classes: list[int]) -> None:
    """get_segmentations drops entries below the confidence threshold, keeping masks aligned."""
    bboxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.2], dtype=np.float32)
    class_ids = np.array([0, 1], dtype=np.int64)
    masks = np.stack([np.full((5, 5), i, dtype=np.uint8) for i in range(2)])

    result = get_segmentations([[bboxes, scores, class_ids, masks]], conf_thres=conf_thres)
    assert [seg[2] for seg in result[0]] == kept_classes
    assert [int(seg[3][0, 0]) for seg in result[0]] == kept_classes


def test_box_cropped_masks_match_full_resize_reference() -> None:
    """Resizing only the box region matches resizing the whole mask then cropping."""
    rng = np.random.default_rng(0)
    nm, mh, mw = 4, 40, 40
    input_size = (160, 160)
    ratios = (0.25, 0.25)
    padding = (0.0, 30.0)
    proto = rng.normal(size=(nm, mh, mw)).astype(np.float32) * 3
    coeffs = rng.normal(size=(3, nm)).astype(np.float32)
    boxes = np.array(
        [[10.0, 5.0, 200.0, 90.0], [300.0, 100.0, 500.0, 300.0], [0.0, 0.0, 640.0, 360.0]],
        dtype=np.float32,
    )

    masks = _decode_masks(coeffs, proto, boxes, ratios, padding, input_size, 0.5)

    orig_w, orig_h = 640, 400
    probs = 1.0 / (1.0 + np.exp(-(coeffs @ proto.reshape(nm, -1)).reshape(-1, mh, mw)))
    # content region: rows 30..130 of the 160 input -> proto rows 7.5..32.5
    y0, y1 = 7, 33
    assert masks.shape == (3, orig_h, orig_w)
    for idx, (bx1, by1, bx2, by2) in enumerate(boxes):
        ref = cv2.resize(probs[idx, y0:y1], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) >= 0.5
        region = (slice(int(by1), min(int(np.ceil(by2)), orig_h)), slice(int(bx1), int(bx2)))
        mismatch = np.mean(masks[idx][region] != ref[region])
        assert mismatch < 0.02
        outside = masks[idx].copy()
        outside[region] = 0
        assert not outside.any()
