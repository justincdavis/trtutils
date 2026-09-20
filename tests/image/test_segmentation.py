# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the YOLO-style segmentation postprocessor."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.postprocessors._segmentation import get_segmentations, postprocess_yolo_seg

pytestmark = pytest.mark.cpu


def _make_output0(pred: np.ndarray) -> np.ndarray:
    """Wrap a (C, N) prediction tensor into the (1, C, N) engine output shape."""
    return pred[None].astype(np.float32)


def _make_proto(nm: int, mh: int, mw: int, fill: float = -10.0) -> np.ndarray:
    """Build a (1, nm, mh, mw) prototype tensor filled with a constant logit."""
    return np.full((1, nm, mh, mw), fill, dtype=np.float32)


def test_postprocess_unletterboxes_bbox_exactly() -> None:
    """A single detection's bbox is remapped out of letterboxed coords exactly."""
    nm, mh, mw = 4, 8, 8
    # cxcywh (network coords) -> xyxy (12, 12, 20, 20)
    cxcywh = np.array([[16.0], [16.0], [8.0], [8.0]], dtype=np.float32)
    cls_scores = np.array([[0.1], [0.9], [0.05]], dtype=np.float32)  # class 1 wins
    coeffs = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)

    pred = np.concatenate([cxcywh, cls_scores, coeffs], axis=0)  # (4+nc+nm, 1)
    output0 = _make_output0(pred)

    # proto channel 0 is a large positive constant -> sigmoid ~= 1 everywhere,
    # remaining channels are zero so the one-hot coeff isolates channel 0
    proto = _make_proto(nm, mh, mw, fill=0.0)
    proto[0, 0] = 10.0

    ratios = [(0.5, 0.5)]
    padding = [(4.0, 4.0)]
    input_size = (32, 32)

    result = postprocess_yolo_seg(
        [output0, proto],
        ratios,
        padding,
        input_size=input_size,
    )
    assert len(result) == 1
    bboxes, scores, class_ids, masks = result[0]

    # orig = (coord - pad) / ratio
    np.testing.assert_allclose(bboxes[0], [16.0, 16.0, 32.0, 32.0])
    assert scores[0] == pytest.approx(0.9)
    assert class_ids[0] == 1

    orig_w = round((input_size[0] - 2 * padding[0][0]) / ratios[0][0])
    orig_h = round((input_size[1] - 2 * padding[0][1]) / ratios[0][1])
    assert masks.shape == (1, orig_h, orig_w)
    assert masks.dtype == np.uint8


def test_mask_is_zero_outside_bbox_and_binary_inside() -> None:
    """The decoded mask is thresholded to {0, 1} and zeroed outside its own bbox."""
    nm, mh, mw = 2, 8, 8
    cxcywh = np.array([[16.0], [16.0], [8.0], [8.0]], dtype=np.float32)
    cls_scores = np.array([[0.9], [0.1]], dtype=np.float32)
    coeffs = np.array([[1.0], [0.0]], dtype=np.float32)
    pred = np.concatenate([cxcywh, cls_scores, coeffs], axis=0)
    output0 = _make_output0(pred)

    proto = _make_proto(nm, mh, mw, fill=0.0)
    proto[0, 0] = 10.0  # sigmoid(10) ~= 1 everywhere in the content region

    ratios = [(1.0, 1.0)]
    padding = [(0.0, 0.0)]
    input_size = (32, 32)

    result = postprocess_yolo_seg([output0, proto], ratios, padding, input_size=input_size)
    bboxes, _scores, _class_ids, masks = result[0]
    mask = masks[0]

    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1}

    x1, y1, x2, y2 = (round(v) for v in bboxes[0])
    outside = mask.copy()
    outside[y1:y2, x1:x2] = 0
    assert not outside.any()
    assert mask[y1:y2, x1:x2].all()


def test_nms_dedupes_overlapping_same_class_boxes() -> None:
    """Two heavily overlapping same-class boxes are deduped, keeping the higher score."""
    nm, mh, mw = 2, 4, 4
    cxcywh = np.array(
        [[10.0, 11.0], [10.0, 11.0], [8.0, 8.0], [8.0, 8.0]],
        dtype=np.float32,
    )
    cls_scores = np.array([[0.9, 0.6]], dtype=np.float32)
    coeffs = np.zeros((nm, 2), dtype=np.float32)
    pred = np.concatenate([cxcywh, cls_scores, coeffs], axis=0)
    output0 = _make_output0(pred)
    proto = _make_proto(nm, mh, mw, fill=-10.0)

    result = postprocess_yolo_seg(
        [output0, proto],
        ratios=[(1.0, 1.0)],
        padding=[(0.0, 0.0)],
        input_size=(16, 16),
        nms_iou_thres=0.5,
    )
    bboxes, scores, _class_ids, masks = result[0]
    assert len(bboxes) == 1
    assert scores[0] == pytest.approx(0.9)
    assert len(masks) == 1


def test_batch_of_two_images_returns_two_entries() -> None:
    """A batch of 2 images produces 2 postprocessed entries with independent detections."""
    nm, mh, mw = 2, 4, 4
    # image 0: one detection above conf_thres, one below
    cxcywh0 = np.array([[4.0, 12.0], [4.0, 12.0], [4.0, 4.0], [4.0, 4.0]], dtype=np.float32)
    cls_scores0 = np.array([[0.9, 0.05], [0.05, 0.02]], dtype=np.float32)
    coeffs0 = np.zeros((nm, 2), dtype=np.float32)
    pred0 = np.concatenate([cxcywh0, cls_scores0, coeffs0], axis=0)

    # image 1: two non-overlapping detections, both above conf_thres
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


def test_round_trip_postprocess_to_get_segmentations() -> None:
    """postprocess_yolo_seg output flows through get_segmentations correctly."""
    nm, mh, mw = 4, 8, 8
    cxcywh = np.array([[16.0], [16.0], [8.0], [8.0]], dtype=np.float32)
    cls_scores = np.array([[0.1], [0.05], [0.9]], dtype=np.float32)  # class 2 wins
    coeffs = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    pred = np.concatenate([cxcywh, cls_scores, coeffs], axis=0)
    output0 = _make_output0(pred)

    proto = _make_proto(nm, mh, mw, fill=0.0)
    proto[0, 0] = 10.0

    postprocessed = postprocess_yolo_seg(
        [output0, proto],
        ratios=[(1.0, 1.0)],
        padding=[(0.0, 0.0)],
        input_size=(16, 16),
    )

    segmentations = get_segmentations(postprocessed)
    assert len(segmentations) == 1
    assert len(segmentations[0]) == 1

    bbox, score, class_id, mask = segmentations[0][0]
    assert isinstance(bbox, tuple)
    assert all(isinstance(v, int) for v in bbox)
    assert isinstance(score, float)
    assert class_id == 2
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.uint8


def test_get_segmentations_filters_by_conf_thres() -> None:
    """get_segmentations drops entries below the confidence threshold."""
    bboxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.2], dtype=np.float32)
    class_ids = np.array([0, 1], dtype=np.int64)
    masks = np.zeros((2, 5, 5), dtype=np.uint8)

    result = get_segmentations([[bboxes, scores, class_ids, masks]], conf_thres=0.5)
    assert len(result[0]) == 1
    assert result[0][0][2] == 0
