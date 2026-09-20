# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/postprocessors/ -- detection and classification postprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.postprocessors import (
    get_classifications,
    get_detections,
    postprocess_classifications,
    postprocess_detr,
    postprocess_efficient_nms,
    postprocess_rfdetr,
    postprocess_yolov10,
)

pytestmark = pytest.mark.cpu

NUM_DETS = 10
INPUT_SIZE = (640, 640)
BATCH_SIZES = [
    pytest.param(1, id="batch1"),
    pytest.param(2, id="batch2"),
    pytest.param(4, id="batch4"),
]


def _boxes(batch_size: int) -> np.ndarray:
    """(B, NUM_DETS, 4) xyxy boxes: 100 + 10i per detection, shifted 50 per batch element."""
    base = np.arange(NUM_DETS, dtype=np.float32)[:, None] * 10 + np.array(
        [100, 100, 200, 200], dtype=np.float32
    )
    return np.stack([base + b * 50 for b in range(batch_size)])


def _scores(batch_size: int) -> np.ndarray:
    """(B, NUM_DETS) scores descending from 0.9 in steps of 0.05."""
    return np.tile(0.9 - np.arange(NUM_DETS, dtype=np.float32) * 0.05, (batch_size, 1))


def _classes(batch_size: int) -> np.ndarray:
    """(B, NUM_DETS) class ids 0..NUM_DETS-1."""
    return np.tile(np.arange(NUM_DETS, dtype=np.float32), (batch_size, 1))


def _yolov10_output(batch_size: int) -> list[np.ndarray]:
    """(B, 300, 6) rows of x1, y1, x2, y2, score, class_id with unused rows zeroed."""
    output = np.zeros((batch_size, 300, 6), dtype=np.float32)
    output[:, :NUM_DETS, :4] = _boxes(batch_size)
    output[:, :NUM_DETS, 4] = _scores(batch_size)
    output[:, :NUM_DETS, 5] = _classes(batch_size)
    return [output]


def _efficient_nms_output(batch_size: int) -> list[np.ndarray]:
    """EfficientNMS plugin outputs: num_dets (B, 1), boxes (B, 100, 4), scores (B, 100), classes (B, 100)."""
    boxes = np.zeros((batch_size, 100, 4), dtype=np.float32)
    scores = np.zeros((batch_size, 100), dtype=np.float32)
    classes = np.zeros((batch_size, 100), dtype=np.int32)
    boxes[:, :NUM_DETS] = _boxes(batch_size)
    scores[:, :NUM_DETS] = _scores(batch_size)
    classes[:, :NUM_DETS] = _classes(batch_size)
    return [np.full((batch_size, 1), NUM_DETS, dtype=np.int32), boxes, scores, classes]


def _rfdetr_output(batch_size: int) -> list[np.ndarray]:
    """RF-DETR outputs: dets (B, 300, 4) normalized cxcywh and 1-indexed logits (B, 300, 81)."""
    dets = np.zeros((batch_size, 300, 4), dtype=np.float32)
    logits = np.full((batch_size, 300, 81), -10.0, dtype=np.float32)
    boxes = _boxes(batch_size)
    dets[:, :NUM_DETS, 0] = (boxes[..., 0] + boxes[..., 2]) / 2 / INPUT_SIZE[0]
    dets[:, :NUM_DETS, 1] = (boxes[..., 1] + boxes[..., 3]) / 2 / INPUT_SIZE[1]
    dets[:, :NUM_DETS, 2] = (boxes[..., 2] - boxes[..., 0]) / INPUT_SIZE[0]
    dets[:, :NUM_DETS, 3] = (boxes[..., 3] - boxes[..., 1]) / INPUT_SIZE[1]
    for b in range(batch_size):
        for i in range(NUM_DETS):
            logits[b, i, i + 1] = 3.0 - i * 0.5
    return [dets, logits]


def _detr_output(batch_size: int) -> list[np.ndarray]:
    """DETR outputs: scores (B, 300), labels (B, 300), boxes (B, 300, 4)."""
    scores = np.zeros((batch_size, 300), dtype=np.float32)
    labels = np.zeros((batch_size, 300), dtype=np.float32)
    boxes = np.zeros((batch_size, 300, 4), dtype=np.float32)
    scores[:, :NUM_DETS] = _scores(batch_size)
    labels[:, :NUM_DETS] = _classes(batch_size)
    boxes[:, :NUM_DETS] = _boxes(batch_size)
    return [scores, labels, boxes]


def _classification_output(batch_size: int, num_classes: int = 1000) -> list[np.ndarray]:
    """(B, num_classes) logits where image b peaks at class b and then class b + 1."""
    output = np.random.default_rng(42).standard_normal((batch_size, num_classes)).astype(np.float32)
    for b in range(batch_size):
        output[b, b] = 10.0
        output[b, b + 1] = 8.0
    return [output]


@pytest.mark.parametrize(
    ("postprocess", "make_output", "remaps"),
    [
        pytest.param(postprocess_yolov10, _yolov10_output, True, id="yolov10"),
        pytest.param(postprocess_efficient_nms, _efficient_nms_output, True, id="efficient-nms"),
        pytest.param(postprocess_rfdetr, _rfdetr_output, True, id="rfdetr"),
        pytest.param(postprocess_detr, _detr_output, False, id="detr"),
    ],
)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_postprocess_detections(batch_size, postprocess, make_output, remaps) -> None:
    """Batch postprocessing matches per-image calls, undoes ratios/padding, and filters by conf_thres."""
    outputs = make_output(batch_size)
    identity = ([(1.0, 1.0)] * batch_size, [(0.0, 0.0)] * batch_size)
    scaled = ([(2.0, 2.0)] * batch_size, [(10.0, 10.0)] * batch_size)

    results = postprocess(
        [o.copy() for o in outputs], *identity, conf_thres=0.1, input_size=INPUT_SIZE
    )
    remapped = postprocess(
        [o.copy() for o in outputs], *scaled, conf_thres=0.1, input_size=INPUT_SIZE
    )
    filtered = postprocess(
        [o.copy() for o in outputs], *identity, conf_thres=0.8, input_size=INPUT_SIZE
    )
    assert len(results) == len(remapped) == len(filtered) == batch_size

    for i in range(batch_size):
        bboxes, scores, class_ids = results[i]
        assert bboxes.shape == (NUM_DETS, 4)
        assert scores.shape == (NUM_DETS,)
        np.testing.assert_array_equal(class_ids, np.arange(NUM_DETS))
        np.testing.assert_allclose(bboxes, _boxes(batch_size)[i], atol=1e-3)

        single = postprocess(
            [o[i : i + 1].copy() for o in outputs],
            [identity[0][i]],
            [identity[1][i]],
            conf_thres=0.1,
            input_size=INPUT_SIZE,
        )[0]
        for full, part in zip(results[i], single):
            np.testing.assert_array_equal(full, part)

        # detr models receive the original image size and already emit original coordinates
        expected_remap = (bboxes - 10.0) / 2.0 if remaps else bboxes
        np.testing.assert_allclose(remapped[i][0], expected_remap, atol=1e-3)

        assert 0 < len(filtered[i][1]) < NUM_DETS
        assert np.all(filtered[i][1] >= 0.8)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_get_detections(batch_size) -> None:
    """get_detections() converts postprocessed arrays to (bbox, score, class_id) tuples and filters."""
    ratios = [(1.0, 1.0)] * batch_size
    padding = [(0.0, 0.0)] * batch_size
    postprocessed = postprocess_yolov10(_yolov10_output(batch_size), ratios, padding, conf_thres=0.1)

    detections = get_detections(postprocessed)
    assert len(detections) == batch_size
    for image_dets, (bboxes, scores, class_ids) in zip(detections, postprocessed):
        assert len(image_dets) == NUM_DETS
        for (bbox, score, class_id), exp_bbox, exp_score, exp_class in zip(
            image_dets, bboxes, scores, class_ids
        ):
            assert bbox == tuple(int(v) for v in exp_bbox)
            assert all(isinstance(v, int) for v in bbox)
            assert score == float(exp_score)
            assert isinstance(score, float)
            assert class_id == int(exp_class)
            assert isinstance(class_id, int)

    filtered = get_detections(postprocessed, conf_thres=0.8)
    for image_dets, (_bboxes, scores, _class_ids) in zip(filtered, postprocessed):
        assert len(image_dets) == int(np.sum(scores >= 0.8))
        assert all(score >= 0.8 for _bbox, score, _class_id in image_dets)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_postprocess_classifications(batch_size) -> None:
    """Softmax probabilities sum to 1, rank the planted classes first, and match per-image calls."""
    outputs = _classification_output(batch_size)
    results = postprocess_classifications([o.copy() for o in outputs])
    assert len(results) == batch_size
    for b, (probs,) in enumerate(results):
        assert probs.shape == (1, 1000)
        assert np.sum(probs) == pytest.approx(1.0, rel=1e-5)
        assert np.argmax(probs) == b
        single = postprocess_classifications([o[b : b + 1].copy() for o in outputs])[0][0]
        np.testing.assert_array_equal(probs, single)


@pytest.mark.parametrize(
    "top_k",
    [
        pytest.param(1, id="top-1"),
        pytest.param(3, id="top-3"),
        pytest.param(10, id="top-10"),
    ],
)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_get_classifications(batch_size, top_k) -> None:
    """get_classifications() returns top_k (class_id, confidence) tuples in descending order."""
    postprocessed = postprocess_classifications(_classification_output(batch_size))
    classifications = get_classifications(postprocessed, top_k=top_k)
    assert len(classifications) == batch_size
    for b, image_classes in enumerate(classifications):
        assert len(image_classes) == top_k
        assert [class_id for class_id, _confidence in image_classes][:2] == [b, b + 1][:top_k]
        confidences = [confidence for _class_id, confidence in image_classes]
        assert confidences == sorted(confidences, reverse=True)
        for class_id, confidence in image_classes:
            assert isinstance(class_id, int)
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0
