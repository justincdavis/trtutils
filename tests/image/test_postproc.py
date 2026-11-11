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
    postprocess_depth,
    postprocess_detr,
    postprocess_detr_lbs,
    postprocess_efficient_nms,
    postprocess_hand_interactions,
    postprocess_rfdetr,
    postprocess_rtdetrv3,
    postprocess_text_detection,
    postprocess_text_recognition,
    postprocess_yolo_obb,
    postprocess_yolo_pose,
    postprocess_yolo_seg,
    postprocess_yolov10,
)
from trtutils.image.postprocessors._cpu_detection import _get_detections_core

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
    """RF-DETR outputs: dets (B, 300, 4) normalized cxcywh and logits (B, 300, 80)."""
    dets = np.zeros((batch_size, 300, 4), dtype=np.float32)
    logits = np.full((batch_size, 300, 80), -10.0, dtype=np.float32)
    boxes = _boxes(batch_size)
    dets[:, :NUM_DETS, 0] = (boxes[..., 0] + boxes[..., 2]) / 2 / INPUT_SIZE[0]
    dets[:, :NUM_DETS, 1] = (boxes[..., 1] + boxes[..., 3]) / 2 / INPUT_SIZE[1]
    dets[:, :NUM_DETS, 2] = (boxes[..., 2] - boxes[..., 0]) / INPUT_SIZE[0]
    dets[:, :NUM_DETS, 3] = (boxes[..., 3] - boxes[..., 1]) / INPUT_SIZE[1]
    for b in range(batch_size):
        for i in range(NUM_DETS):
            logits[b, i, i] = 3.0 - i * 0.5
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

        # detr models receive the original image size and already emit original coordinates,
        # which the scaled case clips to its 310 px image
        expected_remap = (bboxes - 10.0) / 2.0 if remaps else np.minimum(bboxes, 310.0)
        np.testing.assert_allclose(remapped[i][0], expected_remap, atol=1e-3)

        assert 0 < len(filtered[i][1]) < NUM_DETS
        assert np.all(filtered[i][1] >= 0.8)


@pytest.mark.parametrize(
    ("postprocess", "make_output"),
    [
        pytest.param(postprocess_yolov10, _yolov10_output, id="yolov10"),
        pytest.param(postprocess_efficient_nms, _efficient_nms_output, id="efficient-nms"),
        pytest.param(postprocess_rfdetr, _rfdetr_output, id="rfdetr"),
        pytest.param(postprocess_detr, _detr_output, id="detr"),
    ],
)
def test_postprocess_detections_clipped_to_image(postprocess, make_output) -> None:
    """Boxes overshooting the original image are clipped to [0, width] x [0, height]."""
    # 320x240 image letterboxed into 640x640: ratio 2, padding (0, 80), so the image is
    # (640 - 0) / 2 = 320 wide and (640 - 160) / 2 = 240 tall
    outputs = make_output(1)
    outputs_big = [o.copy() for o in outputs]
    if postprocess is postprocess_rfdetr:
        outputs_big[0][:, :NUM_DETS] = 1.0  # boxes reaching past the input corner
    elif postprocess is postprocess_efficient_nms:
        outputs_big[1][:, :NUM_DETS, 2:] = 640.0
    elif postprocess is postprocess_yolov10:
        outputs_big[0][:, :NUM_DETS, 2:4] = 640.0
    else:
        outputs_big[2][:, :NUM_DETS, 2:] = [999.0, 999.0]

    bboxes = postprocess(
        outputs_big, [(2.0, 2.0)], [(0.0, 80.0)], conf_thres=0.1, input_size=INPUT_SIZE
    )[0][0]

    assert bboxes.shape == (NUM_DETS, 4)
    assert bboxes.min() >= 0
    assert bboxes[:, 0::2].max() <= 320
    assert bboxes[:, 1::2].max() <= 240
    assert bboxes[:, 2].max() == 320
    assert bboxes[:, 3].max() == 240


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


def _rtdetrv3_output(batch_size: int) -> list[np.ndarray]:
    """RT-DETRv3 outputs: combined (B * NUM_DETS, 6) rows of class, score, x1, y1, x2, y2 and counts (B,)."""
    dets = np.concatenate(
        [
            np.stack([_classes(batch_size)[b], _scores(batch_size)[b], *_boxes(batch_size)[b].T], 1)
            for b in range(batch_size)
        ]
    )
    return [dets.astype(np.float32), np.full(batch_size, NUM_DETS, dtype=np.int32)]


def _yolo_head_output(batch_size: int, channels: int) -> np.ndarray:
    """(B, channels, 300) cxcywh + class scores + extras with sizeable boxes."""
    rng = np.random.default_rng(0)
    out = rng.random((batch_size, channels, 300), dtype=np.float32)
    out[:, :2] *= 600
    out[:, 2:4] = out[:, 2:4] * 100 + 20
    return out


def _seg_output(batch_size: int) -> list[np.ndarray]:
    """YOLO-seg outputs: (B, 4 + 3 classes + 8 coeffs, 300) and prototypes (B, 8, 16, 16)."""
    protos = np.random.default_rng(1).standard_normal((batch_size, 8, 16, 16), dtype=np.float32)
    return [_yolo_head_output(batch_size, 15), protos]


def _hand_output(batch_size: int) -> list[np.ndarray]:
    """Hand interaction outputs: boxes (B, K, 4), scores, labels, pair_probs (B, K, K, 2), side."""
    k = NUM_DETS
    return [
        _boxes(batch_size),
        _scores(batch_size),
        (np.arange(k) % 2).astype(np.int64)[None].repeat(batch_size, 0),
        np.full((batch_size, k, k, 2), 0.6, dtype=np.float32),
        (np.arange(k) % 2).astype(np.int64)[None].repeat(batch_size, 0),
    ]


def _prob_map_output(batch_size: int) -> list[np.ndarray]:
    """(B, 1, 64, 128) probability maps with one text-like block."""
    out = np.zeros((batch_size, 1, 64, 128), dtype=np.float32)
    out[:, :, 20:40, 30:100] = 0.9
    return [out]


def _ratios(batch_size: int) -> tuple[list, list]:
    """Non-trivial letterbox ratios and padding per image."""
    return [(0.5, 0.5)] * batch_size, [(8.0, 16.0)] * batch_size


_SIZED = {"input_size": INPUT_SIZE}


@pytest.mark.parametrize(
    ("postprocess", "make_output", "needs_geometry", "has_conf", "kwargs"),
    [
        pytest.param(postprocess_yolov10, _yolov10_output, True, True, _SIZED, id="yolov10"),
        pytest.param(
            postprocess_efficient_nms, _efficient_nms_output, True, True, _SIZED, id="efficient-nms"
        ),
        pytest.param(postprocess_rfdetr, _rfdetr_output, True, True, _SIZED, id="rfdetr"),
        pytest.param(postprocess_detr, _detr_output, True, True, _SIZED, id="detr"),
        pytest.param(
            postprocess_detr_lbs,
            lambda b: [_detr_output(b)[i] for i in (1, 2, 0)],
            True,
            True,
            _SIZED,
            id="detr-lbs",
        ),
        pytest.param(postprocess_rtdetrv3, _rtdetrv3_output, True, True, _SIZED, id="rtdetrv3"),
        pytest.param(
            postprocess_yolo_pose,
            lambda b: [_yolo_head_output(b, 5 + 17 * 3)],
            True,
            True,
            _SIZED,
            id="yolo-pose",
        ),
        pytest.param(
            postprocess_yolo_obb,
            lambda b: [_yolo_head_output(b, 4 + 3 + 1)],
            True,
            True,
            {},
            id="yolo-obb",
        ),
        pytest.param(postprocess_yolo_seg, _seg_output, True, True, _SIZED, id="yolo-seg"),
        pytest.param(
            postprocess_hand_interactions, _hand_output, True, True, _SIZED, id="hand-interaction"
        ),
        pytest.param(
            postprocess_text_detection, _prob_map_output, True, False, {}, id="text-detection"
        ),
        pytest.param(
            postprocess_text_recognition,
            lambda b: [np.random.default_rng(2).random((b, 20, 30), dtype=np.float32)],
            False,
            False,
            {},
            id="text-recognition",
        ),
        pytest.param(
            postprocess_classifications,
            _classification_output,
            False,
            False,
            {},
            id="classifier",
        ),
        pytest.param(
            postprocess_depth,
            lambda b: [np.random.default_rng(3).random((b, 1, 16, 16), dtype=np.float32) + 1],
            False,
            False,
            {},
            id="depth",
        ),
    ],
)
@pytest.mark.parametrize(
    "conf_thres", [pytest.param(None, id="conf-none"), pytest.param(0.1, id="conf-set")]
)
def test_postprocess_leaves_raw_outputs_untouched(
    postprocess, make_output, needs_geometry, has_conf, kwargs, conf_thres
) -> None:
    """Postprocessing must not modify the raw engine outputs, or a second call double-transforms."""
    outputs = make_output(2)
    originals = [o.copy() for o in outputs]
    args = _ratios(2) if needs_geometry else ()
    if conf_thres is not None:
        if not has_conf:
            pytest.skip("no confidence threshold")
        kwargs = {**kwargs, "conf_thres": conf_thres}

    postprocess(outputs, *args, **kwargs)

    for out, original in zip(outputs, originals):
        np.testing.assert_array_equal(out, original)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((2, 16), id="2d"),
        pytest.param((1, 16, 16), id="3d"),
        pytest.param((2, 16, 16), id="3d-batch"),
    ],
)
def test_postprocess_depth_shape_and_dtype(shape) -> None:
    """Depth maps keep each image's shape and dtype and are normalized to [0, 1]."""
    out = np.random.default_rng(4).random(shape, dtype=np.float32) + 1
    results = postprocess_depth([out])
    assert len(results) == shape[0]
    for i, result in enumerate(results):
        image = out[i : i + 1]
        expected = (image - image.min()) / (image.max() - image.min() + 1e-8)
        np.testing.assert_array_equal(result[0], expected)
        assert result[0].shape == (1, *shape[1:])
        assert result[0].dtype == np.float32
        assert result[0].min() == 0
        assert result[0].max() == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize("conf_thres", [None, 0.0, 0.5, 1.1])
def test_get_detections_core_matches_scalar_loop(conf_thres) -> None:
    """The vectorized conversion equals per-element int()/float() casts, negatives included."""
    rng = np.random.default_rng(0)
    bboxes = rng.uniform(-50.0, 700.0, (64, 4)).astype(np.float32)
    scores = rng.uniform(0.0, 1.0, 64).astype(np.float32)
    scores[:4] = 0.5  # exactly at the threshold
    class_ids = rng.integers(0, 80, 64)
    thres = 0.0 if conf_thres is None else conf_thres
    expected = [
        ((int(b[0]), int(b[1]), int(b[2]), int(b[3])), float(s), int(c))
        for b, s, c in zip(bboxes, scores, class_ids)
        if s >= thres
    ]
    assert _get_detections_core([bboxes, scores, class_ids], conf_thres) == expected


@pytest.mark.cpu
def test_postprocess_depth_constant_fp16_is_finite() -> None:
    """A constant fp16 depth map normalizes to zeros, not NaN (1e-8 underflows in fp16)."""
    out = np.full((1, 1, 8, 8), 3.0, dtype=np.float16)
    depth = postprocess_depth([out])[0][0]
    assert np.isfinite(depth).all()
    assert (depth == 0).all()
