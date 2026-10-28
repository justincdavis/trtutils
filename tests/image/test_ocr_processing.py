# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/{pre,post}processors/_ocr.py -- PP-OCR pre/postprocessing."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from tests.conftest import DATA_DIR
from trtutils._flags import FLAGS
from trtutils.compat._libs import cv2
from trtutils.image._geometry import _fill_poly
from trtutils.image.postprocessors import (
    get_text_boxes,
    get_texts,
    postprocess_text_detection,
    postprocess_text_recognition,
    sort_text_boxes,
)
from trtutils.image.preprocessors import (
    crop_text_region,
    preprocess_text_detection,
    preprocess_text_recognition,
    resize_text_detection,
)

pytestmark = pytest.mark.cpu

# expected boxes of the reference pp-ocr db postprocess on _prob_map() at (480, 640)
REFERENCE_BOXES = np.array(
    [
        [[32, 18], [328, 18], [328, 162], [32, 162]],
        [[338, 33], [622, 33], [622, 147], [338, 147]],
        [[80, 314], [541, 193], [558, 343], [97, 464]],
    ],
    np.float32,
)


def _box_points(center: tuple[float, float], size: tuple[float, float], angle: float) -> np.ndarray:
    """Corners of a rotated rect as int32, like cv2.boxPoints(...).astype(np.int32)."""
    (cx, cy), (w, h) = center, size
    b, a = np.cos(np.radians(angle)) * 0.5, np.sin(np.radians(angle)) * 0.5
    p0 = (cx - a * h - b * w, cy + b * h - a * w)
    p1 = (cx + a * h - b * w, cy - b * h - a * w)
    pts = np.array([p0, p1, (2 * cx - p0[0], 2 * cy - p0[1]), (2 * cx - p1[0], 2 * cy - p1[1])])
    return pts.astype(np.float32).astype(np.int32)


def _fill(pred: np.ndarray, pts: np.ndarray, value: float) -> None:
    mask = np.zeros(pred.shape, dtype=np.uint8)
    _fill_poly(mask, np.ascontiguousarray(pts, dtype=np.int64))
    pred[mask > 0] = value


def _prob_map() -> np.ndarray:
    m = np.zeros((160, 320), np.float32)
    m[20:40, 30:150] = 0.9
    m[22:38, 180:300] = 0.8
    _fill(m, _box_points((160, 110), (200, 18), -10), 0.95)
    m[140:142, 10:20] = 0.9  # too thin to be text
    return m


@pytest.mark.parametrize(
    ("shape", "limit", "limit_type", "max_shape", "expected"),
    [
        pytest.param((100, 400), 736, "min", None, (736, 2944), id="min-upscale"),
        pytest.param((1080, 1920), 736, "min", None, (1088, 1920), id="min-noop-round32"),
        pytest.param((1080, 1920), 960, "max", None, (544, 960), id="max-downscale"),
        pytest.param((300, 500), 960, "max", None, (288, 512), id="max-noop-round32"),
        pytest.param((15, 15), 960, "max", None, (32, 32), id="tiny-floor32"),
        pytest.param((30, 4000), 960, "max", None, (32, 960), id="thin-strip-floor32"),
        pytest.param((100, 400), 736, "min", (2048, 2048), (512, 2048), id="fit-max-shape"),
        pytest.param((100, 400), 736, "min", (320, 320), (64, 320), id="fit-small-max-shape"),
        pytest.param((32, 32), 960, "avg", None, None, id="invalid-limit-type"),
    ],
)
def test_resize_text_detection(shape, limit, limit_type, max_shape, expected) -> None:
    """Images resize by the limit rule, round to multiples of 32 and fit the engine's max shape."""
    ctx = (
        pytest.raises(ValueError, match="limit_type")
        if expected is None
        else contextlib.nullcontext()
    )
    with ctx:
        out = resize_text_detection(np.zeros((*shape, 3), np.uint8), limit, limit_type, max_shape)
        assert out.shape[:2] == expected


@pytest.mark.parametrize(
    ("kwargs", "low", "high"),
    [
        pytest.param({}, -0.485 / 0.229, (1 - 0.406) / 0.225, id="imagenet-default"),
        pytest.param({"mean": (0.5,) * 3, "std": (0.5,) * 3}, -1.0, 1.0, id="half-mean-std"),
        pytest.param(
            {"mean": None, "std": None, "input_range": (0.0, 255.0)}, 0.0, 255.0, id="input-range"
        ),
    ],
)
def test_preprocess_text_detection(kwargs, low, high) -> None:
    """A mixed-size batch is normalized per channel and zero-padded to its largest image."""
    small = np.zeros((64, 96, 3), np.uint8)
    small[..., 2] = 255
    large = np.zeros((128, 64, 3), np.uint8)
    tensor, ratios, padding = preprocess_text_detection([small, large], **kwargs)
    assert tensor.shape == (2, 3, 128, 96)
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(tensor[0, 0, :64], low, rtol=1e-6)
    np.testing.assert_allclose(tensor[0, 2, :64], high, rtol=1e-6)
    np.testing.assert_array_equal(tensor[0, :, 64:], 0.0)
    np.testing.assert_array_equal(tensor[1, :, :, 64:], 0.0)
    assert ratios == [(1.0, 1.0), (1.0, 1.0)]
    assert padding == [(0.0, 64.0), (32.0, 0.0)]


@pytest.mark.parametrize(
    ("shapes", "max_width", "expected_width", "expected_resized"),
    [
        # ratio 2 -> 96 px wide at height 48, ratio 50 -> 2400 px
        pytest.param([(24, 48), (10, 500)], None, 2400, [96, 2400], id="pads-to-widest"),
        pytest.param([(48, 48)], None, 320, [48], id="min-width"),
        pytest.param([(10, 2000)], 1024, 1024, [1024], id="squeeze-to-max-width"),
    ],
)
def test_preprocess_text_recognition(shapes, max_width, expected_width, expected_resized) -> None:
    """Lines resize to the model height and right-pad to a width within [min_width, max_width]."""
    lines = [np.full((*shape, 3), 255, np.uint8) for shape in shapes]
    tensor, ratios, padding = preprocess_text_recognition(lines, max_width=max_width)
    assert tensor.shape == (len(lines), 3, 48, expected_width)
    for i, (resized_w, (h, w)) in enumerate(zip(expected_resized, shapes)):
        np.testing.assert_allclose(tensor[i, :, :, :resized_w], 1.0)
        np.testing.assert_array_equal(tensor[i, :, :, resized_w:], 0.0)
        assert ratios[i] == (resized_w / w, 48 / h)
        assert padding[i] == (expected_width - resized_w, 0.0)


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        pytest.param([[20, 10], [120, 10], [120, 40], [20, 40]], (30, 100, 3), id="axis-aligned"),
        pytest.param([[10, 10], [30, 10], [30, 90], [10, 90]], (20, 80, 3), id="tall-rotated"),
    ],
)
def test_crop_text_region(box, expected) -> None:
    """Axis-aligned boxes crop exactly and tall crops are rotated to horizontal."""
    img = np.arange(100 * 200 * 3, dtype=np.uint32).reshape(100, 200, 3).astype(np.uint8)
    assert crop_text_region(img, np.array(box, np.float32)).shape == expected


@pytest.mark.parametrize(
    ("prob_map", "pad", "expected"),
    [
        pytest.param(_prob_map(), (0, 0), REFERENCE_BOXES, id="reference"),
        pytest.param(_prob_map(), (64, 32), REFERENCE_BOXES, id="batch-padded"),
        pytest.param(np.zeros((160, 320), np.float32), (0, 0), np.zeros((0, 4, 2)), id="empty"),
    ],
)
def test_postprocess_text_detection(prob_map, pad, expected) -> None:
    """Postprocess -> get_text_boxes matches the reference PP-OCR DB postprocess, ignoring padding."""
    pad_x, pad_y = pad
    padded = np.pad(prob_map, ((0, pad_y), (0, pad_x)))[None, None]
    ratios = [(320 / 640, 160 / 480)]
    outputs = postprocess_text_detection(
        [padded], ratios, [(float(pad_x), float(pad_y))], 0.3, 0.5, 1000, 1.6, use_dilation=True
    )
    boxes, scores = outputs[0]
    np.testing.assert_array_equal(boxes, expected.astype(np.float32))
    assert scores.shape == (len(expected),)
    assert np.all((scores > 0.5) & (scores <= 1.0))
    text_boxes = get_text_boxes(outputs)[0]
    assert [float(s) for s in scores] == [score for _box, score in text_boxes]
    for (box, _score), ref in zip(text_boxes, boxes):
        np.testing.assert_array_equal(box, ref)


def test_sort_text_boxes_reading_order() -> None:
    """Boxes on the same line (within 10 px) sort left to right, lines top to bottom."""

    def box(x: float, y: float) -> list[list[float]]:
        return [[x, y], [x + 10, y], [x + 10, y + 5], [x, y + 5]]

    boxes = np.array([box(50, 102), box(0, 100), box(0, 0), box(80, 3)], np.float32)
    sorted_boxes, sorted_scores = sort_text_boxes(boxes, np.arange(4, dtype=np.float32))
    np.testing.assert_array_equal(sorted_scores, [2, 3, 1, 0])
    assert sorted_boxes.shape == (4, 4, 2)


def test_postprocess_text_recognition_ctc() -> None:
    """Postprocess -> get_texts collapses repeats, drops blanks and scores the mean kept probability."""
    chars = ["blank", "a", "b", " "]
    seq = [1, 1, 0, 1, 2, 2, 0, 3]  # -> "aab "
    probs = np.full((1, len(seq), len(chars)), 0.0, np.float32)
    for t, c in enumerate(seq):
        probs[0, t, c] = 0.5 + 0.05 * t
    empty = np.zeros((1, len(seq), len(chars)), np.float32)
    empty[0, :, 0] = 1.0
    outputs = postprocess_text_recognition([np.concatenate([probs, empty])])
    np.testing.assert_array_equal(outputs[0][0], seq)
    res = get_texts(outputs, chars)
    assert res[0][0] == "aab "
    assert res[0][1] == pytest.approx(np.mean([0.5, 0.65, 0.7, 0.85]))
    assert res[1] == ("", 0.0)


_needs_cv2 = pytest.mark.skipif(not FLAGS.FOUND_CV2, reason="the reference is OpenCV")
# full-photo crops in plain python take minutes, so they compare the compiled kernel only
_needs_numba = pytest.mark.skipif(not FLAGS.FOUND_NUMBA, reason="needs numba for the jit kernels")
_PHOTOS = [pytest.param(n, id=n) for n in ("horse.jpg", "street.jpg", "people.jpeg", "cars.jpeg")]


def _text_map(seed: int, shape: tuple[int, int] = (736, 960)) -> np.ndarray:
    """Probability map of many axis-aligned and rotated text-line blobs plus speckle."""
    rng = np.random.default_rng(seed)
    pred = rng.uniform(0.0, 0.25, shape).astype(np.float32)
    for _ in range(40):
        cx, cy = rng.uniform(60, shape[1] - 60), rng.uniform(20, shape[0] - 20)
        w, h = rng.uniform(20, 220), rng.uniform(3, 30)
        angle = rng.choice([0, rng.uniform(-30, 30), 90])
        _fill(pred, _box_points((cx, cy), (w, h), angle), float(rng.uniform(0.4, 0.98)))
    if seed % 2:
        pad = np.pad(pred, 1, mode="edge")
        pred = sum(pad[i : i + shape[0], j : j + shape[1]] for i in range(3) for j in range(3)) / 9
    return pred.astype(np.float32)


def _detect(pred: np.ndarray, thresh: float, box_thresh: float, unclip: float, dilate: bool):
    return postprocess_text_detection(
        [pred[None, None]], [(0.5, 0.5)], [(0.0, 0.0)], thresh, box_thresh, 1000, unclip,
        use_dilation=dilate,
    )[0]  # fmt: skip


@_needs_cv2
@pytest.mark.parametrize(
    "dilate", [pytest.param(True, id="dilate"), pytest.param(False, id="plain")]
)
@pytest.mark.parametrize(
    ("thresh", "box_thresh", "unclip"),
    [
        pytest.param(0.3, 0.6, 1.5, id="defaults"),
        pytest.param(0.5, 0.3, 2.0, id="high-thresh-wide-unclip"),
        pytest.param(0.3, 0.65, 1.0, id="strict-box"),
        pytest.param(0.35, 0.5, 0.0, id="no-unclip"),
    ],
)
@pytest.mark.parametrize("seed", [pytest.param(s, id=f"seed{s}") for s in range(4)])
def test_text_detection_without_cv2_identical(
    monkeypatch: pytest.MonkeyPatch, seed, thresh, box_thresh, unclip, dilate
) -> None:
    """Boxes and scores are identical whether cv2 or the numba/numpy geometry runs."""
    pred = _text_map(seed)
    monkeypatch.setattr(FLAGS, "FOUND_CV2", True)
    want_boxes, want_scores = _detect(pred, thresh, box_thresh, unclip, dilate)
    monkeypatch.setattr(FLAGS, "FOUND_CV2", False)
    boxes, scores = _detect(pred, thresh, box_thresh, unclip, dilate)
    assert len(want_boxes) > 0
    if int(cv2.__version__.split(".")[0]) < 5:
        np.testing.assert_array_equal(boxes, want_boxes)
        np.testing.assert_array_equal(scores, want_scores)
    else:
        # opencv 5 rewrote minAreaRect: a rotated blob can move a box corner by a pixel or two
        assert boxes.shape == want_boxes.shape
        np.testing.assert_allclose(boxes, want_boxes, atol=2)
        np.testing.assert_allclose(scores, want_scores, atol=0.02)


@_needs_cv2
@pytest.mark.parametrize("name", _PHOTOS)
def test_text_preprocess_without_cv2_identical(monkeypatch: pytest.MonkeyPatch, name) -> None:
    """Detection and recognition tensors are identical with and without cv2."""
    image = cv2.imread(str(DATA_DIR / name))
    crop = image[40:100, 20:300]
    monkeypatch.setattr(FLAGS, "FOUND_CV2", True)
    want_det = preprocess_text_detection([image, crop], max_shape=(1024, 1024))
    want_rec = preprocess_text_recognition([crop, image[:30, :50]])
    monkeypatch.setattr(FLAGS, "FOUND_CV2", False)
    got_det = preprocess_text_detection([image, crop], max_shape=(1024, 1024))
    got_rec = preprocess_text_recognition([crop, image[:30, :50]])
    np.testing.assert_array_equal(got_det[0], want_det[0])
    assert got_det[1:] == want_det[1:]
    np.testing.assert_array_equal(got_rec[0], want_rec[0])
    assert got_rec[1:] == want_rec[1:]


@_needs_cv2
@_needs_numba
@pytest.mark.parametrize("name", _PHOTOS)
def test_crop_text_region_without_cv2_close(monkeypatch: pytest.MonkeyPatch, name) -> None:
    """Crops match cv2 exactly on OpenCV 4.x and within a few levels on 5.x."""
    image = cv2.imread(str(DATA_DIR / name))
    height, width = image.shape[:2]
    rng = np.random.default_rng(3)
    for _ in range(25):
        cx, cy = rng.uniform(0.25, 0.75) * width, rng.uniform(0.25, 0.75) * height
        bw, bh = rng.uniform(0.04, 0.2) * width, rng.uniform(0.01, 0.06) * height
        if rng.random() < 0.3:
            bw, bh = bh, bw
        ang = rng.uniform(-0.3, 0.3)
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        box = np.array([[-bw, -bh], [bw, -bh], [bw, bh], [-bw, bh]]) @ rot.T + [cx, cy]
        monkeypatch.setattr(FLAGS, "FOUND_CV2", True)
        want = crop_text_region(image, box.astype(np.float32))
        monkeypatch.setattr(FLAGS, "FOUND_CV2", False)
        got = crop_text_region(image, box.astype(np.float32))
        assert got.shape == want.shape
        if int(cv2.__version__.split(".")[0]) < 5:
            np.testing.assert_array_equal(got, want)
        else:
            diff = np.abs(got.astype(int) - want.astype(int))
            assert diff.mean() < 0.5
            assert diff.max() <= 8
