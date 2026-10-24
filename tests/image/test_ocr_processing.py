# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the OCR (PP-OCR) pre/postprocessing in src/trtutils/image/{pre,post}processors/_ocr.py."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from trtutils.image.postprocessors import (
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


@pytest.mark.parametrize(
    ("shape", "limit_type", "expected"),
    [
        pytest.param((100, 400), "min", (736, 2944), id="min-upscale"),
        pytest.param((1080, 1920), "min", (1088, 1920), id="min-noop-round32"),
        pytest.param((1080, 1920), "max", (544, 960), id="max-downscale"),
        pytest.param((300, 500), "max", (288, 512), id="max-noop-round32"),
    ],
)
def test_resize_text_detection(shape, limit_type, expected) -> None:
    """Images resize by the limit rule, then round each side to a multiple of 32."""
    limit = 736 if limit_type == "min" else 960
    out = resize_text_detection(np.zeros((*shape, 3), np.uint8), limit, limit_type)
    assert out.shape[:2] == expected


def test_resize_text_detection_fits_max_shape() -> None:
    """A resize larger than the engine's max shape is shrunk to fit, keeping multiples of 32."""
    out = resize_text_detection(np.zeros((100, 400, 3), np.uint8), max_shape=(2048, 2048))
    h, w = out.shape[:2]
    assert h <= 2048
    assert w <= 2048
    assert h % 32 == 0
    assert w % 32 == 0
    assert w / h == pytest.approx(4, rel=0.05)


def test_resize_text_detection_invalid_limit_type() -> None:
    """An unknown limit_type raises."""
    with pytest.raises(ValueError, match="limit_type"):
        resize_text_detection(np.zeros((32, 32, 3), np.uint8), limit_type="avg")


def test_preprocess_text_detection() -> None:
    """Output is a contiguous (1, 3, H, W) float32 tensor normalized to [-1, 1]."""
    img = np.zeros((64, 96, 3), np.uint8)
    img[..., 2] = 255
    x = preprocess_text_detection(img, limit_side_len=64)
    assert x.shape == (1, 3, 64, 96)
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(x[0, 0], -1.0)
    np.testing.assert_allclose(x[0, 2], 1.0)


def test_preprocess_text_recognition_pads_to_widest() -> None:
    """Lines resize to the model height, keep aspect ratio and right-pad to the widest one."""
    narrow = np.full((24, 48, 3), 255, np.uint8)  # ratio 2 -> 96 px wide at height 48
    wide = np.full((10, 500, 3), 255, np.uint8)  # ratio 50 -> 2400 px wide
    x = preprocess_text_recognition([narrow, wide])
    assert x.shape == (2, 3, 48, 2400)
    np.testing.assert_allclose(x[0, :, :, :96], 1.0)
    np.testing.assert_allclose(x[0, :, :, 96:], 0.0)
    np.testing.assert_allclose(x[1], 1.0)


def test_preprocess_text_recognition_min_and_max_width() -> None:
    """The batch is at least min_width wide and wide lines are squeezed to max_width."""
    line = np.zeros((48, 48, 3), np.uint8)
    assert preprocess_text_recognition([line]).shape == (1, 3, 48, 320)
    long_line = np.zeros((10, 2000, 3), np.uint8)
    assert preprocess_text_recognition([long_line], max_width=1024).shape == (1, 3, 48, 1024)


def test_crop_text_region() -> None:
    """Axis-aligned boxes crop exactly; tall crops are rotated to horizontal."""
    img = np.arange(100 * 200 * 3, dtype=np.uint32).reshape(100, 200, 3).astype(np.uint8)
    box = np.array([[20, 10], [120, 10], [120, 40], [20, 40]], np.float32)
    assert crop_text_region(img, box).shape == (30, 100, 3)
    tall = np.array([[10, 10], [30, 10], [30, 90], [10, 90]], np.float32)
    assert crop_text_region(img, tall).shape == (20, 80, 3)


def _prob_map() -> np.ndarray:
    m = np.zeros((160, 320), np.float32)
    m[20:40, 30:150] = 0.9
    m[22:38, 180:300] = 0.8
    pts = cv2.boxPoints(((160, 110), (200, 18), -10)).astype(np.int32)
    cv2.fillPoly(m, [pts], 0.95)
    m[140:142, 10:20] = 0.9  # too thin to be text
    return m


def test_postprocess_text_detection_matches_reference() -> None:
    """Boxes match the reference PP-OCR DB postprocess (incl. unclip rounding) exactly."""
    boxes, scores = postprocess_text_detection(_prob_map()[None, None], (480, 640))
    expected = [
        [[32, 18], [328, 18], [328, 162], [32, 162]],
        [[338, 33], [622, 33], [622, 147], [338, 147]],
        [[80, 314], [541, 193], [558, 343], [97, 464]],
    ]
    np.testing.assert_array_equal(boxes, np.array(expected, np.float32))
    assert scores.shape == (3,)
    assert np.all((scores > 0.5) & (scores <= 1.0))


def test_postprocess_text_detection_empty() -> None:
    """An empty probability map yields no boxes."""
    boxes, scores = postprocess_text_detection(np.zeros((1, 1, 64, 64), np.float32), (64, 64))
    assert boxes.shape == (0, 4, 2)
    assert scores.shape == (0,)


def test_sort_text_boxes_reading_order() -> None:
    """Boxes on the same line (within 10 px) sort left to right, lines top to bottom."""

    def box(x: float, y: float) -> list[list[float]]:
        return [[x, y], [x + 10, y], [x + 10, y + 5], [x, y + 5]]

    boxes = np.array([box(50, 102), box(0, 100), box(0, 0), box(80, 3)], np.float32)
    sorted_boxes, sorted_scores = sort_text_boxes(boxes, np.arange(4, dtype=np.float32))
    np.testing.assert_array_equal(sorted_scores, [2, 3, 1, 0])
    assert sorted_boxes.shape == (4, 4, 2)


def test_postprocess_text_recognition_ctc() -> None:
    """Greedy CTC: collapse repeats, drop blanks, score is the mean kept probability."""
    chars = ["blank", "a", "b", " "]
    seq = [1, 1, 0, 1, 2, 2, 0, 3]  # -> "aab "
    probs = np.full((1, len(seq), len(chars)), 0.0, np.float32)
    for t, c in enumerate(seq):
        probs[0, t, c] = 0.5 + 0.05 * t
    empty = np.zeros((1, len(seq), len(chars)), np.float32)
    empty[0, :, 0] = 1.0
    res = postprocess_text_recognition(np.concatenate([probs, empty]), chars)
    assert res[0][0] == "aab "
    assert res[0][1] == pytest.approx(np.mean([0.5, 0.65, 0.7, 0.85]))
    assert res[1] == ("", 0.0)
