# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_ops.py -- numpy image ops against references built from plain cv2 and numpy."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils._flags import FLAGS
from trtutils.compat._libs import cv2
from trtutils.image import _ops

pytestmark = pytest.mark.cpu

# references built from opencv; the numpy-only tests also run on free-threaded installs
_needs_cv2 = pytest.mark.skipif(not FLAGS.FOUND_CV2, reason="the reference is OpenCV")

_BACKENDS = [
    pytest.param(True, id="cv2", marks=_needs_cv2),
    pytest.param(False, id="numpy"),
]

_RESIZE_CASES = [
    pytest.param((360, 640), (640, 640), id="upscale-height"),
    pytest.param((480, 640), (1280, 960), id="upscale-2x"),
    pytest.param((375, 500), (640, 480), id="upscale-non-integer"),
    pytest.param((1080, 1920), (640, 360), id="downscale-3x"),
    pytest.param((720, 1280), (640, 640), id="downscale-wide"),
    pytest.param((333, 517), (211, 97), id="downscale-odd"),
    pytest.param((7, 9), (640, 640), id="tiny-7x9"),
    pytest.param((9, 7), (5, 3), id="tiny-downscale"),
    pytest.param((100, 100), (37, 300), id="mixed-axes"),
    pytest.param((64, 48), (64, 48), id="identity"),
    pytest.param((64, 48), (128, 48), id="width-only"),
    pytest.param((64, 48), (64, 96), id="height-only"),
    pytest.param((1, 5), (9, 9), id="single-row"),
]


@pytest.fixture
def use_cv2(request, monkeypatch) -> bool:
    """Set FLAGS.FOUND_CV2 to the param."""
    monkeypatch.setattr(FLAGS, "FOUND_CV2", request.param)
    return request.param


def _random_image(shape: tuple[int, ...], dtype: type, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dtype == np.uint8:
        return rng.integers(0, 256, shape, dtype=np.uint8)
    return rng.random(shape).astype(dtype)


def _cv2_letterbox(image: np.ndarray, new_shape: tuple[int, int], color: tuple[int, int, int]):
    h, w = image.shape[:2]
    r = min(new_shape[1] / h, new_shape[0] / w)
    unpad = (round(w * r), round(h * r))
    dw, dh = (new_shape[0] - unpad[0]) / 2, (new_shape[1] - unpad[1]) / 2
    if (w, h) != unpad:
        image = cv2.resize(image, unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, (r, r), (dw, dh)


def _iou(a: tuple, b: tuple) -> float:
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _greedy_nms(boxes: list, iou_threshold: float, agnostic: bool | None) -> list:
    kept: list = []
    for box in sorted(boxes, key=lambda b: b[1], reverse=True):
        clash = any(
            (agnostic or k[2] == box[2]) and _iou(k[0], box[0]) > iou_threshold for k in kept
        )
        if not clash:
            kept.append(box)
    return kept


def _assert_close_u8(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    diff = np.abs(actual.astype(int) - expected.astype(int))
    assert diff.max() <= 1
    assert diff.mean() < 0.05


@_needs_cv2
@pytest.mark.parametrize(("src", "dst"), _RESIZE_CASES)
@pytest.mark.parametrize(
    ("channels", "dtype"),
    [
        pytest.param(3, np.uint8, id="u8-3ch"),
        pytest.param(1, np.uint8, id="u8-1ch"),
        pytest.param(None, np.uint8, id="u8-2d"),
        pytest.param(4, np.uint8, id="u8-4ch"),
        pytest.param(3, np.float32, id="f32-3ch"),
        pytest.param(None, np.float32, id="f32-2d"),
    ],
)
def test_resize_bilinear_matches_cv2(src, dst, channels, dtype) -> None:
    """The numpy bilinear core matches cv2.resize INTER_LINEAR."""
    shape = src if channels is None else (*src, channels)
    image = _random_image(shape, dtype)
    expected = cv2.resize(image, dst, interpolation=cv2.INTER_LINEAR)
    actual = _ops._resize_bilinear(image, dst[0], dst[1])
    if dtype == np.uint8:
        _assert_close_u8(actual, expected)
    else:
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=1e-4)


@_needs_cv2
@pytest.mark.parametrize("seed", [pytest.param(i, id=f"seed-{i}") for i in range(8)])
def test_resize_bilinear_random_shapes(seed) -> None:
    """Random shapes stay within the resize bar."""
    rng = np.random.default_rng(seed)
    h, w = rng.integers(2, 300, 2)
    new_w, new_h = rng.integers(2, 500, 2)
    image = _random_image((h, w, 3), np.uint8, seed)
    expected = cv2.resize(image, (int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR)
    _assert_close_u8(_ops._resize_bilinear(image, int(new_w), int(new_h)), expected)


@_needs_cv2
@pytest.mark.parametrize("use_cv2", _BACKENDS, indirect=True)
@pytest.mark.parametrize(("src", "dst"), _RESIZE_CASES)
def test_resize_linear_dispatch(use_cv2, src, dst) -> None:
    """resize_linear matches cv2.resize on both backends, with (x, y) ratios and no padding."""
    image = _random_image((*src, 3), np.uint8)
    actual, ratios, padding = _ops.resize_linear(image, dst)
    _assert_close_u8(actual, cv2.resize(image, dst, interpolation=cv2.INTER_LINEAR))
    # same (x, y) contract as resize_letterbox and the gpu preprocessors
    assert ratios == (dst[0] / src[1], dst[1] / src[0])
    assert padding == (0.0, 0.0)


@_needs_cv2
@pytest.mark.parametrize("use_cv2", _BACKENDS, indirect=True)
@pytest.mark.parametrize(
    "src",
    [
        pytest.param((360, 640), id="landscape"),
        pytest.param((640, 360), id="portrait"),
        pytest.param((480, 480), id="square"),
        pytest.param((375, 500), id="odd"),
        pytest.param((100, 640), id="wide-strip"),
        pytest.param((7, 9), id="tiny"),
        pytest.param((1080, 1920), id="downscale"),
    ],
)
@pytest.mark.parametrize(
    "new_shape",
    [
        pytest.param((640, 640), id="640x640"),
        pytest.param((512, 384), id="512x384"),
        pytest.param((321, 255), id="odd-target"),
    ],
)
@pytest.mark.parametrize(
    "color",
    [
        pytest.param((114, 114, 114), id="default-color"),
        pytest.param((1, 2, 3), id="custom-color"),
    ],
)
def test_resize_letterbox_matches_cv2(use_cv2, src, new_shape, color) -> None:
    """Letterbox returns the cv2 reference's ratios, padding and pad color; image within the resize bar."""
    image = _random_image((*src, 3), np.uint8)
    expected, e_ratio, e_pad = _cv2_letterbox(image, new_shape, color)
    actual, ratio, pad = _ops.resize_letterbox(image, new_shape=new_shape, color=color)
    assert ratio == e_ratio
    assert pad == e_pad
    _assert_close_u8(actual, expected)
    # padding pixels are exact: wherever the reference holds the color, so do we
    border = (expected == np.array(color, np.uint8)).all(axis=2)
    assert (actual[border] == np.array(color, np.uint8)).all()


def test_resize_letterbox_identity_shape_no_resize() -> None:
    """An image already at the target shape is returned unchanged."""
    image = _random_image((64, 64, 3), np.uint8)
    actual, ratio, pad = _ops.resize_letterbox(image, new_shape=(64, 64))
    np.testing.assert_array_equal(actual, image)
    assert ratio == (1.0, 1.0)
    assert pad == (0.0, 0.0)


def test_resize_letterbox_2d() -> None:
    """A 2D image pads with the first color channel."""
    image = _random_image((20, 40), np.uint8)
    actual, _, _ = _ops.resize_letterbox(image, new_shape=(40, 40), color=(9, 8, 7))
    assert actual.shape == (40, 40)
    assert (actual[:10] == 9).all()
    np.testing.assert_array_equal(actual[10:30], image)


@_needs_cv2
@pytest.mark.parametrize("use_cv2", _BACKENDS, indirect=True)
@pytest.mark.parametrize(
    "src",
    [
        pytest.param((1080, 1920), id="hd"),
        pytest.param((480, 640), id="vga"),
        pytest.param((333, 517), id="odd"),
        pytest.param((100, 100), id="smaller-than-patch"),
    ],
)
@pytest.mark.parametrize(
    "patch_size",
    [
        pytest.param((640, 640), id="640"),
        pytest.param((320, 240), id="320x240"),
    ],
)
@pytest.mark.parametrize(
    ("padding", "overlap"),
    [
        pytest.param(None, None, id="no-overlap"),
        pytest.param(32, None, id="padding-int"),
        pytest.param((16, 48), None, id="padding-tuple"),
        pytest.param(None, 0.25, id="overlap-float"),
        pytest.param(None, (0.1, 0.3), id="overlap-tuple"),
    ],
)
def test_patch_matches_cv2(use_cv2, src, patch_size, padding, overlap) -> None:
    """Patches tile the resized image on an even grid that just covers it, matching cv2.resize."""
    image = _random_image((*src, 3), np.uint8)
    if overlap is not None:
        ox, oy = overlap if isinstance(overlap, tuple) else (overlap, overlap)
        pad = (int(ox * patch_size[0]), int(oy * patch_size[1]))
    else:
        pad = padding if isinstance(padding, tuple) else (padding or 0, padding or 0)
    patches, offsets, (width, height) = _ops.patch(
        image, patch_size, padding=padding, overlap=overlap
    )
    xs = sorted({x for x, _ in offsets})
    ys = sorted({y for _, y in offsets})
    assert offsets == [(x, y) for y in ys for x in xs]
    for starts, dim, size, p, stride in (
        (xs, src[1], width, patch_size[0], patch_size[0] - pad[0]),
        (ys, src[0], height, patch_size[1], patch_size[1] - pad[1]),
    ):
        # evenly strided, the last patch ends on the edge, and one fewer row/column won't cover
        assert starts == list(range(0, size - p + 1, stride))
        assert starts[-1] + p == size
        assert size >= dim
        assert len(starts) == 1 or size - stride < dim
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    for (x, y), actual in zip(offsets, patches):
        _assert_close_u8(actual, resized[y : y + patch_size[1], x : x + patch_size[0]])


@pytest.mark.parametrize(
    ("padding", "overlap"),
    [
        pytest.param(640, None, id="padding-x-too-big"),
        pytest.param((0, 640), None, id="padding-y-too-big"),
        pytest.param(None, 1.0, id="overlap-too-big"),
    ],
)
def test_patch_bad_overlap(padding, overlap) -> None:
    """Overlap not smaller than the patch raises."""
    image = _random_image((800, 800, 3), np.uint8)
    with pytest.raises(ValueError, match="cannot be greater"):
        _ops.patch(image, (640, 640), padding=padding, overlap=overlap)


@pytest.mark.parametrize(
    "value_range",
    [
        pytest.param((0.0, 1.0), id="unit"),
        pytest.param((-1.0, 1.0), id="symmetric"),
        pytest.param((0.0, 255.0), id="identity"),
        pytest.param((0.5, 2.5), id="offset"),
    ],
)
def test_rescale_matches_formula(value_range) -> None:
    """Rescale matches the closed-form formula."""
    image = _random_image((16, 16, 3), np.uint8)
    low, high = value_range
    expected = image.astype(np.float64) / 255.0 * (high - low) + low
    np.testing.assert_allclose(_ops.rescale(image, value_range), expected, atol=1e-12)


@pytest.mark.parametrize(
    "value_range",
    [
        pytest.param((1.0, 1.0), id="equal"),
        pytest.param((1.0, 0.0), id="inverted"),
    ],
)
def test_rescale_bad_range(value_range) -> None:
    """A non-increasing range raises ValueError."""
    with pytest.raises(ValueError, match="Low value"):
        _ops.rescale(np.zeros((2, 2), np.uint8), value_range)


def _random_boxes(n: int, seed: int, classes: int, score_levels: int | None) -> list:
    rng = np.random.default_rng(seed)
    boxes = []
    for _ in range(n):
        x1, y1 = rng.integers(0, 100, 2)
        w, h = rng.integers(0, 60, 2)
        if score_levels is None:
            score = float(rng.random())
        else:
            score = float(rng.integers(0, score_levels)) / score_levels
        boxes.append(
            ((int(x1), int(y1), int(x1 + w), int(y1 + h)), score, int(rng.integers(0, classes)))
        )
    return boxes


@pytest.mark.parametrize("agnostic", [pytest.param(None, id="default"), True, False])
@pytest.mark.parametrize("iou_threshold", [pytest.param(t, id=f"iou-{t}") for t in (0.1, 0.5, 0.9)])
@pytest.mark.parametrize(
    ("n", "classes", "score_levels"),
    [
        pytest.param(0, 3, None, id="empty"),
        pytest.param(1, 3, None, id="single"),
        pytest.param(60, 1, None, id="one-class"),
        pytest.param(120, 5, None, id="multi-class"),
        pytest.param(120, 5, 3, id="score-ties"),
    ],
)
@pytest.mark.parametrize("seed", [pytest.param(i, id=f"seed-{i}") for i in range(3)])
def test_nms_matches_greedy(n, classes, score_levels, seed, iou_threshold, agnostic) -> None:
    """Nms keeps exactly the boxes greedy nms keeps, in the same order."""
    boxes = _random_boxes(n, seed, classes, score_levels)
    expected = _greedy_nms(boxes, iou_threshold, agnostic)
    assert _ops.nms(boxes, iou_threshold, agnostic=agnostic) == expected


def test_nms_suppresses_overlap() -> None:
    """Overlapping same-class boxes drop the lower score; agnostic also drops across classes."""
    boxes = [((0, 0, 10, 10), 0.9, 0), ((1, 1, 10, 10), 0.8, 0), ((1, 1, 10, 10), 0.7, 1)]
    assert [b[1] for b in _ops.nms(boxes)] == [0.9, 0.7]
    assert [b[1] for b in _ops.nms(boxes, agnostic=True)] == [0.9]


@pytest.mark.parametrize("use_cv2", _BACKENDS, indirect=True)
@pytest.mark.parametrize(
    ("src_shape", "scale", "offset", "size"),
    [
        pytest.param((40, 40), (0.25, 0.25), (0.3, 0.3), (120, 120), id="proto-quarter"),
        pytest.param((40, 40), (0.2667, 0.2667), (-0.1, -0.1), (120, 120), id="proto-fifth"),
        pytest.param((40, 40), (0.5, 0.5), (1.7, 1.7), (120, 120), id="proto-half"),
        pytest.param((160, 160), (0.25, 0.25), (3.3, 2.1), (480, 300), id="seg-proto-160"),
        pytest.param((20, 35), (0.8, 0.3), (-2.5, 4.4), (60, 70), id="anisotropic"),
        pytest.param((30, 30), (1.4, 1.4), (-1.7, -1.4), (50, 71), id="downsample"),
        pytest.param((9, 7), (0.05, 0.07), (0.0, 0.0), (130, 90), id="tiny-source"),
        pytest.param((50, 50), (1.0, 1.0), (0.0, 0.0), (50, 50), id="identity"),
        pytest.param((40, 40), (0.3, 0.3), (-8.0, 30.0), (40, 40), id="out-of-bounds"),
    ],
)
def test_warp_scale_matches_cv2(use_cv2, src_shape, scale, offset, size) -> None:
    """warp_scale is float bilinear with replicated borders; cv2 4 and 5 agree on the 0.5 mask."""
    src = np.random.default_rng(0).random(src_shape, dtype=np.float32)
    # reference: sample src at scale * out + offset, clamped to the border
    ys = np.clip(np.arange(size[1]) * scale[1] + offset[1], 0, src_shape[0] - 1)
    xs = np.clip(np.arange(size[0]) * scale[0] + offset[0], 0, src_shape[1] - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, src_shape[0] - 1), np.minimum(x0 + 1, src_shape[1] - 1)
    fy, fx = (ys - y0)[:, None], xs - x0
    top = src[y0][:, x0] * (1 - fx) + src[y0][:, x1] * fx
    bottom = src[y1][:, x0] * (1 - fx) + src[y1][:, x1] * fx
    expected = top * (1 - fy) + bottom * fy
    actual = _ops.warp_scale(src, scale, offset, size)
    assert actual.shape == (size[1], size[0])
    assert actual.dtype == np.float32
    # opencv 4 snaps sample points to a 1/32 px grid, opencv 5 and numpy don't
    atol = 0.03 if use_cv2 else 1e-5
    assert np.abs(actual - expected).max() < atol
    flips = ((actual >= 0.5) != (expected >= 0.5)).mean()
    assert flips <= (0.01 if use_cv2 else 0.0)
