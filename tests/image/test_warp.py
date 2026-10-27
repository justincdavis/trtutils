# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_warp.py -- perspective_crop against cv2 warpPerspective cubic."""

from __future__ import annotations

import time

import numpy as np
import pytest

from trtutils._flags import FLAGS
from trtutils.compat._libs import cv2
from trtutils.image import _warp
from trtutils.image.preprocessors import crop_text_region

pytestmark = pytest.mark.cpu

_needs_cv2 = pytest.mark.skipif(not FLAGS.FOUND_CV2, reason="the reference is OpenCV")
# opencv 5.0 changed the cubic warpPerspective arithmetic (about half the pixels differ by up to 8),
# the port matches 4.x, which is what perspective_crop targets
_needs_cv2_4 = pytest.mark.skipif(
    not FLAGS.FOUND_CV2 or int(cv2.__version__.split(".")[0]) >= 5,
    reason="the bit-exact reference is OpenCV 4.x",
)
_needs_numba = pytest.mark.skipif(not FLAGS.FOUND_NUMBA, reason="needs numba for the jit variant")

_JIT = [
    pytest.param(True, id="jit", marks=_needs_numba),
    pytest.param(False, id="python"),
]

_SHAPES = [
    pytest.param((5, 7), id="5x7"),
    pytest.param((32, 48), id="32x48"),
    pytest.param((480, 640), id="480x640"),
    pytest.param((1080, 1920), id="1080x1920"),
]
_CHANNELS = [pytest.param(3, id="c3"), pytest.param(1, id="c1")]

# quads as fractions of (w, h); an entry >1 or <0 lies outside the image
_QUADS = [
    pytest.param([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)], 40, 24, id="axis"),
    pytest.param([(0.1, 0.2), (0.9, 0.1), (0.95, 0.8), (0.12, 0.9)], 64, 32, id="rotated"),
    pytest.param([(0.3, 0.1), (0.7, 0.1), (0.95, 0.9), (0.05, 0.9)], 50, 30, id="trapezoid"),
    pytest.param([(-0.3, 0.2), (0.6, -0.2), (1.3, 0.7), (0.2, 1.2)], 48, 36, id="outside"),
    pytest.param([(0.4, 0.4), (0.45, 0.4), (0.45, 0.45), (0.4, 0.45)], 3, 2, id="tiny"),
    pytest.param([(0.05, 0.4), (0.95, 0.4), (0.95, 0.5), (0.05, 0.5)], 320, 1, id="wide"),
    pytest.param([(0.4, 0.05), (0.5, 0.05), (0.5, 0.95), (0.4, 0.95)], 1, 100, id="tall"),
    pytest.param([(0.4, 0.05), (0.5, 0.05), (0.5, 0.95), (0.4, 0.95)], 12, 48, id="vertical-1to4"),
    pytest.param([(0.1, 0.1), (0.9, 0.12), (0.88, 0.9), (0.08, 0.88)], 320, 40, id="ocr-line"),
]


def _image(shape: tuple[int, int], chans: int, seed: int) -> np.ndarray:
    full = (*shape, chans) if chans > 1 else shape
    return np.random.default_rng(seed).integers(0, 256, full, dtype=np.uint8)


def _quad(frac: list[tuple[float, float]], shape: tuple[int, int]) -> np.ndarray:
    return (np.array(frac) * [shape[1], shape[0]]).astype(np.float32)


def _reference(image: np.ndarray, quad: np.ndarray, width: int, height: int) -> np.ndarray:
    target = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    return cv2.warpPerspective(
        image,
        cv2.getPerspectiveTransform(quad, target),
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )


@pytest.fixture
def no_cv2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the numba/numpy path."""
    monkeypatch.setattr(FLAGS, "FOUND_CV2", False)


@_needs_cv2_4
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("chans", _CHANNELS)
@pytest.mark.parametrize(("frac", "width", "height"), _QUADS)
@pytest.mark.parametrize("jit", _JIT)
def test_matches_cv2(
    no_cv2: None,
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int],
    chans: int,
    frac: list[tuple[float, float]],
    width: int,
    height: int,
    jit: bool,
) -> None:
    """Bit-equality with cv2 over shapes, quads, channels and jit modes."""
    if not jit and shape[0] * shape[1] > 640 * 480 and height * width > 5000:
        pytest.skip("pure python is too slow for large outputs")
    monkeypatch.setattr(FLAGS, "JIT", jit)
    image = _image(shape, chans, seed=shape[0] + chans)
    quad = _quad(frac, shape)
    got = _warp.perspective_crop(image, quad, width, height)
    monkeypatch.setattr(FLAGS, "FOUND_CV2", True)
    expected = _reference(image, quad, width, height)
    assert got.shape == expected.shape
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, expected)


@_needs_cv2_4
@pytest.mark.parametrize("seed", [pytest.param(s, id=f"seed{s}") for s in range(40)])
def test_matches_cv2_random_quads(no_cv2: None, monkeypatch: pytest.MonkeyPatch, seed: int) -> None:
    """Bit-equality with cv2 on random quads."""
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(5, 300)), int(rng.integers(7, 400)))
    image = _image(shape, 3, seed)
    centre = rng.uniform(0, 1, 2) * [shape[1], shape[0]]
    size = rng.uniform(0.1, 1.2, 2) * [shape[1], shape[0]]
    angle = rng.uniform(-0.6, 0.6)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    base = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * size
    quad = (base @ rot.T + centre + rng.uniform(-0.15, 0.15, (4, 2)) * size).astype(np.float32)
    width, height = int(rng.integers(1, 160)), int(rng.integers(1, 120))
    got = _warp.perspective_crop(image, quad, width, height)
    monkeypatch.setattr(FLAGS, "FOUND_CV2", True)
    np.testing.assert_array_equal(got, _reference(image, quad, width, height))


@_needs_cv2_4
@pytest.mark.parametrize("seed", [pytest.param(s, id=f"seed{s}") for s in range(8)])
def test_crop_text_region_equivalent(
    no_cv2: None, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    """Match crop_text_region on realistic text boxes."""
    rng = np.random.default_rng(seed)
    image = _image((720, 1280), 3, seed)
    x, y = rng.integers(0, 900), rng.integers(0, 600)
    w, h = int(rng.integers(40, 360)), int(rng.integers(12, 60))
    jitter = rng.uniform(-3, 3, (4, 2))
    box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) + jitter
    points = box.astype(np.float32)
    width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    got = _warp.perspective_crop(image, points, width, height)
    monkeypatch.setattr(FLAGS, "FOUND_CV2", True)
    expected = crop_text_region(image, box)
    if expected.shape != got.shape:
        got = np.rot90(got)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("jit", _JIT)
@pytest.mark.parametrize("chans", _CHANNELS)
def test_identity_returns_crop(
    no_cv2: None, monkeypatch: pytest.MonkeyPatch, jit: bool, chans: int
) -> None:
    """A target-sized quad at the origin returns the image."""
    monkeypatch.setattr(FLAGS, "JIT", jit)
    image = _image((20, 30), chans, seed=1)
    quad = np.array([[0, 0], [30, 0], [30, 20], [0, 20]], dtype=np.float32)
    np.testing.assert_array_equal(_warp.perspective_crop(image, quad, 30, 20), image)


@pytest.mark.parametrize("jit", _JIT)
def test_offset_identity_returns_subimage(
    no_cv2: None, monkeypatch: pytest.MonkeyPatch, jit: bool
) -> None:
    """An integer aligned quad returns that sub image."""
    monkeypatch.setattr(FLAGS, "JIT", jit)
    image = _image((20, 30), 3, seed=2)
    quad = np.array([[5, 4], [17, 4], [17, 14], [5, 14]], dtype=np.float32)
    np.testing.assert_array_equal(_warp.perspective_crop(image, quad, 12, 10), image[4:14, 5:17])


@pytest.mark.parametrize("jit", _JIT)
def test_horizontal_stretch_of_gradient(
    no_cv2: None, monkeypatch: pytest.MonkeyPatch, jit: bool
) -> None:
    """A 2x stretch of a ramp stays a monotone ramp."""
    monkeypatch.setattr(FLAGS, "JIT", jit)
    image = np.tile(np.arange(0, 200, 2, dtype=np.uint8), (8, 1))
    quad = np.array([[0, 0], [100, 0], [100, 8], [0, 8]], dtype=np.float32)
    crop = _warp.perspective_crop(image, quad, 200, 8)
    assert crop.shape == (8, 200)
    # output x samples source (x + 0.5) / 2 - 0.5, a linear ramp away from the borders
    centre = crop[4, 8:-8].astype(np.float64)
    expected = 2 * ((np.arange(8, 192) + 0.5) / 2 - 0.5)
    np.testing.assert_allclose(centre, expected, atol=1.0)
    assert np.all(np.diff(crop[4].astype(int)) >= 0)
    assert np.all(crop == crop[0])


@pytest.mark.parametrize("jit", _JIT)
@pytest.mark.parametrize("chans", _CHANNELS)
def test_constant_image_gives_constant_crop(
    no_cv2: None, monkeypatch: pytest.MonkeyPatch, jit: bool, chans: int
) -> None:
    """A constant image gives a constant crop."""
    monkeypatch.setattr(FLAGS, "JIT", jit)
    shape = (24, 40, chans) if chans > 1 else (24, 40)
    image = np.full(shape, 137, dtype=np.uint8)
    quad = np.array([[-5, 3], [30, -2], [45, 20], [2, 30]], dtype=np.float32)
    crop = _warp.perspective_crop(image, quad, 33, 17)
    assert crop.shape[:2] == (17, 33)
    assert np.all(crop == 137)


def test_cubic_table_rows_sum_to_one() -> None:
    """Every fixed-point weight set sums to 1.0."""
    sums = _warp._CUBIC_TAB.reshape(-1, 16).sum(axis=1)
    assert np.all(sums == 1 << 15)


@_needs_cv2
@_needs_numba
@pytest.mark.parametrize(
    ("out_size", "label"),
    [pytest.param((320, 40), "ocr-crop", id="ocr-crop"), pytest.param((1200, 200), "big", id="big")],
)
def test_timing(
    no_cv2: None, monkeypatch: pytest.MonkeyPatch, out_size: tuple[int, int], label: str
) -> None:
    """Print numba vs cv2 timing."""
    monkeypatch.setattr(FLAGS, "JIT", True)
    image = _image((1080, 1920), 3, seed=3)
    width, height = out_size
    quad = np.array([[100, 300], [100 + width, 310], [96 + width, 310 + height], [98, 300 + height]])
    quad = quad.astype(np.float32)
    _warp.perspective_crop(image, quad, width, height)

    def best(fn: object) -> float:
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            fn()  # type: ignore[operator]
            times.append(time.perf_counter() - t0)
        return min(times) * 1e3

    ours = best(lambda: _warp.perspective_crop(image, quad, width, height))
    monkeypatch.setattr(FLAGS, "FOUND_CV2", True)
    theirs = best(lambda: _warp.perspective_crop(image, quad, width, height))
    print(f"{label} {width}x{height}: numba {ours:.3f} ms, cv2 {theirs:.3f} ms")
