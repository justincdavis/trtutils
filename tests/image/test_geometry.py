# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_geometry.py -- find_contours and min_area_rect against OpenCV."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils._flags import FLAGS
from trtutils.compat._libs import cv2
from trtutils.image import _geometry

pytestmark = pytest.mark.cpu

_needs_cv2 = pytest.mark.skipif(not FLAGS.FOUND_CV2, reason="the reference is OpenCV")
_needs_numba = pytest.mark.skipif(not FLAGS.FOUND_NUMBA, reason="needs numba for the jit variant")

_JIT = [
    pytest.param(True, id="jit", marks=_needs_numba),
    pytest.param(False, id="python"),
]


def _blocks(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    for y, x, h, w in rects:
        m[y : y + h, x : x + w] = 1
    return m


def _noise(shape: tuple[int, int], p: float, seed: int) -> np.ndarray:
    return (np.random.default_rng(seed).random(shape) < p).astype(np.uint8)


def _text_like(shape: tuple[int, int], seed: int, n: int = 150) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = np.zeros(shape, dtype=np.uint8)
    for _ in range(n):
        h = int(rng.integers(2, shape[0] // 24 + 3))
        w = int(rng.integers(2, shape[1] // 8 + 3))
        y = int(rng.integers(0, shape[0] - h))
        x = int(rng.integers(0, shape[1] - w))
        m[y : y + h, x : x + w] = 1
    return m


def _diag(n: int) -> np.ndarray:
    return np.eye(n, dtype=np.uint8)


_RING = _blocks((9, 9), [(1, 1, 7, 7)])
_RING[3:6, 3:6] = 0
_ISLAND = _RING.copy()
_ISLAND[4, 4] = 1
_TOUCH = _blocks((8, 8), [(1, 1, 2, 2), (3, 3, 2, 2)])
_EDGES = _blocks((8, 8), [(0, 0, 2, 2), (0, 6, 2, 2), (6, 0, 2, 2), (6, 6, 2, 2), (0, 3, 1, 2)])

_SMALL = [
    pytest.param(_blocks((5, 5), [(2, 2, 1, 1)]), id="single-pixel"),
    pytest.param(_blocks((6, 6), [(2, 2, 2, 2)]), id="block-2x2"),
    pytest.param(_blocks((5, 9), [(2, 1, 1, 7)]), id="line-horizontal"),
    pytest.param(_blocks((9, 5), [(1, 2, 7, 1)]), id="line-vertical"),
    pytest.param(_diag(9), id="line-diagonal"),
    pytest.param(np.fliplr(_diag(9)).copy(), id="line-antidiagonal"),
    pytest.param(_RING, id="ring"),
    pytest.param(_ISLAND, id="ring-island"),
    pytest.param(_TOUCH, id="diagonal-touching"),
    pytest.param(_EDGES, id="image-edges"),
    pytest.param(np.ones((6, 7), dtype=np.uint8), id="full"),
    pytest.param(np.zeros((6, 7), dtype=np.uint8), id="empty"),
    pytest.param(np.ones((1, 1), dtype=np.uint8), id="one-by-one"),
    pytest.param(np.ones((1, 8), dtype=np.uint8), id="single-row"),
    pytest.param(np.ones((8, 1), dtype=np.uint8), id="single-col"),
    pytest.param(_noise((24, 30), 0.1, 1), id="noise-0.1"),
    pytest.param(_noise((24, 30), 0.3, 2), id="noise-0.3"),
    pytest.param(_noise((24, 30), 0.5, 3), id="noise-0.5"),
    pytest.param(_noise((24, 30), 0.7, 4), id="noise-0.7"),
    pytest.param(_text_like((60, 80), 5, 12), id="text-small"),
]

_BIG = [
    pytest.param(_text_like((736, 960), 10), id="text-960x736-a"),
    pytest.param(_text_like((736, 960), 11, 400), id="text-960x736-dense"),
    pytest.param(_noise((300, 300), 0.5, 12), id="noise-0.5-300"),
]


@pytest.fixture
def jit(request, monkeypatch) -> bool:
    """Set FLAGS.JIT to the param."""
    monkeypatch.setattr(FLAGS, "JIT", request.param)
    return request.param


@pytest.fixture
def no_cv2(monkeypatch) -> None:
    """Force the numpy path."""
    monkeypatch.setattr(FLAGS, "FOUND_CV2", False)


def _reference(binary: np.ndarray) -> list[np.ndarray]:
    mask = (binary != 0).astype(np.uint8) * 255
    found = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return [c.reshape(-1, 2) for c in found]


def _key(contour: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(map(tuple, contour.tolist())))


def _assert_matches_cv2(binary: np.ndarray) -> None:
    got = _geometry.find_contours(binary)
    want = _reference(binary)
    assert [_key(c) for c in got] == [_key(c) for c in want]
    assert all(c.dtype == np.int32 and c.ndim == 2 and c.shape[1] == 2 for c in got)


@_needs_cv2
@pytest.mark.parametrize("jit", _JIT, indirect=True)
@pytest.mark.parametrize("binary", _SMALL)
def test_matches_cv2_small(binary, jit, no_cv2) -> None:
    """Same contours, same point multisets, same order as cv2."""
    _assert_matches_cv2(binary)


@_needs_cv2
@_needs_numba
@pytest.mark.parametrize("jit", [True], indirect=True)
@pytest.mark.parametrize("binary", _BIG)
def test_matches_cv2_big(binary, jit, no_cv2) -> None:
    """Large maps, jit only."""
    _assert_matches_cv2(binary)


@_needs_cv2
@_needs_numba
@pytest.mark.parametrize("jit", [True], indirect=True)
@pytest.mark.parametrize("seed", range(40))
def test_matches_cv2_random_shapes(seed, jit, no_cv2) -> None:
    """Random shapes and densities."""
    rng = np.random.default_rng(1000 + seed)
    shape = (int(rng.integers(1, 20)), int(rng.integers(1, 20)))
    _assert_matches_cv2(_noise(shape, float(rng.choice([0.1, 0.3, 0.5, 0.7, 0.9])), seed))


@_needs_cv2
@pytest.mark.parametrize("use_cv2", [True, False], ids=["cv2", "numpy"])
def test_dispatch_and_nonzero(use_cv2, monkeypatch) -> None:
    """Both paths accept arbitrary nonzero values."""
    monkeypatch.setattr(FLAGS, "FOUND_CV2", use_cv2)
    got = _geometry.find_contours(_RING.astype(np.float32) * 0.7)
    assert [_key(c) for c in got] == [_key(c) for c in _reference(_RING)]


_LITERAL = [
    pytest.param(_blocks((5, 5), [(2, 3, 1, 1)]), [[[3, 2]]], id="single-pixel"),
    pytest.param(
        _blocks((6, 6), [(1, 1, 3, 3)]),
        [[[1, 1], [1, 3], [3, 3], [3, 1]]],
        id="block-3x3-corners",
    ),
    pytest.param(
        _blocks((3, 7), [(1, 1, 1, 5)]),
        [[[1, 1], [5, 1]]],
        id="line-out-and-back",
    ),
    pytest.param(
        _blocks((7, 7), [(1, 1, 5, 5)]) - _blocks((7, 7), [(2, 2, 3, 3)]),
        [
            [[1, 2], [1, 4], [2, 5], [4, 5], [5, 4], [5, 2], [4, 1], [2, 1]],
            [[1, 1], [1, 5], [5, 5], [5, 1]],
        ],
        id="ring-outer-and-hole",
    ),
    pytest.param(np.zeros((4, 4), dtype=np.uint8), [], id="empty"),
    pytest.param(
        _blocks((5, 5), [(0, 0, 1, 1), (1, 1, 1, 1)]),
        [[[0, 0], [1, 1]]],
        id="diagonal-pair-8-connected",
    ),
    pytest.param(
        _blocks((4, 7), [(1, 1, 1, 1), (1, 5, 1, 1)]),
        [[[5, 1]], [[1, 1]]],
        id="two-pixels-last-found-first",
    ),
]


@pytest.mark.parametrize("jit", _JIT, indirect=True)
@pytest.mark.parametrize(("binary", "expected"), _LITERAL)
def test_literal_contours(binary, expected, jit, no_cv2) -> None:
    """Hand written expectations, no cv2 needed."""
    got = _geometry.find_contours(binary)
    assert [_key(np.array(c, dtype=np.int32).reshape(-1, 2)) for c in expected] == [
        _key(c) for c in got
    ]


def _sorted(corners: np.ndarray) -> np.ndarray:
    return corners[np.lexsort((corners[:, 1], corners[:, 0]))]


def _area(corners: np.ndarray) -> float:
    c = corners.astype(np.float64)
    c = c[np.argsort(np.arctan2(*(c - c.mean(axis=0)).T[::-1]))]
    x, y = c.T
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ref_rect(points: np.ndarray) -> tuple[np.ndarray, float]:
    rect = cv2.minAreaRect(points)
    return cv2.boxPoints(rect), float(min(rect[1]))


def _assert_rect_matches_cv2(points: np.ndarray, *, exact: bool = True) -> None:
    corners, short = _geometry.min_area_rect(points)
    want, want_short = _ref_rect(points)
    assert corners.shape == (4, 2)
    assert corners.dtype == np.float32
    if exact:
        # same arithmetic as opencv: bitwise equal corners, in the same order
        np.testing.assert_array_equal(corners, want)
        assert short == want_short
    else:
        assert _area(corners) == pytest.approx(_area(want), rel=1e-4, abs=1e-3)
        assert short == pytest.approx(want_short, rel=1e-4, abs=1e-3)


def _int_cloud(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 201))
    return rng.integers(0, int(rng.choice([10, 100, 1000])), (n, 2)).astype(np.int32)


def _float_cloud(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 201))
    return (rng.random((n, 2)) * rng.choice([1, 50, 900])).astype(np.float32)


def _rotated(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cx, cy = rng.uniform(50, 400, 2)
    w, h = rng.uniform(3, 200, 2)
    a = rng.uniform(0, np.pi)
    q = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.round(q @ rot.T + (cx, cy)).astype(np.int32)


def _polygon(n: int, *, rounded: bool) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.c_[np.cos(t), np.sin(t)] * 100 + 200
    return np.round(pts).astype(np.int32) if rounded else pts.astype(np.float32)


def _i32(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int32)


_RECT_SMALL = [
    pytest.param(_i32([[3, 4]]), id="one-point"),
    pytest.param(_i32([[1, 1], [1, 1], [1, 1]]), id="same-point-x3"),
    pytest.param(_i32([[0, 0], [4, 2]]), id="two-points"),
    pytest.param(_i32([[5, 0], [0, 0]]), id="two-points-reversed"),
    pytest.param(_i32([[0, 0], [2, 2], [4, 4]]), id="collinear-diagonal"),
    pytest.param(_i32([[0, 5], [0, 0], [0, 2]]), id="collinear-vertical"),
    pytest.param(_i32([[0, 0], [9, 0], [3, 0], [4, 0]]), id="collinear-horizontal"),
    pytest.param(_i32([[0, 0], [10, 0], [10, 5], [0, 5]]), id="rect-wide"),
    pytest.param(_i32([[0, 0], [5, 0], [5, 10], [0, 10]]), id="rect-tall"),
    pytest.param(_i32([[127, 657], [127, 662], [248, 662], [248, 657]]), id="rect-text-line"),
    pytest.param(_i32([[0, 0], [5, 5], [0, 10], [-5, 5]]), id="square-45deg"),
    pytest.param(_i32([[0, 0], [4, 0], [0, 3]]), id="triangle-right"),
    pytest.param(_i32([[0, 0], [9, 1], [4, 8]]), id="triangle-scalene"),
    pytest.param(_i32([[0, 0], [4, 0], [4, 4], [0, 4], [2, 2], [2, 0]]), id="square-inner-points"),
    pytest.param(_i32([[-5, -5], [-1, -9], [3, -2], [-7, 4]]), id="negative-coords"),
    pytest.param(_polygon(3, rounded=True), id="polygon-3-int"),
    pytest.param(_polygon(6, rounded=True), id="polygon-6-int"),
    pytest.param(_polygon(12, rounded=False), id="polygon-12-float"),
    pytest.param(_rotated(1), id="rotated-int-1"),
    pytest.param(_int_cloud(1), id="cloud-int-1"),
    pytest.param(_float_cloud(1), id="cloud-float-1"),
]


@_needs_cv2
@pytest.mark.parametrize("jit", _JIT, indirect=True)
@pytest.mark.parametrize("points", _RECT_SMALL)
def test_rect_matches_cv2_small(points, jit, no_cv2) -> None:
    """Corners and short side equal cv2.boxPoints / min(minAreaRect size), jit and python."""
    _assert_rect_matches_cv2(points)


@_needs_cv2
@_needs_numba
@pytest.mark.parametrize("jit", [True], indirect=True)
@pytest.mark.parametrize(
    "make",
    [
        pytest.param(_int_cloud, id="int-cloud"),
        pytest.param(_float_cloud, id="float-cloud"),
        pytest.param(_rotated, id="rotated-rect"),
    ],
)
@pytest.mark.parametrize("seed", range(60))
def test_rect_matches_cv2_random(make, seed, jit, no_cv2) -> None:
    """Random clouds (sizes 3..200) and rounded rotated rectangles."""
    _assert_rect_matches_cv2(make(seed))


@_needs_cv2
@_needs_numba
@pytest.mark.parametrize("jit", [True], indirect=True)
@pytest.mark.parametrize("seed", [10, 11, 12])
def test_rect_matches_cv2_text_contours(seed, jit, no_cv2) -> None:
    """Every contour of a 960x736 text-like map, corners often land on integers."""
    contours = _geometry.find_contours(_text_like((736, 960), seed, 300))
    assert len(contours) > 50
    for contour in contours:
        _assert_rect_matches_cv2(contour)


@_needs_cv2
@_needs_numba
@pytest.mark.parametrize("jit", [True], indirect=True)
@pytest.mark.parametrize("seed", range(20))
def test_rect_floor_ceil_round_match_cv2(seed, jit, no_cv2) -> None:
    """The ops OCR applies to corners never flip against cv2."""
    for contour in _geometry.find_contours(_text_like((736, 960), 100 + seed, 100)):
        corners, _ = _geometry.min_area_rect(contour)
        want, _ = _ref_rect(contour)
        for op in (np.floor, np.ceil, np.round):
            np.testing.assert_array_equal(op(corners), op(want))


@_needs_cv2
@_needs_numba
@pytest.mark.parametrize("jit", [True], indirect=True)
@pytest.mark.parametrize("n", range(3, 41))
@pytest.mark.parametrize("rounded", [True, False], ids=["int", "float"])
def test_rect_regular_polygons(n, rounded, jit, no_cv2) -> None:
    """Regular polygons tie on area, only the area and short side are compared."""
    _assert_rect_matches_cv2(_polygon(n, rounded=rounded), exact=False)


@_needs_cv2
@_needs_numba
@pytest.mark.parametrize("jit", [True], indirect=True)
@pytest.mark.parametrize("seed", range(40))
def test_rect_duplicates(seed, jit, no_cv2) -> None:
    """Heavy duplicates make opencv's unstable sort pick any tied rectangle."""
    rng = np.random.default_rng(seed)
    points = rng.integers(0, int(rng.choice([2, 3, 5])), (int(rng.integers(3, 60)), 2))
    _assert_rect_matches_cv2(points.astype(np.int32), exact=False)


@_needs_cv2
@pytest.mark.parametrize("use_cv2", [True, False], ids=["cv2", "numpy"])
@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.int64, np.float64])
def test_rect_dispatch_and_dtypes(use_cv2, dtype, monkeypatch) -> None:
    """Both paths accept any numeric dtype and agree with cv2."""
    monkeypatch.setattr(FLAGS, "FOUND_CV2", use_cv2)
    points = _rotated(3)
    got, short = _geometry.min_area_rect(points.astype(dtype))
    want, want_short = _ref_rect(points)
    np.testing.assert_array_equal(got, want)
    assert short == want_short


_S2 = float(np.sqrt(8))
_RECT_LITERAL = [
    pytest.param(
        _i32([[0, 0], [4, 0], [4, 2], [0, 2]]),
        [[0, 0], [0, 2], [4, 0], [4, 2]],
        2.0,
        id="axis-aligned",
    ),
    pytest.param(
        _i32([[2, 0], [4, 2], [2, 4], [0, 2]]),
        [[0, 2], [2, 0], [2, 4], [4, 2]],
        _S2,
        id="square-45deg",
    ),
    pytest.param(
        _i32([[0, 0], [4, 0], [4, 2], [0, 2], [2, 1], [1, 1], [4, 0]]),
        [[0, 0], [0, 2], [4, 0], [4, 2]],
        2.0,
        id="axis-aligned-inner-and-duplicate",
    ),
    pytest.param(
        _i32([[0, 0], [4, 0], [0, 3]]), [[0, 0], [0, 3], [4, 0], [4, 3]], 3.0, id="right-triangle"
    ),
    pytest.param(_i32([[3, 4]]), [[3, 4]] * 4, 0.0, id="one-point"),
    pytest.param(_i32([[2, 2], [2, 2], [2, 2]]), [[2, 2]] * 4, 0.0, id="same-point-x3"),
    pytest.param(_i32([[0, 0], [4, 2]]), [[0, 0], [0, 0], [4, 2], [4, 2]], 0.0, id="two-points"),
    pytest.param(
        _i32([[0, 0], [2, 2], [1, 1], [4, 4]]),
        [[0, 0], [0, 0], [4, 4], [4, 4]],
        0.0,
        id="collinear",
    ),
]


@pytest.mark.parametrize("jit", _JIT, indirect=True)
@pytest.mark.parametrize(("points", "expected", "short"), _RECT_LITERAL)
def test_rect_literal(points, expected, short, jit, no_cv2) -> None:
    """Hand written expectations, no cv2 needed."""
    corners, got_short = _geometry.min_area_rect(points)
    np.testing.assert_allclose(_sorted(corners), _sorted(np.array(expected)), atol=1e-5)
    assert got_short == pytest.approx(short, abs=1e-5)


@pytest.mark.parametrize("jit", _JIT, indirect=True)
def test_rect_no_points(jit, no_cv2) -> None:
    """No points gives an all zero rectangle, like cv2."""
    corners, short = _geometry.min_area_rect(np.zeros((0, 2), dtype=np.int32))
    np.testing.assert_array_equal(corners, np.zeros((4, 2), dtype=np.float32))
    assert short == 0.0


def _cv2_dilate(binary: np.ndarray) -> np.ndarray:
    return cv2.dilate(binary, np.ones((2, 2), dtype=np.uint8))


@_needs_cv2
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1), id="1x1"),
        pytest.param((1, 9), id="1xN"),
        pytest.param((9, 1), id="Nx1"),
        pytest.param((2, 2), id="2x2"),
        pytest.param((3, 7), id="3x7"),
        pytest.param((64, 80), id="64x80"),
        pytest.param((736, 960), id="736x960"),
    ],
)
@pytest.mark.parametrize("density", [0.0, 0.02, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_dilate_matches_cv2(shape, density, seed, no_cv2) -> None:
    """Random maps of many shapes and densities, anchor and borders match cv2."""
    binary = _noise(shape, density, seed)
    np.testing.assert_array_equal(_geometry.dilate_2x2(binary), _cv2_dilate(binary))


@_needs_cv2
@pytest.mark.parametrize("y", [0, 3, 7])
@pytest.mark.parametrize("x", [0, 4, 8])
def test_dilate_single_pixel_matches_cv2(y, x, no_cv2) -> None:
    """One lit pixel, including corners and edges, grows down and right only."""
    binary = np.zeros((8, 9), dtype=np.uint8)
    binary[min(y, 7), x] = 1
    np.testing.assert_array_equal(_geometry.dilate_2x2(binary), _cv2_dilate(binary))


@_needs_cv2
def test_dilate_dispatch(monkeypatch) -> None:
    """The cv2 path gives the same answer as the numpy path."""
    binary = _noise((20, 30), 0.1, 3)
    monkeypatch.setattr(FLAGS, "FOUND_CV2", True)
    with_cv2 = _geometry.dilate_2x2(binary)
    monkeypatch.setattr(FLAGS, "FOUND_CV2", False)
    np.testing.assert_array_equal(_geometry.dilate_2x2(binary), with_cv2)


def test_dilate_literal(no_cv2) -> None:
    """A pixel spreads to the pixel below, right and diagonal; the input is untouched."""
    binary = np.zeros((4, 4), dtype=np.uint8)
    binary[1, 1] = 1
    binary[3, 3] = 1
    before = binary.copy()
    want = np.array(
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(_geometry.dilate_2x2(binary), want)
    np.testing.assert_array_equal(binary, before)


def _random_box(kind: int, seed: int) -> tuple[tuple[int, int], np.ndarray]:
    """A (shape, int32 polygon) for the fillPoly matrix."""
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(1, 60)), int(rng.integers(1, 80))
    if kind == 0:
        centre = rng.random(2) * (w, h)
        angle = rng.random() * 6.3
        size = rng.random(2) * (w, h)
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * size / 2
    elif kind == 1:
        centre = rng.random(2) * (w, h)
        angle = rng.random() * 6.3
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * (
            rng.random() * max(w, h) / 2,
            rng.random() * 0.75,
        )
    elif kind == 2:
        pts = rng.random((4, 2)) * 3 + rng.random(2) * (w, h)
        return (h, w), pts.astype(np.int32)
    elif kind == 3:
        pts = (rng.random((4, 2)) - 0.3) * (w * 1.6, h * 1.6)
        return (h, w), pts.astype(np.int32)
    elif kind == 4:
        base = rng.random((2, 2)) * (w, h)
        return (h, w), base[rng.integers(0, 2, 4)].astype(np.int32)
    else:
        pts = rng.integers(-5, max(w, h) + 5, (4, 2))
        return (h, w), pts.astype(np.int32)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (h, w), (corners @ rot.T + centre).astype(np.int32)


_KINDS = [
    pytest.param(0, id="rotated-rect"),
    pytest.param(1, id="sliver"),
    pytest.param(2, id="tiny"),
    pytest.param(3, id="partly-outside"),
    pytest.param(4, id="degenerate"),
    pytest.param(5, id="random-quad"),
]


@_needs_cv2
@pytest.mark.parametrize("jit", _JIT, indirect=True)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("seed", range(60))
def test_fill_matches_cv2(kind, seed, jit) -> None:
    """Masks are bit identical to cv2.fillPoly, jit and python."""
    shape, pts = _random_box(kind, seed)
    want = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(want, [pts], (1,))
    np.testing.assert_array_equal(_geometry._fill_convex_poly(shape, pts), want)


@_needs_cv2
@_needs_numba
@pytest.mark.parametrize("jit", [True], indirect=True)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("block", range(10))
def test_fill_matches_cv2_many(kind, block, jit) -> None:
    """Hundreds more masks per kind with jit on."""
    for seed in range(1000 + block * 100, 1100 + block * 100):
        shape, pts = _random_box(kind, seed)
        want = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(want, [pts], (1,))
        np.testing.assert_array_equal(_geometry._fill_convex_poly(shape, pts), want)


@_needs_cv2
@pytest.mark.parametrize("jit", _JIT, indirect=True)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("seed", range(30))
def test_polygon_mean_matches_cv2(kind, seed, jit, monkeypatch) -> None:
    """Float boxes over random maps score like cv2 (the numpy path vs the cv2 path)."""
    rng = np.random.default_rng(seed + 500 * kind)
    shape, pts = _random_box(kind, seed)
    pred = rng.random(shape).astype(np.float32)
    box = pts.astype(np.float32) + rng.random((4, 2)).astype(np.float32)
    monkeypatch.setattr(FLAGS, "FOUND_CV2", True)
    want = _geometry.polygon_mean(pred, box)
    monkeypatch.setattr(FLAGS, "FOUND_CV2", False)
    got = _geometry.polygon_mean(pred, box)
    assert got == pytest.approx(want, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("jit", _JIT, indirect=True)
def test_polygon_mean_literal_rect(jit, no_cv2) -> None:
    """An axis aligned rectangle includes its boundary pixels."""
    mask = _geometry._fill_convex_poly((6, 8), np.array([[1, 1], [5, 1], [5, 3], [1, 3]]))
    want = np.zeros((6, 8), dtype=np.uint8)
    want[1:4, 1:6] = 1
    np.testing.assert_array_equal(mask, want)


@pytest.mark.parametrize("jit", _JIT, indirect=True)
def test_polygon_mean_literal_diamond(jit, no_cv2) -> None:
    """A 45 degree diamond fills row spans between its edges."""
    mask = _geometry._fill_convex_poly((5, 5), np.array([[2, 0], [4, 2], [2, 4], [0, 2]]))
    want = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(mask, want)


@pytest.mark.parametrize("jit", _JIT, indirect=True)
def test_polygon_mean_literal_value(jit, no_cv2) -> None:
    """The mean is over the filled pixels only; empty overlap scores 0."""
    pred = np.arange(48, dtype=np.float32).reshape(6, 8)
    box = np.array([[1.9, 1.2], [5.5, 1.2], [5.5, 3.8], [1.9, 3.8]], dtype=np.float32)
    # window x 1..6, y 1..4; polygon truncates to x 1..5, y 1..3
    want = float(pred[1:4, 1:6].mean(dtype=np.float64))
    assert _geometry.polygon_mean(pred, box) == pytest.approx(want, rel=1e-12)
    outside = np.array([[-9, -9], [-5, -9], [-5, -5], [-9, -5]], dtype=np.float32)
    assert _geometry.polygon_mean(pred, outside) == 0.0


@_needs_cv2
def test_pad_matches_copy_make_border() -> None:
    """np.pad on rows equals cv2.copyMakeBorder with a constant black border."""
    rng = np.random.default_rng(0)
    for pad in (0, 1, 3, 17):
        for shape in ((1, 1, 3), (5, 9, 3), (48, 320, 3)):
            img = rng.integers(0, 256, shape).astype(np.uint8)
            want = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            np.testing.assert_array_equal(np.pad(img, ((pad, pad), (0, 0), (0, 0))), want)


@_needs_cv2
@pytest.mark.parametrize("shape", [(1, 1, 3), (7, 5, 3), (48, 320, 3)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lut_matches_split_lut(shape, seed) -> None:
    """Fancy indexing a (256, 3) table equals cv2.split + cv2.LUT per channel."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, shape).astype(np.uint8)
    lut = rng.random((256, 3)).astype(np.float32)
    want = np.empty((3, *shape[:2]), dtype=np.float32)
    for c, plane in enumerate(cv2.split(img)):
        cv2.LUT(plane, lut[:, c], dst=want[c])
    got = lut[img, np.arange(3)].transpose(2, 0, 1)
    np.testing.assert_array_equal(got, want)
