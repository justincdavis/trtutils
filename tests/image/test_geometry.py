# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_geometry.py -- find_contours against cv2.findContours."""

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
