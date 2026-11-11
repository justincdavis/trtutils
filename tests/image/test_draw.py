# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_draw.py -- box and text drawing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image import draw_bboxes, draw_text

pytestmark = pytest.mark.cpu


def _canvas() -> np.ndarray:
    return np.zeros((120, 200, 3), dtype=np.uint8)


def test_draw_text() -> None:
    """draw_text draws red pixels in place and returns the same image."""
    canvas = _canvas()
    out = draw_text(canvas, "hi", (10, 40))
    assert out is canvas
    assert canvas[..., 2].any()
    assert not canvas[..., :2].any()


def test_draw_text_color() -> None:
    """draw_text uses the given BGR color."""
    canvas = _canvas()
    draw_text(canvas, "hi", (10, 40), color=(0, 255, 0))
    assert canvas[..., 1].any()
    assert not canvas[..., [0, 2]].any()


def test_draw_bboxes_rectangle() -> None:
    """Without tags only the box outline is drawn, in red, in place."""
    canvas = _canvas()
    out = draw_bboxes(canvas, [(20, 30, 100, 90)])
    assert out is canvas
    assert canvas[30, 60, 2] == 255
    assert not canvas[40:80, 30:90].any()
    assert not canvas[..., :2].any()
    ys, xs = np.nonzero(canvas[..., 2])
    assert 17 <= xs.min() <= 20
    assert 99 <= xs.max() <= 103
    assert 27 <= ys.min() <= 30
    assert 89 <= ys.max() <= 93


@pytest.mark.parametrize(
    ("scores", "classes"),
    [
        pytest.param([0.5], None, id="score-only"),
        pytest.param(None, [3], id="class-only"),
        pytest.param([0.5], [3], id="score-and-class"),
    ],
)
def test_draw_bboxes_tag(scores, classes) -> None:
    """A score and/or class adds a tag above the box top-left that the plain box lacks."""
    plain = draw_bboxes(_canvas(), [(20, 60, 100, 100)])
    tagged = draw_bboxes(_canvas(), [(20, 60, 100, 100)], scores, classes)
    assert (tagged != plain).any()
