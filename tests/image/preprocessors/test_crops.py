# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/preprocessors/_crops.py -- box crops as views of the image."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.image.preprocessors import crop_boxes

HEIGHT, WIDTH = 60, 80


@pytest.fixture
def image() -> np.ndarray:
    """A (60, 80, 3) uint8 image whose pixels hold distinct values."""
    return (np.arange(HEIGHT * WIDTH * 3) % 251).astype(np.uint8).reshape(HEIGHT, WIDTH, 3)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("box", "expected"),
    [
        pytest.param((10, 5, 30, 25), (10, 5, 30, 25), id="inside"),
        pytest.param((-10, 5, 30, 25), (0, 5, 30, 25), id="clip-left"),
        pytest.param((10, -5, 30, 25), (10, 0, 30, 25), id="clip-top"),
        pytest.param((10, 5, 100, 25), (10, 5, WIDTH, 25), id="clip-right"),
        pytest.param((10, 5, 30, 90), (10, 5, 30, HEIGHT), id="clip-bottom"),
        pytest.param((-5, -5, 200, 200), (0, 0, WIDTH, HEIGHT), id="clip-all"),
        pytest.param((10.7, 5.2, 29.1, 24.9), (10, 5, 30, 25), id="float-widened"),
        pytest.param((10.2, 5.2, 10.8, 5.8), (10, 5, 11, 6), id="float-subpixel"),
    ],
)
@pytest.mark.parametrize("as_array", [True, False], ids=["array", "sequence"])
def test_crop_boxes_region(image, box, expected, as_array) -> None:
    """A box crops its clipped, pixel-widened region as a view of the image."""
    boxes = np.array([box]) if as_array else [box]
    (crop,) = crop_boxes(image, boxes)
    x1, y1, x2, y2 = expected
    np.testing.assert_array_equal(crop, image[y1:y2, x1:x2])
    assert np.shares_memory(crop, image)
    crop[...] = 0
    assert not image[y1:y2, x1:x2].any()


@pytest.mark.cpu
def test_crop_boxes_order(image) -> None:
    """Crops come back in box order, including repeated and overlapping boxes."""
    boxes = [(40, 10, 60, 50), (0, 0, 10, 10), (5, 5, 45, 30), (0, 0, 10, 10)]
    crops = crop_boxes(image, np.array(boxes, dtype=np.float32))
    assert len(crops) == len(boxes)
    for crop, (x1, y1, x2, y2) in zip(crops, boxes):
        np.testing.assert_array_equal(crop, image[y1:y2, x1:x2])


@pytest.mark.cpu
@pytest.mark.parametrize(
    "boxes",
    [
        pytest.param([], id="empty-list"),
        pytest.param(np.zeros((0, 4)), id="empty-array"),
    ],
)
def test_crop_boxes_none(image, boxes) -> None:
    """No boxes gives no crops."""
    assert crop_boxes(image, boxes) == []


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("boxes", "match"),
    [
        pytest.param([(10, 5, 30, 25), (10, 5, 10, 25)], "box 1 ", id="zero-width"),
        pytest.param([(10, 5, 10.0, 5.0)], "box 0 ", id="zero-area"),
        pytest.param([(10, 5, 30, 25), (0, 0, 5, 5), (30, 5, 10, 25)], "box 2 ", id="inverted-x"),
        pytest.param([(10, 25, 30, 5)], "box 0 ", id="inverted-y"),
        pytest.param([(100, 5, 120, 25)], "box 0 ", id="outside-right"),
        pytest.param([(10, -30, 30, -5)], "box 0 ", id="outside-top"),
        pytest.param([(1, 2, 3)], r"\(N, 4\)", id="three-coords"),
        pytest.param([1, 2, 3, 4], r"\(N, 4\)", id="flat-box"),
    ],
)
def test_crop_boxes_invalid(image, boxes, match) -> None:
    """A box with no area inside the image, or boxes not (N, 4), raise ValueError."""
    with pytest.raises(ValueError, match=match):
        crop_boxes(image, boxes)
