# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_class_maps.py."""

from __future__ import annotations

import pytest

from trtutils.image import COCO91_TO_COCO80


def test_coco91_to_coco80_is_bijective() -> None:
    """80 category ids inside 1..90 map onto exactly the indices 0..79."""
    assert len(COCO91_TO_COCO80) == 80
    assert all(1 <= key <= 90 for key in COCO91_TO_COCO80)
    assert sorted(COCO91_TO_COCO80.values()) == list(range(80))


@pytest.mark.parametrize(
    ("category_id", "index"),
    [
        pytest.param(1, 0, id="person"),
        pytest.param(13, 11, id="stop-sign-after-gap"),
        pytest.param(19, 17, id="horse"),
        pytest.param(90, 79, id="toothbrush"),
        pytest.param(27, 24, id="backpack"),
        pytest.param(67, 60, id="dining-table"),
    ],
)
def test_coco91_to_coco80_known_entries(category_id, index) -> None:
    """Spot-check entries around the id gaps against the official COCO categories."""
    assert COCO91_TO_COCO80[category_id] == index


@pytest.mark.parametrize("gap_id", [12, 26, 29, 30, 45, 66, 68, 69, 71, 83])
def test_coco91_to_coco80_gaps_absent(gap_id) -> None:
    """The ten unused COCO category ids are not in the table."""
    assert gap_id not in COCO91_TO_COCO80
