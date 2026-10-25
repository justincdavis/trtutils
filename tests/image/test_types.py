# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/types.py -- public result type aliases."""

from __future__ import annotations

from typing import Optional, Tuple

import pytest

import trtutils.image.postprocessors
import trtutils.image.types
from trtutils.image import types

_BBOX = Tuple[int, int, int, int]
_SCORED = Tuple[_BBOX, float]


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("Bbox", _BBOX, id="bbox"),
        pytest.param("Detection", Tuple[_BBOX, float, int], id="detection"),
        pytest.param("Classification", Tuple[int, float], id="classification"),
        pytest.param(
            "HandInteraction",
            Tuple[_SCORED, Optional[_SCORED], Optional[_SCORED], Optional[int], Optional[int]],
            id="hand-interaction",
        ),
    ],
)
def test_alias(name, expected) -> None:
    """Each public alias is exported from trtutils.image.types with the documented shape."""
    assert name in types.__all__
    assert getattr(types, name) == expected


@pytest.mark.cpu
def test_hand_interaction_reexport() -> None:
    """postprocessors.HandInteraction is the same object as types.HandInteraction."""
    assert types is trtutils.image.types
    assert trtutils.image.postprocessors.HandInteraction is types.HandInteraction
