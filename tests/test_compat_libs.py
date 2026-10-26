# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/compat/_libs.py -- optional cv2 guard and FLAGS.FOUND_CV2."""

from __future__ import annotations

import cv2
import pytest

from trtutils._flags import FLAGS
from trtutils.compat import _libs

pytestmark = pytest.mark.cpu


def test_empty_cv2_names_extra() -> None:
    """Empty cv2 stub points at the extra."""
    empty = _libs._EmptyModule("cv2", ", install with: pip install trtutils[cv2]")
    with pytest.raises(AttributeError, match=r"trtutils\[cv2\]"):
        empty.findContours  # noqa: B018


def test_cv2_found() -> None:
    """Installed cv2 is routed through compat and flagged."""
    assert _libs.cv2 is cv2
    assert FLAGS.FOUND_CV2
