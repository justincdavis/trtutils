# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/compat/_libs.py -- optional cv2 guard and FLAGS.FOUND_CV2."""

from __future__ import annotations

import importlib.util

import pytest

from trtutils._flags import FLAGS
from trtutils.compat import _libs

pytestmark = pytest.mark.cpu


def test_found_cv2_matches_install() -> None:
    """FLAGS.FOUND_CV2 is True exactly when OpenCV is importable."""
    assert (importlib.util.find_spec("cv2") is not None) == FLAGS.FOUND_CV2


@pytest.mark.skipif(FLAGS.FOUND_CV2, reason="OpenCV is installed")
def test_cv2_missing_raises_hint() -> None:
    """Without OpenCV, any cv2 use raises with the install hint."""
    with pytest.raises(AttributeError, match=r"trtutils\[cv2\]"):
        _libs.cv2.resize  # noqa: B018
