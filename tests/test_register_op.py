# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for trtutils._jit.register_op -- import-time choice, rebound when flags change."""

from __future__ import annotations

import pytest

from trtutils._flags import FLAGS
from trtutils._jit import register_op

pytestmark = pytest.mark.cpu


def test_register_picks_at_import_and_rebinds(monkeypatch) -> None:
    """OpenCV, then numba, then numpy, with the stored function replaced when a flag changes."""
    monkeypatch.setattr(FLAGS, "FOUND_CV2", False)
    monkeypatch.setattr(FLAGS, "JIT", False)
    monkeypatch.setattr(FLAGS, "FOUND_NUMBA", True)

    def cv2_fn(value: int) -> str:
        return f"cv2-{value}"

    def numba_fn(value: int) -> str:
        return f"numba-{value}"

    def numpy_fn(value: int) -> str:
        return f"numpy-{value}"

    ns: dict[str, object] = {}
    register_op(ns, "fn", cv2=cv2_fn, numba=numba_fn, numpy=numpy_fn)
    assert ns["fn"] is numpy_fn

    monkeypatch.setattr(FLAGS, "JIT", True)
    assert ns["fn"] is numba_fn

    monkeypatch.setattr(FLAGS, "FOUND_CV2", True)
    assert ns["fn"] is cv2_fn
    assert ns["fn"](3) == "cv2-3"

    monkeypatch.setattr(FLAGS, "FOUND_CV2", False)
    monkeypatch.setattr(FLAGS, "JIT", False)
    assert ns["fn"] is numpy_fn


def test_register_without_cv2_impl(monkeypatch) -> None:
    """A kernel with no OpenCV implementation stays on numba or numpy."""
    monkeypatch.setattr(FLAGS, "FOUND_CV2", True)
    monkeypatch.setattr(FLAGS, "JIT", True)
    monkeypatch.setattr(FLAGS, "FOUND_NUMBA", True)

    def numba_fn() -> str:
        return "numba"

    def numpy_fn() -> str:
        return "numpy"

    ns: dict[str, object] = {}
    register_op(ns, "fn", numba=numba_fn, numpy=numpy_fn)
    assert ns["fn"] is numba_fn
    monkeypatch.setattr(FLAGS, "JIT", False)
    assert ns["fn"] is numpy_fn
