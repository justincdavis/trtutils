# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for trtutils._jit.register -- import-time choice, rebound when flags change."""

from __future__ import annotations

import pytest

from trtutils._flags import FLAGS
from trtutils._jit import register

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
    register(ns, "fn", cv2=cv2_fn, numba=numba_fn, numpy=numpy_fn)
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
    register(ns, "fn", numba=numba_fn, numpy=numpy_fn)
    assert ns["fn"] is numba_fn
    monkeypatch.setattr(FLAGS, "JIT", False)
    assert ns["fn"] is numpy_fn


def test_register_compiles_when_jit_keywords_are_set(monkeypatch) -> None:
    """Jit keywords compile the numba function once and reuse that dispatcher."""
    pytest.importorskip("numba")
    monkeypatch.setattr(FLAGS, "FOUND_CV2", False)
    monkeypatch.setattr(FLAGS, "FOUND_NUMBA", True)
    monkeypatch.setattr(FLAGS, "JIT", True)

    def add(left: int, right: int) -> int:
        return left + right

    ns: dict[str, object] = {}
    register(ns, "add", numba=add, numpy=add)
    # no jit keywords: the plain function is the numba choice
    assert ns["add"] is add

    ns2: dict[str, object] = {}
    register(ns2, "add", numba=add, numpy=add, parallel=False)
    assert ns2["add"] is not add
    assert ns2["add"](1, 2) == 3
    compiled = ns2["add"]
    monkeypatch.setattr(FLAGS, "JIT", False)
    assert ns2["add"] is add
    monkeypatch.setattr(FLAGS, "JIT", True)
    assert ns2["add"] is compiled
