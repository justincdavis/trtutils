# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for TRTEngineInterface contract tested through TRTEngine."""

from __future__ import annotations

import pytest

from trtutils import TRTEngine
from trtutils._flags import FLAGS


@pytest.mark.parametrize(
    ("pagelocked_mem", "expected"),
    [
        pytest.param(None, True, id="pagelocked-default"),
        pytest.param(False, False, id="pagelocked-false"),
    ],
)
def test_pagelocked_mem(engine_path, pagelocked_mem, expected) -> None:
    """pagelocked_mem parameter is stored correctly."""
    eng = TRTEngine(engine_path, warmup=False, pagelocked_mem=pagelocked_mem)
    assert eng.pagelocked_mem is expected
    del eng


def test_unified_mem_default(engine) -> None:
    """unified_mem=None defaults to FLAGS.IS_JETSON."""
    assert engine.unified_mem == FLAGS.IS_JETSON


def test_unified_mem_explicit_true(engine_path) -> None:
    """unified_mem=True can be explicitly set."""
    eng = TRTEngine(engine_path, warmup=False, unified_mem=True)
    assert eng.unified_mem is True
    del eng


def test_name_from_path(engine, engine_path) -> None:
    """Name property is the stem of the engine path."""
    assert engine.name == engine_path.stem


def test_backend_invalid_raises(engine_path) -> None:
    """Invalid backend raises ValueError."""
    with pytest.raises(ValueError, match="Invalid backend"):
        TRTEngine(engine_path, backend="invalid")


def test_del_frees_bindings(engine_path) -> None:
    """__del__ frees bindings without error."""
    eng = TRTEngine(engine_path, warmup=False)
    del eng


def test_del_double_delete(engine_path) -> None:
    """Double deletion does not crash."""
    eng = TRTEngine(engine_path, warmup=False)
    eng.__del__()
    del eng


def test_del_deletes_context_engine(engine_path) -> None:
    """__del__ removes _context and _engine attributes."""
    eng = TRTEngine(engine_path, warmup=False)
    eng.__del__()
    assert not hasattr(eng, "_context")
    assert not hasattr(eng, "_engine")
