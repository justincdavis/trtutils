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
def test_pagelocked_mem(make_engine, pagelocked_mem, expected) -> None:
    """pagelocked_mem parameter is stored correctly."""
    eng = make_engine(pagelocked_mem=pagelocked_mem)
    assert eng.pagelocked_mem is expected


def test_unified_mem_default(engine) -> None:
    """unified_mem=None defaults to FLAGS.IS_JETSON."""
    assert engine.unified_mem == FLAGS.IS_JETSON


def test_unified_mem_explicit_true(make_engine) -> None:
    """unified_mem=True can be explicitly set."""
    eng = make_engine(unified_mem=True)
    assert eng.unified_mem is True


def test_name_from_path(engine, engine_path) -> None:
    """Name property is the stem of the engine path."""
    assert engine.name == engine_path.stem


def test_backend_invalid_raises(engine_path) -> None:
    """Invalid backend raises ValueError."""
    with pytest.raises(ValueError, match="Invalid backend"):
        TRTEngine(engine_path, backend="invalid")
