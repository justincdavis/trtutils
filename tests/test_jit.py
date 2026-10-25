# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/_jit.py -- register_jit dispatch and the JIT switches."""

from __future__ import annotations

import numpy as np
import pytest

import trtutils
from trtutils._flags import FLAGS
from trtutils._jit import register_jit

numba = pytest.importorskip("numba")

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _restore_jit(monkeypatch):
    monkeypatch.setattr(FLAGS, "JIT", FLAGS.JIT)


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({}, id="defaults"),
        pytest.param({"nogil": True}, id="nogil"),
        pytest.param({"fastmath": True}, id="fastmath"),
        pytest.param({"parallel": True}, id="parallel"),
        pytest.param({"inline": "always"}, id="inline-always"),
        pytest.param({"nogil": True, "fastmath": True}, id="nogil-fastmath"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(np.float64, id="float64"),
        pytest.param(np.int64, id="int64"),
    ],
)
def test_register_jit_dispatch(options, dtype) -> None:
    """register_jit forwards its options and runs compiled only while JIT is on, same outputs."""

    @register_jit(**options)
    def remap(boxes: np.ndarray, pad: float, ratio: float) -> np.ndarray:
        out = (boxes - pad) / ratio
        return np.clip(out, 0, None)

    dispatcher = remap.jitted
    assert isinstance(dispatcher, numba.core.registry.CPUDispatcher)
    target = dispatcher.targetoptions
    assert target["nopython"] is True
    for name, value in options.items():
        assert target[name] == value

    boxes = (np.arange(40).reshape(10, 4) * 7 - 30).astype(dtype)
    expected = remap.py_func(boxes, 5.0, 0.5)

    trtutils.disable_jit()
    plain = remap(boxes, 5.0, 0.5)
    assert dispatcher.signatures == []

    trtutils.enable_jit()
    compiled = remap(boxes, 5.0, 0.5)
    assert len(dispatcher.signatures) == 1

    for out in (plain, compiled):
        assert out.dtype == expected.dtype
        np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    "start_on",
    [
        pytest.param(True, id="start-on"),
        pytest.param(False, id="start-off"),
    ],
)
def test_jit_context_restores_state(start_on) -> None:
    """The JIT context manager enables JIT inside the block and restores the prior state."""
    if start_on:
        trtutils.enable_jit()
    else:
        trtutils.disable_jit()

    with trtutils.JIT:
        assert FLAGS.JIT is True
    assert FLAGS.JIT is start_on
