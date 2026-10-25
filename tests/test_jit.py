# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/_jit.py -- register_jit dispatch and the jitted postprocess cores."""

from __future__ import annotations

import numpy as np
import pytest

import trtutils
from trtutils._flags import FLAGS
from trtutils._jit import register_jit
from trtutils.image.postprocessors._hand_interaction import _remap_boxes_core

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(not FLAGS.FOUND_NUMBA, reason="numba is not installed"),
]


@pytest.fixture(autouse=True)
def _restore_jit(monkeypatch):
    monkeypatch.setattr(FLAGS, "JIT", FLAGS.JIT)


@pytest.mark.parametrize(
    "num",
    [
        pytest.param(100, id="boxes-100"),
        pytest.param(300, id="boxes-300"),
        pytest.param(1, id="boxes-1"),
        pytest.param(0, id="boxes-empty"),
    ],
)
def test_remap_boxes_core_jit_matches_python(num) -> None:
    """_remap_boxes_core runs compiled only while JIT is on, with identical outputs."""
    core = _remap_boxes_core
    rng = np.random.default_rng(num)
    boxes = rng.random((num, 4), dtype=np.float32) * 700 - 20
    scores = rng.random(num, dtype=np.float32)
    args = (boxes, scores, (0.5, 0.75), (0.0, 80.0), 0.25)
    expected = core.__wrapped__(*args)

    with trtutils.JIT:
        compiled = core(*args)
    num_sigs = len(core.jitted.signatures)
    assert num_sigs >= 1

    plain = core(*args)
    assert len(core.jitted.signatures) == num_sigs

    for got in (compiled, plain):
        assert len(got) == len(expected)
        for out, ref in zip(got, expected):
            assert out.dtype == ref.dtype
            np.testing.assert_array_equal(out, ref)


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
def test_register_jit_options(options) -> None:
    """register_jit forwards its options to numba and enable/disable_jit switch at any time."""

    @register_jit(**options)
    def scale(x: np.ndarray) -> np.ndarray:
        return x * 2.0

    x = np.arange(8, dtype=np.float64)
    target = scale.jitted.targetoptions
    assert target["nopython"] is True
    for name, value in options.items():
        assert target[name] == value

    trtutils.disable_jit()
    np.testing.assert_array_equal(scale(x), x * 2.0)
    assert scale.jitted.signatures == []

    trtutils.enable_jit()
    np.testing.assert_array_equal(scale(x), x * 2.0)
    assert len(scale.jitted.signatures) == 1
