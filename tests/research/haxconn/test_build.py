# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/haxconn/_build.py and the GPU time split -- on plain data."""

from __future__ import annotations

import pytest

from tests.research.haxconn.conftest import make_traits
from trtutils.research.haxconn import Schedule, build_engines
from trtutils.research.haxconn._characterize import _fill_unmatched

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    ("accelerators", "match"),
    [
        pytest.param(("gpu",), "expected 2 accelerators", id="short"),
        pytest.param(("gpu", "dla"), "cannot run on 'dla'", id="gpu-only-group"),
    ],
)
def test_build_engines_rejects_assignments_that_do_not_fit(tmp_path, accelerators, match) -> None:
    """An assignment from other traits raises instead of building a truncated mapping."""
    traits = {"a": make_traits([1.0, 1.0], [1.0, None])}
    schedule = Schedule({"a": accelerators}, {"a": 1.0}, 1.0, optimal=False)
    with pytest.raises(ValueError, match=match):
        build_engines({"a": tmp_path / "a.onnx"}, traits, schedule, tmp_path)


def test_fill_unmatched_gives_auto_named_layers_a_neighbour() -> None:
    """Engine layers naming no network layer join the layer before them, or after at the start."""
    matches = [[], [0, 1], [], [2], [], [4, 5]]
    assert _fill_unmatched(matches) == [[0], [0, 1], [1], [2], [2], [4, 5]]
    assert _fill_unmatched([[], []]) == [[], []]
