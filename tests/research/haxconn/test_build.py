# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/haxconn/_build.py and the GPU time split -- on plain data."""

from __future__ import annotations

import pytest

from tests.research.haxconn.conftest import make_traits
from trtutils.research.haxconn import Schedule, _build, build_engines, prebuild_engines
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


def test_prebuild_engines_builds_every_option_once(tmp_path, monkeypatch) -> None:
    """One engine per assignment within the limit; engines already built are kept."""
    built = []

    def build_one(_onnx, _traits, _accs, path, *_args: object, **_kwargs: object):
        path.touch()
        built.append(path.name)

    monkeypatch.setattr(_build, "_build_one", build_one)
    traits = {"a": make_traits([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]), "b": make_traits([1.0], [None])}
    models = {name: tmp_path / f"{name}.onnx" for name in traits}
    for path in models.values():
        path.write_bytes(b"")
    paths = prebuild_engines(models, traits, tmp_path)
    assert {name: len(p) for name, p in paths.items()} == {"a": 6, "b": 1}
    assert len(built) == 7
    assert all(path.exists() for p in paths.values() for path in p)
    prebuild_engines(models, traits, tmp_path)
    assert len(built) == 7
    with pytest.raises(ValueError, match="at least 0"):
        prebuild_engines(models, traits, tmp_path, max_transitions=-1)
