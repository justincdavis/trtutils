# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for pyproject.toml extras -- TensorRT bounds and self-references."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

pytestmark = pytest.mark.cpu

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
_EXTRAS = tomllib.loads(_PYPROJECT.read_text())["project"]["optional-dependencies"]


@pytest.mark.parametrize("extra", ["cu12", "cu13"])
def test_tensorrt_excludes_11(extra):
    """TensorRT 11 removed IInt8EntropyCalibrator2 and EXPLICIT_BATCH."""
    reqs = [Requirement(r) for r in _EXTRAS[extra] if r.startswith("tensorrt_")]
    assert len(reqs) == 1
    assert not reqs[0].specifier.contains("11.0.0")
    assert not reqs[0].specifier.contains("11.4.0.106")
    assert reqs[0].specifier.contains("10.13.0")


@pytest.mark.parametrize(
    ("extra", "req"),
    [
        pytest.param(extra, req, id=f"{extra}-{req}")
        for extra, reqs in _EXTRAS.items()
        for req in reqs
        if req.startswith("trtutils[")
    ],
)
def test_self_reference_is_defined_extra(extra, req):
    """Every trtutils[...] self-reference names a defined extra."""
    name = re.match(r"trtutils\[(\w+)\]", req).group(1)
    assert name in _EXTRAS
