# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/builder/hooks/_common.py -- shared hook utilities."""

from __future__ import annotations

import pytest

from trtutils.builder.hooks._common import make_plugin_field


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("test_int", 42, id="int"),
        pytest.param("test_float", 0.5, id="float"),
        pytest.param("test_list", [1, 2, 3], id="list_int"),
        pytest.param("test_list_f", [1.0, 2.0], id="list_float"),
    ],
)
def test_make_plugin_field(name, value) -> None:
    """Plugin field is created with correct name for each value type."""
    field = make_plugin_field(name, value)
    assert field.name == name
