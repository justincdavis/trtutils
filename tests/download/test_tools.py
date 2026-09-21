# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/download/_tools.py -- handle_batch and handle_dynamic."""

from __future__ import annotations

import pytest

from trtutils.download._tools import handle_batch, handle_dynamic

pytestmark = pytest.mark.cpu


def test_handle_batch_none_defaults_to_one() -> None:
    """A None batch is treated as batch 1."""
    assert handle_batch(None, "model") == 1


def test_handle_batch_invalid_raises_value_error() -> None:
    """A batch size of 0 is rejected."""
    with pytest.raises(ValueError, match="positive"):
        handle_batch(0, "model")


def test_handle_batch_negative_raises_value_error() -> None:
    """A negative batch size is rejected."""
    with pytest.raises(ValueError, match="positive"):
        handle_batch(-1, "model")


def test_handle_batch_unsupported_above_one_raises() -> None:
    """An exporter that cannot honor batch > 1 raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="model"):
        handle_batch(2, "model", supported=False)


def test_handle_batch_unsupported_at_one_passes() -> None:
    """Batch 1 is always fine, even for an unsupported exporter."""
    assert handle_batch(1, "model", supported=False) == 1


def test_handle_batch_supported_passes_through() -> None:
    """A supported exporter returns the requested batch size unchanged."""
    assert handle_batch(8, "model", supported=True) == 8


def test_handle_dynamic_falsy_returns_false() -> None:
    """A falsy dynamic flag (None or False) returns False."""
    assert handle_dynamic(None, "model", dynamic=None) is False
    assert handle_dynamic(None, "model", dynamic=False) is False


def test_handle_dynamic_with_fixed_batch_raises_value_error() -> None:
    """Requesting dynamic export together with a fixed batch > 1 is rejected."""
    with pytest.raises(ValueError, match="model"):
        handle_dynamic(4, "model", dynamic=True)


def test_handle_dynamic_batch_one_is_fine() -> None:
    """Dynamic export combined with batch 1 (or None) is allowed."""
    assert handle_dynamic(1, "model", dynamic=True) is True


def test_handle_dynamic_unsupported_raises() -> None:
    """An exporter that cannot honor dynamic export raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="model"):
        handle_dynamic(None, "model", dynamic=True, supported=False)


def test_handle_dynamic_supported_passes_through() -> None:
    """A supported exporter returns True when dynamic export is requested."""
    assert handle_dynamic(None, "model", dynamic=True, supported=True) is True
