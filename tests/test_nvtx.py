# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/_nvtx.py -- NVTX context manager state restoration."""

from __future__ import annotations

import nvtx
import pytest

from trtutils._flags import FLAGS
from trtutils._nvtx import NVTX, disable_nvtx, enable_nvtx

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _restore_nvtx(monkeypatch):
    monkeypatch.setattr(FLAGS, "NVTX_ENABLED", FLAGS.NVTX_ENABLED)
    monkeypatch.setattr(nvtx, "push_range", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nvtx, "pop_range", lambda *_args, **_kwargs: None)


def test_nvtx_reused_instance_restores_state() -> None:
    """A reused NVTX instance restores whichever state was active at each entry."""
    scope = NVTX("scope")

    enable_nvtx()
    with scope:
        assert FLAGS.NVTX_ENABLED is True
    assert FLAGS.NVTX_ENABLED is True

    disable_nvtx()
    with scope:
        assert FLAGS.NVTX_ENABLED is True
    assert FLAGS.NVTX_ENABLED is False

    with scope:
        with scope:
            assert FLAGS.NVTX_ENABLED is True
        assert FLAGS.NVTX_ENABLED is True
    assert FLAGS.NVTX_ENABLED is False
