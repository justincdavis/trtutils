# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Core test fixtures -- cache dir patching and GPU fixtures."""

from __future__ import annotations

import os

import pytest

from trtutils.core import cache

_CPU_ONLY = os.environ.get("TRTUTILS_IGNORE_MISSING_CUDA", "0") == "1"


@pytest.fixture
def patched_cache_dir(tmp_path, monkeypatch):
    """Provide a temporary cache directory with get_cache_dir patched."""
    cache_dir = tmp_path / "_engine_cache"
    cache_dir.mkdir()
    monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)
    return cache_dir


if not _CPU_ONLY:
    from tests.core._gpu_fixtures import *  # noqa: F403
