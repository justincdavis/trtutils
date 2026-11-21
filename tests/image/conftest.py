# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Image test fixtures, loaded from _gpu_fixtures.py unless TRTUTILS_IGNORE_MISSING_CUDA=1."""

from __future__ import annotations

from tests.conftest import _CPU_ONLY

if not _CPU_ONLY:
    from tests.image._gpu_fixtures import *  # noqa: F403
