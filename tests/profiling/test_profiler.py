# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for profile_engine."""

from __future__ import annotations

import pytest

from trtutils import TRTEngine
from trtutils.profiling import profile_engine

pytestmark = pytest.mark.gpu


def test_profile_engine_with_cuda_graph_samples_every_iteration(engine_path) -> None:
    """A graph-enabled engine is still profiled on every iteration."""
    iterations = 20
    engine = TRTEngine(engine_path, cuda_graph=True, warmup=False)
    result = profile_engine(engine, iterations=iterations)
    assert result.layers
    assert all(len(layer.raw) == iterations for layer in result.layers)
    assert all(t > 0 for t in result.total_time.raw)
