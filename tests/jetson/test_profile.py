# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for trtutils.jetson.profile_engine."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("jetsontools")

from trtutils import TRTEngine
from trtutils.jetson._profile import JetsonLayerProfiler, profile_engine

pytestmark = pytest.mark.jetson


def test_jetson_profile_runs_without_graph_and_restores_engine(engine_path) -> None:
    """A graph-enabled engine is profiled uncaptured, then handed back with its graph flag."""
    iterations = 500
    engine = TRTEngine(engine_path, cuda_graph=True, warmup=False)
    graph_enabled = engine.cuda_graph_enabled
    seen: list[bool] = []
    real_mock_execute = engine.mock_execute

    def spy(*args: object, **kwargs: object) -> object:
        seen.append(engine.cuda_graph_enabled)
        return real_mock_execute(*args, **kwargs)

    with patch.object(engine, "mock_execute", side_effect=spy):
        result = profile_engine(engine, iterations=iterations, warmup_iterations=2)

    assert seen
    assert not any(seen)
    assert result.layers
    assert all(len(layer.raw) == iterations for layer in result.layers)
    assert engine.cuda_graph_enabled == graph_enabled
    assert not isinstance(engine.context.profiler, JetsonLayerProfiler)
