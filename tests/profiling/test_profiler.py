# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for profile_engine."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils import TRTEngine
from trtutils._log import LOG
from trtutils.compat._libs import trt
from trtutils.profiling import profile_engine
from trtutils.profiling._profiler import LayerProfiler

pytestmark = pytest.mark.gpu


class _CountingProfiler(trt.IProfiler):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def report_layer_time(self, layer_name: str, ms: float) -> None:
        self.calls += 1


def test_profile_engine_with_cuda_graph_samples_every_iteration(engine_path) -> None:
    """A graph-enabled engine is still profiled on every iteration."""
    iterations = 20
    engine = TRTEngine(engine_path, cuda_graph=True, warmup=False)
    result = profile_engine(engine, iterations=iterations)
    assert result.layers
    assert all(len(layer.raw) == iterations for layer in result.layers)
    assert all(t > 0 for t in result.total_time.raw)


@pytest.mark.parametrize(
    "own_profiler",
    [
        pytest.param(False, id="no-profiler"),
        pytest.param(True, id="caller-profiler"),
    ],
)
def test_profile_engine_hands_back_engine(engine_path, own_profiler) -> None:
    """Profiling logs no TensorRT error and leaves a caller's engine running without its profiler."""
    # graph replays do not call the profiler
    engine = TRTEngine(engine_path, cuda_graph=False, warmup=False)
    previous = _CountingProfiler() if own_profiler else None
    if previous is not None:
        engine.context.profiler = previous
    inputs = engine.get_random_input()
    expected = engine.execute(inputs)

    with LOG.capture_errors() as errors:
        profile_engine(engine_path, iterations=5)
        result = profile_engine(engine, iterations=5)
        outputs = engine.execute(inputs)
    assert errors == []
    assert result.layers
    for output, ref in zip(outputs, expected):
        np.testing.assert_array_equal(output, ref)

    attached = engine.context.profiler
    assert not isinstance(attached, LayerProfiler)
    if previous is not None:
        assert attached is previous
        calls = previous.calls
        engine.execute(inputs)
        assert previous.calls > calls
