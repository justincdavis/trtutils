# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for dynamic-shape execution: the engine runs at the shapes of the given Buffers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from trtutils import TRTEngine
from trtutils.core import Buffer, MemoryLocation, stream_synchronize

if TYPE_CHECKING:
    from collections.abc import Generator

_MAX_BATCH = 4


@pytest.fixture
def dynamic_engine(dynamic_engine_path) -> Generator[TRTEngine, None, None]:
    """A dynamic-batch (1..4) engine without CUDA graphs."""
    eng = TRTEngine(dynamic_engine_path, warmup=False, cuda_graph=False)
    yield eng
    del eng


@pytest.fixture
def static_engine(engine_path) -> Generator[TRTEngine, None, None]:
    """The batch-1 static version of the same network, used as a reference."""
    eng = TRTEngine(engine_path, warmup=False, cuda_graph=False)
    yield eng
    del eng


def _batch(engine: TRTEngine, n: int, seed: int = 0) -> np.ndarray:
    shape, dtype = engine.input_spec[0]
    rng = np.random.default_rng(seed)
    return rng.random(size=(n, *shape[1:])).astype(dtype)


def _reference(static_engine: TRTEngine, batch: np.ndarray) -> list[np.ndarray]:
    """Run each image through the static batch-1 engine and stack the outputs."""
    per_image = [
        static_engine.execute([Buffer.wrap(batch[i : i + 1].copy())]) for i in range(len(batch))
    ]
    return [np.concatenate([out[o] for out in per_image]) for o in range(len(per_image[0]))]


def test_engine_reports_dynamic_batch(dynamic_engine) -> None:
    """The engine reports a dynamic batch and its max profile batch size."""
    assert dynamic_engine.is_dynamic_batch
    assert dynamic_engine.batch_size == _MAX_BATCH


@pytest.mark.parametrize("sizes", [pytest.param((4, 2, 4, 1, 3), id="4-2-4-1-3")])
def test_execute_runs_submitted_batch(dynamic_engine, static_engine, sizes) -> None:
    """execute() runs at the Buffer's batch and returns outputs of that batch."""
    for seed, n in enumerate(sizes):
        batch = _batch(dynamic_engine, n, seed)
        outputs = dynamic_engine.execute([Buffer.wrap(batch)])
        assert all(out.shape[0] == n for out in outputs)
        assert dynamic_engine.active_input_shapes[0][0] == n
        for out, ref in zip(outputs, _reference(static_engine, batch)):
            np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_device_input_follows_its_own_shape(dynamic_engine) -> None:
    """A device input runs at its own shape, not the shape a previous call left behind."""
    dynamic_engine.execute([Buffer.wrap(_batch(dynamic_engine, 1))])
    batch = _batch(dynamic_engine, 3, seed=7)
    outputs = dynamic_engine.raw_exec([Buffer.from_array(batch, MemoryLocation.DEVICE)])
    stream_synchronize(dynamic_engine.stream)
    assert all(out.shape[0] == 3 for out in outputs)
    expected = dynamic_engine.execute([Buffer.wrap(batch)])
    for out, exp in zip(outputs, expected):
        np.testing.assert_array_equal(out.numpy(), exp)


@pytest.mark.parametrize(
    "n", [pytest.param(0, id="empty"), pytest.param(_MAX_BATCH + 1, id="over-max")]
)
def test_batch_outside_profile_raises(dynamic_engine, n) -> None:
    """Batches outside the optimization profile are rejected before touching the context."""
    before = dynamic_engine.active_input_shapes
    with pytest.raises(ValueError, match="between"):
        dynamic_engine.execute([Buffer.wrap(_batch(dynamic_engine, n))])
    assert dynamic_engine.active_input_shapes == before


@pytest.mark.cuda_graph
def test_graph_only_replays_at_full_shape(dynamic_engine_path) -> None:
    """The CUDA graph is captured at the full shape and bypassed at partial shapes."""
    eng = TRTEngine(dynamic_engine_path, warmup=False, cuda_graph=True)
    try:
        if eng._cuda_graph is None:
            pytest.skip("CUDA graph not enabled")
        full = _batch(eng, _MAX_BATCH, seed=1)
        partial = _batch(eng, 2, seed=2)
        first = eng.execute([Buffer.wrap(full)])
        assert eng._cuda_graph.is_captured
        out_partial = eng.execute([Buffer.wrap(partial)])
        assert all(out.shape[0] == 2 for out in out_partial)
        # back at the full shape: a plain enqueue first, then graph replays
        for _ in range(2):
            again = eng.execute([Buffer.wrap(full)])
            for a, f in zip(again, first):
                np.testing.assert_array_equal(a, f)
        assert eng._cuda_graph.is_captured
    finally:
        del eng
