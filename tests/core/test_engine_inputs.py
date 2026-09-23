# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
CPU tests for how TRTEngine binds Buffer inputs.

A TRTEngine is assembled around a fake TensorRT execution context and the
host-memory CUDA fake from ``test_buffer``, so the input validation, shape
tracking, staging, and CUDA graph selection logic run without a GPU.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from trtutils import _engine
from trtutils._engine import TRTEngine
from trtutils.core import _bindings
from trtutils.core._buffer import Buffer, MemoryLocation

from .test_buffer import fake_cuda  # noqa: F401

if TYPE_CHECKING:
    from typing_extensions import Self

_MAX_BATCH = 4
_FEATURES = 3
_OUT_FEATURES = 2


class _FakeContext:
    """Records shape/address updates; the output batch follows the input batch."""

    def __init__(self) -> None:
        self.shapes: dict[str, tuple[int, ...]] = {"input": (_MAX_BATCH, _FEATURES)}
        self.addresses: dict[str, int] = {}
        self.shape_calls: list[tuple[int, ...]] = []
        self.enqueues = 0
        self.accept_shapes = True

    def set_input_shape(self, name: str, shape: tuple[int, ...]) -> bool:
        self.shape_calls.append(tuple(shape))
        if not self.accept_shapes:
            return False
        self.shapes[name] = tuple(shape)
        return True

    def set_tensor_address(self, name: str, address: int) -> None:
        self.addresses[name] = address

    def get_tensor_shape(self, _name: str) -> tuple[int, ...]:
        return (self.shapes["input"][0], _OUT_FEATURES)

    def execute_async_v3(self, _stream: object) -> bool:
        self.enqueues += 1
        return True


class _FakeGraph:
    def __init__(self) -> None:
        self.is_captured = False
        self.captures = 0
        self.launches = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.captures += 1
        self.is_captured = True

    def launch(self) -> None:
        self.launches += 1

    def invalidate(self) -> None:
        self.is_captured = False


def _make_engine(*, cuda_graph: bool = False) -> tuple[TRTEngine, _FakeContext]:
    """Assemble a dynamic-batch (1..4, 3) -> (batch, 2) engine without TensorRT."""
    inp = _bindings.create_binding(
        np.zeros((_MAX_BATCH, _FEATURES), np.float32), name="input", is_input=True
    )
    out = _bindings.create_binding(
        np.zeros((_MAX_BATCH, _OUT_FEATURES), np.float32), bind_id=1, name="output"
    )
    context = _FakeContext()
    eng = TRTEngine.__new__(TRTEngine)
    eng._name = "fake"
    eng._verbose = False
    eng._nvtx_tags = {}
    eng._device_guard = contextlib.nullcontext()
    eng._stream = None
    eng._context = context
    eng._async_v3 = True
    eng._inputs = [inp]
    eng._outputs = [out]
    eng._input_engine_shapes = [(-1, _FEATURES)]
    eng._input_min_shapes = [(1, _FEATURES)]
    eng._input_max_shapes = [(_MAX_BATCH, _FEATURES)]
    eng._input_shapes = [(_MAX_BATCH, _FEATURES)]
    eng._input_addresses = [inp.allocation]
    eng._output_shapes = [(_MAX_BATCH, _OUT_FEATURES)]
    eng._shapes_changed = False
    eng._v2_pointers = []
    eng._cuda_graph = _FakeGraph() if cuda_graph else None  # ty: ignore[invalid-assignment]
    return eng, context


@pytest.fixture
def no_sync(monkeypatch) -> None:
    """Skip stream synchronization, which the fake runtime does not provide."""
    monkeypatch.setattr(_engine, "stream_synchronize", lambda _stream: None)


def _host(n: int) -> Buffer:
    return Buffer.wrap(np.arange(n * _FEATURES, dtype=np.float32).reshape(n, _FEATURES))


pytestmark = [pytest.mark.cpu, pytest.mark.usefixtures("fake_cuda", "no_sync")]


def test_host_input_is_staged_into_binding(fake_cuda) -> None:  # noqa: F811
    """Host inputs are copied into the engine binding, which stays bound."""
    eng, context = _make_engine()
    eng.execute([_host(_MAX_BATCH)])
    assert eng._input_addresses == [eng._inputs[0].allocation]
    assert context.shape_calls == []
    np.testing.assert_array_equal(eng._inputs[0].device.numpy(), _host(_MAX_BATCH).array)


def test_partial_batch_sets_shape_and_output_shape() -> None:
    """A smaller batch sets the context shape and shrinks the returned outputs."""
    eng, context = _make_engine()
    outputs = eng.execute([_host(2)])
    assert context.shapes["input"] == (2, _FEATURES)
    assert eng.active_output_shapes == [(2, _OUT_FEATURES)]
    assert outputs[0].shape == (2, _OUT_FEATURES)


def test_unchanged_shape_is_not_reset() -> None:
    """The context shape is only set when the submitted shape changes."""
    eng, context = _make_engine()
    eng.execute([_host(2)])
    eng.execute([_host(2)])
    assert context.shape_calls == [(2, _FEATURES)]


def test_aligned_device_input_is_bound_in_place(fake_cuda) -> None:  # noqa: F811
    """Aligned device inputs are bound directly without an input copy."""
    eng, context = _make_engine()
    device = Buffer.from_array(_host(3).array, MemoryLocation.DEVICE)
    fake_cuda.copies.clear()
    eng.execute([device])
    assert context.addresses["input"] == device.ptr
    # only the output D2H, no input copy
    assert [kind for kind, _ in fake_cuda.copies] == ["D2H"]


def test_misaligned_device_input_is_staged(fake_cuda) -> None:  # noqa: F811
    """Device inputs TensorRT cannot address directly are copied into the binding."""
    eng, context = _make_engine()
    backing = Buffer.empty((_MAX_BATCH * _FEATURES + 1,), np.float32, MemoryLocation.DEVICE)
    misaligned = backing[1:].view((2, _FEATURES))
    fake_cuda.copies.clear()
    eng.execute([misaligned])
    assert context.addresses.get("input", eng._inputs[0].allocation) == eng._inputs[0].allocation
    assert [kind for kind, _ in fake_cuda.copies] == ["D2D", "D2H"]


def test_raw_exec_runs_at_its_own_shape_after_execute() -> None:
    """A device input after a smaller execute() call runs at its own shape (stale-shape regression)."""
    eng, context = _make_engine()
    eng.execute([_host(1)])
    device = Buffer.from_array(_host(3).array, MemoryLocation.DEVICE)
    outputs = eng.raw_exec([device])
    assert context.shapes["input"] == (3, _FEATURES)
    assert outputs[0].is_device
    assert outputs[0].shape == (3, _OUT_FEATURES)
    assert outputs[0].ptr == eng._outputs[0].allocation


@pytest.mark.parametrize("n", [0, _MAX_BATCH + 1], ids=["empty", "over-max"])
def test_shape_outside_profile_raises_without_side_effects(n) -> None:
    """Out-of-profile shapes raise before the context is touched."""
    eng, context = _make_engine()
    with pytest.raises(ValueError, match="between"):
        eng.execute([Buffer.wrap(np.zeros((n, _FEATURES), np.float32))])
    assert context.shape_calls == []
    assert eng.active_input_shapes == [(_MAX_BATCH, _FEATURES)]


def test_wrong_rank_raises() -> None:
    """A shape of the wrong rank is rejected."""
    eng, _ = _make_engine()
    with pytest.raises(ValueError, match="shape"):
        eng.execute([Buffer.wrap(np.zeros((2, _FEATURES, 1), np.float32))])


def test_rejected_shape_raises() -> None:
    """A shape TensorRT refuses surfaces as ValueError instead of being ignored."""
    eng, context = _make_engine()
    context.accept_shapes = False
    with pytest.raises(ValueError, match="rejected"):
        eng.execute([_host(2)])


def test_dtype_and_count_are_checked() -> None:
    """Wrong dtypes, input counts, and non-Buffer inputs are rejected."""
    eng, _ = _make_engine()
    with pytest.raises(ValueError, match="dtype"):
        eng.execute([Buffer.wrap(np.zeros((2, _FEATURES), np.float64))])
    with pytest.raises(ValueError, match="expects 1 inputs"):
        eng.execute([_host(2), _host(2)])
    with pytest.raises(TypeError, match=r"Buffer\.wrap"):
        eng.execute([np.zeros((2, _FEATURES), np.float32)])  # ty: ignore[invalid-argument-type]


def test_cuda_graph_only_at_full_engine_bindings() -> None:
    """The engine graph replays only for its own bindings at full shape."""
    eng, context = _make_engine(cuda_graph=True)
    graph = eng._cuda_graph
    assert isinstance(graph, _FakeGraph)

    eng.execute([_host(_MAX_BATCH)])  # plain run, then capture for later calls
    assert (context.enqueues, graph.captures, graph.launches) == (2, 1, 0)

    eng.execute([_host(_MAX_BATCH)])  # replay
    assert (context.enqueues, graph.launches) == (2, 1)

    eng.execute([_host(2)])  # partial shape: plain enqueue
    assert (context.enqueues, graph.launches) == (3, 1)

    eng.execute([_host(_MAX_BATCH)])  # shape changed back: one plain enqueue first
    assert (context.enqueues, graph.launches) == (4, 1)

    eng.execute([_host(_MAX_BATCH)])  # then the graph again
    assert (context.enqueues, graph.launches) == (4, 2)

    device = Buffer.from_array(_host(_MAX_BATCH).array, MemoryLocation.DEVICE)
    eng.execute([device])  # foreign address: plain enqueue, graph untouched
    assert (context.enqueues, graph.launches, graph.captures) == (5, 2, 1)


def test_raw_exec_never_uses_the_graph() -> None:
    """raw_exec always enqueues directly so it can be captured by callers."""
    eng, context = _make_engine(cuda_graph=True)
    eng.raw_exec([_host(_MAX_BATCH)])
    eng.raw_exec([_host(_MAX_BATCH)])
    graph = eng._cuda_graph
    assert isinstance(graph, _FakeGraph)
    assert (context.enqueues, graph.captures, graph.launches) == (2, 0, 0)


def test_stage_inputs_returns_device_buffers(fake_cuda) -> None:  # noqa: F811
    """stage_inputs passes device Buffers through and copies host ones."""
    eng, _ = _make_engine()
    device = Buffer.from_array(_host(2).array, MemoryLocation.DEVICE)
    assert eng.stage_inputs([device])[0] is device
    staged = eng.stage_inputs([_host(2)])[0]
    assert staged.is_device
    assert staged.ptr == eng._inputs[0].allocation
    assert staged.shape == (2, _FEATURES)
