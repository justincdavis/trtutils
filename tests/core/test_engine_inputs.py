# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
CPU tests for how TRTEngine binds Buffer inputs.

A TRTEngine is assembled around a fake TensorRT execution context and the
host-memory CUDA fake from ``conftest``, so the input validation, shape
tracking, staging, and CUDA graph selection logic run without a GPU.
"""

from __future__ import annotations

import contextlib
import gc
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pytest

from trtutils import _engine
from trtutils._engine import TRTEngine
from trtutils.core import _bindings, _device
from trtutils.core._buffer import Buffer, MemoryLocation

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
        self.reject_enqueue_at: int | None = None

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
        return self.enqueues != self.reject_enqueue_at


class _FakeGraph:
    """Stands in for CUDAGraph; every instance is recorded in ``created``."""

    created: ClassVar[list[_FakeGraph]] = []
    fail_capture: ClassVar[bool] = False

    def __init__(self, _stream: object = None) -> None:
        self.is_captured = False
        self.launches = 0
        _FakeGraph.created.append(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.is_captured = not _FakeGraph.fail_capture

    def launch(self) -> None:
        self.launches += 1

    def invalidate(self) -> None:
        self.is_captured = False


def _make_engine(
    *, cuda_graph: bool = False, shape_contexts: int | None = None
) -> tuple[TRTEngine, _FakeContext]:
    """Assemble a dynamic-batch (1..4, 3) -> (batch, 2) engine without TensorRT."""
    inp = _bindings.create_binding(
        np.zeros((_MAX_BATCH, _FEATURES), np.float32),
        name="input",
        is_input=True,
        min_shape=(1, _FEATURES),
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
    eng._allocations = [inp.allocation, out.allocation]
    eng._init_io_state(cuda_graph=cuda_graph, shape_contexts=shape_contexts)
    context.shape_calls.clear()  # only record the calls the tests make
    return eng, context


@pytest.fixture
def fake_graphs(monkeypatch) -> list[_FakeGraph]:
    """Replace CUDAGraph in the engine; returns every graph the engine creates."""
    _FakeGraph.created = []
    _FakeGraph.fail_capture = False
    monkeypatch.setattr(_engine, "CUDAGraph", _FakeGraph)
    return _FakeGraph.created


def _host(n: int) -> Buffer:
    return Buffer.wrap(np.arange(n * _FEATURES, dtype=np.float32).reshape(n, _FEATURES))


pytestmark = [pytest.mark.cpu, pytest.mark.usefixtures("fake_cuda")]


def test_host_input_is_staged_into_binding(fake_cuda) -> None:
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


def test_aligned_device_input_is_bound_in_place(fake_cuda) -> None:
    """Aligned device inputs are bound directly without an input copy."""
    eng, context = _make_engine()
    device = Buffer.from_array(_host(3).array, MemoryLocation.DEVICE)
    fake_cuda.copies.clear()
    eng.execute([device])
    assert context.addresses["input"] == device.ptr
    # only the output D2H, no input copy
    assert [kind for kind, _ in fake_cuda.copies] == ["D2H"]


def test_misaligned_device_input_is_staged(fake_cuda) -> None:
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
    not_a_buffer: Any = np.zeros((2, _FEATURES), np.float32)
    with pytest.raises(TypeError, match=r"Buffer\.wrap"):
        eng.execute([not_a_buffer])


def _launches(graphs: list[_FakeGraph]) -> int:
    return sum(graph.launches for graph in graphs)


def test_cuda_graph_per_input_configuration(fake_graphs) -> None:
    """Each input configuration gets its own graph once it repeats, dynamic shapes included."""
    eng, context = _make_engine(cuda_graph=True)

    eng.execute([_host(_MAX_BATCH)])  # first sighting: plain run only
    assert (context.enqueues, len(fake_graphs)) == (1, 0)
    eng.execute([_host(_MAX_BATCH)])  # second sighting: plain run, then capture
    assert (context.enqueues, len(fake_graphs)) == (3, 1)
    eng.execute([_host(_MAX_BATCH)])  # replay
    assert (context.enqueues, _launches(fake_graphs)) == (3, 1)

    # a partial batch is a configuration of its own; the call that changes
    # the shapes only runs plain, so it is captured on its second run after that
    eng.execute([_host(2)])
    eng.execute([_host(2)])
    assert (context.enqueues, len(fake_graphs)) == (5, 1)
    eng.execute([_host(2)])
    assert (context.enqueues, len(fake_graphs)) == (7, 2)
    eng.execute([_host(2)])
    assert (context.enqueues, _launches(fake_graphs)) == (7, 2)

    # back at the full shape: one plain run after the shape change, then replay
    eng.execute([_host(_MAX_BATCH)])
    assert (context.enqueues, _launches(fake_graphs)) == (8, 2)
    eng.execute([_host(_MAX_BATCH)])
    assert (context.enqueues, _launches(fake_graphs), len(fake_graphs)) == (8, 3, 2)

    # a device input bound in place is keyed by its address
    device = Buffer.from_array(_host(_MAX_BATCH).array, MemoryLocation.DEVICE)
    for _ in range(3):
        eng.execute([device])
    assert (context.enqueues, _launches(fake_graphs), len(fake_graphs)) == (11, 4, 3)


def test_alternating_shapes_are_not_captured(fake_graphs) -> None:
    """Shapes that change every call could never replay a graph, so none is captured."""
    eng, context = _make_engine(cuda_graph=True)
    for _ in range(4):
        eng.execute([_host(2)])
        eng.execute([_host(3)])
    assert (context.enqueues, len(fake_graphs)) == (8, 0)
    assert len(eng._seen_configs) == 0


def test_cuda_graph_cache_evicts_least_recently_used(fake_graphs, monkeypatch) -> None:
    """Beyond the cache limit the least recently used graph is invalidated and dropped."""
    monkeypatch.setattr(_engine, "_MAX_CUDA_GRAPHS", 2)
    eng, _ = _make_engine(cuda_graph=True)
    for batch in (4, 3, 2):
        for _ in range(3):  # a shape change, then two runs at unchanged shapes
            eng.execute([_host(batch)])
    assert len(fake_graphs) == 3
    assert not fake_graphs[0].is_captured  # batch 4, evicted
    assert all(graph.is_captured for graph in fake_graphs[1:])
    assert len(eng._cuda_graphs) == 2


def test_one_off_inputs_do_not_evict_captured_graphs(fake_graphs, monkeypatch) -> None:
    """Seen-once placeholders are dropped before captured graphs when the cache is full."""
    monkeypatch.setattr(_engine, "_MAX_CUDA_GRAPHS", 2)
    eng, _ = _make_engine(cuda_graph=True)
    eng.execute([_host(_MAX_BATCH)])
    eng.execute([_host(_MAX_BATCH)])  # captured
    # a burst of configurations that never repeat: fresh device buffers
    one_offs = [Buffer.from_array(_host(_MAX_BATCH).array, MemoryLocation.DEVICE) for _ in range(3)]
    for device in one_offs:
        eng.execute([device])
    assert fake_graphs[0].is_captured
    assert len(eng._cuda_graphs) == 1
    assert len(eng._seen_configs) == 2  # seen-once tracking is bounded on its own
    eng.execute([_host(_MAX_BATCH)])  # the graph captured earlier replays
    assert fake_graphs[0].launches == 1
    assert len(fake_graphs) == 1


def test_failed_capture_falls_back_without_graphs(fake_graphs) -> None:
    """A failed capture disables graphs; the call that captured still returns its outputs."""
    _FakeGraph.fail_capture = True
    eng, context = _make_engine(cuda_graph=True)
    for _ in range(2):
        outputs = eng.execute([_host(_MAX_BATCH)])
    assert outputs[0].shape == (_MAX_BATCH, _OUT_FEATURES)
    assert not eng._cuda_graph_enabled
    assert len(eng._cuda_graphs) == 0
    eng.execute([_host(_MAX_BATCH)])
    assert len(fake_graphs) == 1  # no further capture attempts
    assert context.enqueues == 4  # 3 real runs + 1 failed capture


def test_rejected_capture_enqueue_disables_graphs(fake_graphs) -> None:
    """A capture whose enqueue TensorRT refuses is discarded, never replayed."""
    eng, context = _make_engine(cuda_graph=True)
    context.reject_enqueue_at = 3  # run, run, capture
    for _ in range(3):
        eng.execute([_host(_MAX_BATCH)])
    assert not eng._cuda_graph_enabled
    assert len(eng._cuda_graphs) == 0
    assert not fake_graphs[0].is_captured
    assert context.enqueues == 4  # the third execute ran for real


def test_bound_device_input_outlives_the_caller(fake_cuda) -> None:
    """The engine keeps the Buffers it reads alive for graph_exec and async raw_exec."""
    eng, _ = _make_engine()
    device = Buffer.from_array(_host(3).array, MemoryLocation.DEVICE)
    ptr = device.ptr
    eng.raw_exec([device])
    del device
    gc.collect()
    assert ("device", ptr) not in fake_cuda.freed
    eng.raw_exec([_host(3)])
    gc.collect()
    assert ("device", ptr) in fake_cuda.freed


def test_freeing_an_output_keeps_the_engine_usable() -> None:
    """Returned outputs are views, so freeing one never frees the engine's binding."""
    eng, _ = _make_engine()
    for out in eng.raw_exec([_host(_MAX_BATCH)]):
        out.free()
    outputs = eng.execute([_host(_MAX_BATCH)])
    assert outputs[0].shape == (_MAX_BATCH, _OUT_FEATURES)


def test_rejected_enqueue_raises() -> None:
    """An enqueue TensorRT refuses raises instead of returning stale outputs."""
    eng, context = _make_engine()
    context.reject_enqueue_at = 1
    with pytest.raises(RuntimeError, match="failed to enqueue"):
        eng.execute([_host(2)])


def test_execute_restores_the_callers_device(monkeypatch) -> None:
    """execute() leaves the caller on its own CUDA device (the guard is never nested)."""
    current = {"device": 0}
    monkeypatch.setattr(_device, "get_device", lambda: current["device"])
    monkeypatch.setattr(_device, "set_device", lambda d: current.update(device=d))
    eng, _ = _make_engine()
    eng._device_guard = _device.Device(1)
    eng.execute([_host(2)])
    assert current["device"] == 0


def test_raw_exec_never_uses_the_graph(fake_graphs) -> None:
    """raw_exec always enqueues directly so it can be captured by callers."""
    eng, context = _make_engine(cuda_graph=True)
    for _ in range(3):
        eng.raw_exec([_host(_MAX_BATCH)])
    assert (context.enqueues, len(fake_graphs)) == (3, 0)


def _pooled(
    shape_contexts: int, *, fail_first: frozenset[int] = frozenset()
) -> tuple[TRTEngine, list[_FakeContext]]:
    """A pooled engine; the contexts it creates are listed in order after the first."""
    eng, first = _make_engine(cuda_graph=True, shape_contexts=shape_contexts)
    contexts = [first]

    def create() -> _FakeContext:
        context = _FakeContext()
        if len(contexts) in fail_first:
            context.reject_enqueue_at = 1
        contexts.append(context)
        return context

    eng._create_context = create  # type: ignore[method-assign]
    return eng, contexts


def test_no_pool_by_default(fake_graphs) -> None:
    """Without shape_contexts one context serves every shape, as before."""
    eng, context = _make_engine(cuda_graph=True)
    for batch in (4, 2, 3, 2):
        eng.execute([_host(batch)])
    assert eng._shape_contexts is None
    assert len(eng._parked) == 0
    assert eng._context is context
    assert len(context.shape_calls) == 3  # 2, 3, 2; the first batch is the max shape


@pytest.mark.parametrize("shape_contexts", [0, -1])
def test_pool_size_must_be_positive(shape_contexts) -> None:
    """A pool of fewer than one context is rejected."""
    with pytest.raises(ValueError, match="shape_contexts"):
        _make_engine(shape_contexts=shape_contexts)


@pytest.mark.parametrize(
    ("batches", "created"),
    [
        pytest.param([4, 4, 4], 0, id="max-shape-uses-first-context"),
        pytest.param([2, 2, 2], 1, id="one-shape"),
        pytest.param([4, 2, 3, 2, 3, 4], 2, id="cycle"),
        pytest.param([1, 2, 3, 1, 2, 3, 1, 2, 3], 3, id="all-new"),
    ],
)
def test_pool_creates_one_context_per_shape(fake_graphs, batches, created) -> None:
    """Each distinct input shape gets a context, which is reused and never changes shape."""
    eng, contexts = _pooled(8)
    for batch in batches:
        eng.execute([_host(batch)])
    assert len(contexts) - 1 == created
    assert len(eng._parked) + 1 == len(contexts)
    # a context only sees the one shape it was made for (plus its creation call)
    for context in contexts[1:]:
        assert len(context.shape_calls) == 2
        assert context.shape_calls[-1] == (context.shapes["input"])


def test_pool_properties_describe_the_last_run(fake_graphs) -> None:
    """Shapes and outputs follow the context of the most recent run."""
    eng, _ = _pooled(8)
    for batch in (2, 4, 3, 2, 3):
        outputs = eng.execute([_host(batch)])
        assert eng.active_input_shapes == [(batch, _FEATURES)]
        assert eng.active_output_shapes == [(batch, _OUT_FEATURES)]
        assert outputs[0].shape == (batch, _OUT_FEATURES)


def test_pool_contexts_share_bindings(fake_graphs) -> None:
    """Every context reads and writes the engine's own input and output allocations."""
    eng, contexts = _pooled(8)
    for batch in (2, 3, 4):
        eng.execute([_host(batch)])
    for context in contexts:
        assert context.addresses["input"] == eng._inputs[0].allocation
        assert context.addresses["output"] == eng._outputs[0].allocation


def test_pool_replays_graphs_per_context(fake_graphs) -> None:
    """Alternating shapes replay a graph per context, which one context could never do."""
    eng, contexts = _pooled(8)
    for _ in range(5):
        for batch in (4, 2):
            eng.execute([_host(batch)])
    assert len(fake_graphs) == 2
    # the first context (batch 4): plain, plain + capture, then replays
    # the second (batch 2): its shape-change run is plain and never counts toward a capture
    assert [c.enqueues for c in contexts] == [3, 4]
    assert [g.launches for g in fake_graphs] == [3, 2]


def test_pool_failed_first_run_keeps_its_shape_change(fake_graphs) -> None:
    """A context whose shape-change run was rejected still owes a plain run, whatever ran since."""
    eng, contexts = _pooled(8, fail_first=frozenset({1}))
    with pytest.raises(RuntimeError, match="failed to enqueue"):
        eng.execute([_host(2)])
    eng.execute([_host(4)])  # another context's plain run
    eng.execute([_host(2)])  # the plain run the shape change owes: not a sighting
    eng.execute([_host(2)])  # first sighting
    assert len(fake_graphs) == 0
    eng.execute([_host(2)])  # second sighting: capture
    assert len(fake_graphs) == 1
    assert contexts[1].enqueues == 5  # the rejected run, three plain runs, the capture


def test_pool_evicts_least_recently_used(fake_graphs) -> None:
    """Over the limit the least recently used context goes, its graphs invalidated."""
    eng, contexts = _pooled(2)
    for _ in range(3):
        eng.execute([_host(4)])  # first context, captured
    assert fake_graphs[0].is_captured
    eng.execute([_host(2)])
    assert len(contexts) == 2
    assert fake_graphs[0].is_captured  # both still fit
    eng.execute([_host(3)])  # evicts batch 4, least recently used
    assert not fake_graphs[0].is_captured
    assert len(contexts) == 3
    assert len(eng._parked) == 1
    eng.execute([_host(2)])  # batch 2 survived
    assert len(contexts) == 3
    eng.execute([_host(4)])  # batch 4 pays for a new context
    assert len(contexts) == 4
    assert eng.active_input_shapes == [(4, _FEATURES)]


def test_pool_of_one_rebuilds_on_every_shape_change(fake_graphs) -> None:
    """With a single context every change of shape evicts and creates one."""
    eng, contexts = _pooled(1)
    for batch in (2, 3, 2, 3):
        eng.execute([_host(batch)])
    assert len(contexts) == 5
    assert len(eng._parked) == 0
