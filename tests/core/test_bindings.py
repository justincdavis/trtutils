# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Tests for src/trtutils/core/_bindings.py -- Binding, create_binding, and allocate_bindings.

Tests taking the ``calls`` fixture run twice: on the host-memory CUDA fake
from ``conftest`` (``cpu``) and, with a GPU, on the real runtime. Both
record the memcpy and free calls Buffers make, so copy routing and memory
lifetimes are asserted the same way on each.
"""

from __future__ import annotations

import gc
import itertools
import math
import os
import re
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import DATA_DIR, SIMPLE_DYNAMIC_PROFILES
from trtutils._flags import FLAGS
from trtutils.compat._libs import trt
from trtutils.core import _bindings, _buffer, create_stream, destroy_stream, stream_synchronize
from trtutils.core._bindings import Binding, allocate_bindings, create_binding
from trtutils.core._buffer import DEVICE_ALIGNMENT, Buffer, MemoryLocation
from trtutils.core._engine import create_engine

_CPU_ONLY = os.environ.get("TRTUTILS_IGNORE_MISSING_CUDA", "0") == "1"
_F32 = np.dtype(np.float32)
_FORMAT = object()

MEMORY = [
    pytest.param({"pagelocked_mem": False}, id="pageable"),
    pytest.param({"pagelocked_mem": True}, id="pagelocked"),
    pytest.param({"pagelocked_mem": True, "unified_mem": True}, id="unified"),
]
SOURCES = [pytest.param(kind, id=kind) for kind in ("pageable", "pinned", "mapped", "device")]
LOCATIONS = [
    pytest.param(MemoryLocation.HOST, id="host"),
    pytest.param(MemoryLocation.DEVICE, id="device"),
]


@dataclass
class _Calls:
    """The runtime in use and the memcpy ``(kind, is_async)`` and free ``(memory, ptr)`` calls."""

    runtime: str
    copies: list[tuple[str, bool]] = field(default_factory=list)
    freed: list[tuple[str, int]] = field(default_factory=list)


@pytest.fixture(
    params=[
        pytest.param("fake", marks=pytest.mark.cpu, id="fake-cuda"),
        pytest.param("cuda", id="cuda"),
    ]
)
def calls(request, monkeypatch):
    """Run on the CUDA fake and the real runtime, recording memcpy and free calls."""
    if request.param == "fake":
        request.getfixturevalue("fake_cuda")
    elif _CPU_ONLY:
        pytest.skip("the real CUDA runtime needs a GPU")
    # drop garbage from earlier tests so its frees are not recorded here
    gc.collect()
    cudart = _buffer.cudart
    kinds = {
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice: "H2D",
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost: "D2H",
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice: "D2D",
    }
    recorded = _Calls(request.param)
    memcpy, memcpy_async = cudart.cudaMemcpy, cudart.cudaMemcpyAsync
    free, free_host = cudart.cudaFree, cudart.cudaFreeHost

    def _memcpy(dst, src, nbytes, kind):
        recorded.copies.append((kinds[kind], False))
        return memcpy(dst, src, nbytes, kind)

    def _memcpy_async(dst, src, nbytes, kind, stream):
        recorded.copies.append((kinds[kind], True))
        return memcpy_async(dst, src, nbytes, kind, stream)

    def _free(ptr):
        recorded.freed.append(("device", int(ptr)))
        return free(ptr)

    def _free_host(ptr):
        recorded.freed.append(("host", int(ptr)))
        return free_host(ptr)

    monkeypatch.setattr(cudart, "cudaMemcpy", _memcpy)
    monkeypatch.setattr(cudart, "cudaMemcpyAsync", _memcpy_async)
    monkeypatch.setattr(cudart, "cudaFree", _free)
    monkeypatch.setattr(cudart, "cudaFreeHost", _free_host)
    yield recorded
    # release what the test left behind while the runtime it came from is installed
    gc.collect()


@pytest.fixture(params=[pytest.param(False, id="sync"), pytest.param(True, id="async")])
def stream(request, calls):
    """No stream (synchronous copies), or a stream the test synchronizes."""
    if not request.param:
        yield None
    elif calls.runtime == "fake":
        # the fake runs async copies inline and ignores the handle
        yield 1
    else:
        handle = create_stream()
        yield handle
        destroy_stream(handle)


@pytest.fixture
def make_binding(calls):
    """Factory for bindings on the recorded runtime, freed after the test."""
    made: list[Binding] = []

    def _make(shape=(4, 3), dtype=_F32, *, array=None, **kwargs: object) -> Binding:
        binding = create_binding(np.zeros(shape, dtype) if array is None else array, **kwargs)
        made.append(binding)
        return binding

    yield _make
    for binding in made:
        binding.free()


@pytest.fixture
def allocate():
    """allocate_bindings, freeing every binding after the test."""
    made: list[Binding] = []

    def _allocate(engine, context, **kwargs: object):
        inputs, outputs, allocations = allocate_bindings(engine, context, **kwargs)
        made.extend(inputs + outputs)
        return inputs, outputs, allocations

    yield _allocate
    for binding in made:
        binding.free()


@pytest.fixture(params=[pytest.param(True, id="tensor-api"), pytest.param(False, id="binding-api")])
def tensor_api(request, monkeypatch) -> bool:
    """Take the TensorRT 10 tensor API path of allocate_bindings, or the older binding API."""
    monkeypatch.setattr(FLAGS, "TRT_10", request.param)
    return request.param


@pytest.fixture(params=[pytest.param(False, id="nvtx-off"), pytest.param(True, id="nvtx-on")])
def nvtx_ranges(request, monkeypatch) -> list[str | None]:
    """Record the NVTX pushes (names) and pops (None) of allocate_bindings."""
    ranges: list[str | None] = []
    monkeypatch.setattr(FLAGS, "NVTX_ENABLED", request.param)
    monkeypatch.setattr(
        _bindings,
        "nvtx",
        SimpleNamespace(push_range=ranges.append, pop_range=lambda: ranges.append(None)),
    )
    return ranges


def _data(shape: tuple[int, ...], start: int = 1) -> np.ndarray:
    return np.arange(start, start + math.prod(shape), dtype=_F32).reshape(shape)


def _sync(stream) -> None:
    if stream is not None:
        stream_synchronize(stream)


def _source(kind: str, data: np.ndarray, *, aligned: bool = True) -> Buffer:
    """``data`` in a Buffer of ``kind``, one row past the start of its allocation unless aligned."""
    shape = (len(data) + (not aligned), *data.shape[1:])
    if kind == "pageable":
        backing = Buffer.wrap(np.empty(shape, data.dtype))
    elif kind == "device":
        backing = Buffer.empty(shape, data.dtype, MemoryLocation.DEVICE)
    else:
        backing = Buffer.empty(shape, data.dtype, pinned=True, mapped=kind == "mapped")
    src = backing if aligned else backing[1:]
    src.copy_from(Buffer.wrap(data))
    return src


def _staging_copies(binding: Binding, src: Buffer, stream) -> list[tuple[str, bool]]:
    """What stage() copies: host data lands in unified memory with a plain host copy."""
    if binding.unified_mem:
        kind = None if src.is_host else "D2H"
    else:
        kind = "H2D" if src.is_host else "D2D"
    return [] if kind is None else [(kind, stream is not None)]


def _meta_binding(shape=(4, 3), dtype=_F32, *, name="x", **kwargs: object) -> Binding:
    """A Binding over pageable host Buffers, for logic that never touches its memory."""
    return Binding(
        0,
        name,
        dtype,
        list(shape),
        is_input=True,
        host=Buffer.empty(shape, dtype, pinned=False),
        device=Buffer.empty(shape, dtype, pinned=False),
        **kwargs,
    )


def _unbacked(shape: tuple[int, ...], dtype, location: MemoryLocation) -> Buffer:
    """A Buffer with the given metadata whose memory is never touched."""
    if location == MemoryLocation.HOST:
        return Buffer.wrap(np.zeros(shape, dtype))
    return Buffer.from_ptr(DEVICE_ALIGNMENT, shape, dtype)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("dtype", "shape", "min_shape", "is_input"),
    [
        pytest.param(np.float32, [1, 3, 64, 64], None, True, id="static-input"),
        pytest.param(np.dtype("float16"), (2, 8), (1, 8), True, id="dynamic-input"),
        pytest.param("int32", [np.int64(4)], [np.int64(1)], False, id="numpy-dims"),
        pytest.param(np.uint8, (5, 1), None, False, id="static-output"),
    ],
)
def test_binding_metadata(dtype, shape, min_shape, is_input) -> None:
    """Every property reflects the constructor arguments, normalized to numpy dtypes and ints."""
    host = Buffer.empty(shape, dtype, pinned=False)
    device = Buffer.empty(shape, dtype, pinned=False)
    binding = Binding(
        7,
        "t",
        dtype,
        shape,
        is_input=is_input,
        host=host,
        device=device,
        tensor_format=_FORMAT,
        min_shape=min_shape,
    )
    dims = tuple(int(d) for d in shape)
    assert (binding.index, binding.name, binding.is_input) == (7, "t", is_input)
    assert isinstance(binding.dtype, np.dtype)
    assert binding.dtype == np.dtype(dtype)
    assert binding.shape == list(dims)
    assert binding.max_shape == dims
    assert binding.min_shape == (dims if min_shape is None else tuple(min_shape))
    assert all(type(d) is int for d in [*binding.shape, *binding.min_shape])
    assert binding.tensor_format is _FORMAT
    assert binding.host is host
    assert binding.device is device
    assert binding.host_allocation is host.array
    assert binding.allocation == device.ptr
    assert (binding.pagelocked_mem, binding.unified_mem) == (False, False)


@pytest.mark.cpu
def test_binding_default_tensor_format() -> None:
    """Bindings are LINEAR unless told otherwise."""
    assert _meta_binding().tensor_format == trt.TensorFormat.LINEAR


@pytest.mark.cpu
def test_shape_hands_out_a_copy() -> None:
    """Mutating a returned shape never changes the binding."""
    binding = _meta_binding((2, 3))
    binding.shape.append(7)
    shape = binding.shape
    shape[0] = 9
    assert binding.shape == [2, 3]
    assert binding.max_shape == (2, 3)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "attr",
    [
        "index",
        "name",
        "dtype",
        "shape",
        "min_shape",
        "max_shape",
        "is_input",
        "host",
        "device",
        "tensor_format",
        "allocation",
        "host_allocation",
        "pagelocked_mem",
        "unified_mem",
    ],
)
def test_binding_fields_are_read_only(attr) -> None:
    """Every property rejects assignment and keeps its value."""
    binding = _meta_binding()
    before = getattr(binding, attr)
    with pytest.raises(AttributeError):
        setattr(binding, attr, None)
    after = getattr(binding, attr)
    assert after is before or after == before


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("is_input", "kind"),
    [pytest.param(True, "input", id="input"), pytest.param(False, "output", id="output")],
)
def test_repr(is_input, kind) -> None:
    """Repr names the index, tensor, direction, shape, and dtype."""
    shape = (1, 1000)
    binding = Binding(
        2,
        "probs",
        np.float16,
        list(shape),
        is_input=is_input,
        host=Buffer.empty(shape, np.float16, pinned=False),
        device=Buffer.empty(shape, np.float16, pinned=False),
    )
    assert repr(binding) == f"Binding(2, 'probs', {kind}, shape=[1, 1000], dtype=float16)"


_STATIC = ((2, 3), None)
_DYNAMIC = ((4, 8), (1, 2))


@pytest.mark.cpu
@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize(
    ("bounds", "shape", "dtype", "error"),
    [
        pytest.param(_STATIC, (2, 3), _F32, None, id="static-exact"),
        pytest.param(_STATIC, (1, 3), _F32, "got shape (1, 3), expected exactly (2, 3).", id="static-smaller"),
        pytest.param(_STATIC, (3, 3), _F32, "got shape (3, 3), expected exactly (2, 3).", id="static-larger"),
        pytest.param(_STATIC, (6,), _F32, "got shape (6,), expected exactly (2, 3).", id="static-flat"),
        pytest.param(_STATIC, (2, 3, 1), _F32, "got shape (2, 3, 1), expected exactly (2, 3).", id="static-extra-dim"),
        pytest.param(_STATIC, (2, 3), np.float64, "expects dtype float32, got float64.", id="static-dtype"),
        pytest.param(_DYNAMIC, (1, 2), _F32, None, id="dynamic-min"),
        pytest.param(_DYNAMIC, (4, 8), _F32, None, id="dynamic-max"),
        pytest.param(_DYNAMIC, (2, 5), _F32, None, id="dynamic-inside"),
        pytest.param(_DYNAMIC, (4, 2), _F32, None, id="dynamic-corner"),
        pytest.param(_DYNAMIC, (0, 5), _F32, "got shape (0, 5), expected between (1, 2) and (4, 8).", id="dynamic-below-min-batch"),
        pytest.param(_DYNAMIC, (2, 1), _F32, "got shape (2, 1), expected between (1, 2) and (4, 8).", id="dynamic-below-min-inner"),
        pytest.param(_DYNAMIC, (5, 5), _F32, "got shape (5, 5), expected between (1, 2) and (4, 8).", id="dynamic-above-max-batch"),
        pytest.param(_DYNAMIC, (2, 9), _F32, "got shape (2, 9), expected between (1, 2) and (4, 8).", id="dynamic-above-max-inner"),
        pytest.param(_DYNAMIC, (5,), _F32, "got shape (5,), expected between (1, 2) and (4, 8).", id="dynamic-fewer-dims"),
        pytest.param(_DYNAMIC, (2, 5, 1), _F32, "got shape (2, 5, 1), expected between (1, 2) and (4, 8).", id="dynamic-more-dims"),
        pytest.param(_DYNAMIC, (2, 5), np.int32, "expects dtype float32, got int32.", id="dynamic-dtype"),
    ],
)  # fmt: skip
def test_check(bounds, shape, dtype, error, location) -> None:
    """check() accepts dtype-matching shapes within [min, max] and names the tensor and engine."""
    max_shape, min_shape = bounds
    binding = _meta_binding(max_shape, min_shape=min_shape, name="images")
    buffer = _unbacked(shape, dtype, location)
    if error is None:
        binding.check(buffer, "net")
        return
    message = f"Input 'images' of engine 'net' {error}"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        binding.check(buffer, "net")


@pytest.mark.cpu
def test_check_remembers_only_accepted_shapes() -> None:
    """A rejected shape never becomes the remembered one, and dtypes are always checked."""
    binding = _meta_binding((4, 8), min_shape=(1, 2))
    accepted = Buffer.wrap(np.zeros((2, 5), _F32))
    rejected = Buffer.wrap(np.zeros((5, 8), _F32))
    binding.check(accepted, "net")
    binding.check(accepted, "net")
    for _ in range(2):
        with pytest.raises(ValueError, match="between"):
            binding.check(rejected, "net")
    binding.check(accepted, "net")
    # the dtype is checked even for the remembered shape
    with pytest.raises(ValueError, match="expects dtype"):
        binding.check(Buffer.wrap(np.zeros((2, 5), np.float64)), "net")


_CREATE_DEFAULTS = {
    "bind_id": 0,
    "name": "binding",
    "is_input": False,
    "tensor_format": trt.TensorFormat.LINEAR,
    "min_shape": None,
}


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="defaults"),
        pytest.param(
            {"bind_id": 3, "name": "images", "is_input": True, "min_shape": (1, 3, 8, 8)},
            id="dynamic-input",
        ),
        pytest.param({"bind_id": 1, "name": "scores", "is_input": False}, id="output"),
        pytest.param({"is_input": None}, id="is-input-none"),
        pytest.param({"tensor_format": _FORMAT}, id="tensor-format"),
    ],
)
def test_create_binding_metadata(make_binding, kwargs) -> None:
    """create_binding forwards its metadata, defaulting to an output named "binding" at index 0."""
    shape = (2, 3, 8, 8)
    binding = make_binding(shape, **kwargs)
    expected = {**_CREATE_DEFAULTS, **kwargs}
    assert binding.index == expected["bind_id"]
    assert binding.name == expected["name"]
    assert binding.is_input is bool(expected["is_input"])
    assert binding.tensor_format == expected["tensor_format"]
    assert binding.max_shape == shape
    assert binding.min_shape == (expected["min_shape"] or shape)


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        pytest.param(dtype, shape, id=f"{np.dtype(dtype).name}-{'x'.join(map(str, shape))}")
        for dtype, shape in itertools.product(
            [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.float16, np.float32],
            [(1,), (10,), (1, 3), (4, 4), (1, 1, 100), (1, 3, 64, 64)],
        )
    ],
)
def test_create_binding_shape_dtype(make_binding, dtype, shape) -> None:
    """Both buffers are allocated at the binding's shape and dtype."""
    binding = make_binding(shape, dtype)
    assert binding.shape == list(shape)
    assert binding.dtype == dtype
    for buffer in (binding.host, binding.device):
        assert (buffer.shape, buffer.dtype) == (shape, np.dtype(dtype))
        assert buffer.nbytes == math.prod(shape) * np.dtype(dtype).itemsize
    assert binding.host_allocation.shape == shape


@pytest.mark.parametrize(
    "use_array_data",
    [
        pytest.param(None, id="data-default"),
        pytest.param(True, id="data-copied"),
        pytest.param(False, id="data-skipped"),
    ],
)
@pytest.mark.parametrize(
    "pagelocked_mem",
    [
        pytest.param(None, id="pinned-default"),
        pytest.param(True, id="pinned"),
        pytest.param(False, id="pageable"),
    ],
)
@pytest.mark.parametrize(
    "unified_mem",
    [
        pytest.param(None, id="unified-default"),
        pytest.param(True, id="unified"),
        pytest.param(False, id="not-unified"),
    ],
)
def test_create_binding_memory(make_binding, calls, use_array_data, pagelocked_mem, unified_mem):
    """Pinned by default; unified needs pinned memory; array data is copied only on request."""
    data = _data((4, 3))
    binding = make_binding(
        array=data,
        use_array_data=use_array_data,
        pagelocked_mem=pagelocked_mem,
        unified_mem=unified_mem,
    )
    pinned = pagelocked_mem is not False
    unified = pinned and unified_mem is True
    host, device = binding.host, binding.device
    assert (host.location, device.location) == (MemoryLocation.HOST, MemoryLocation.DEVICE)
    assert (binding.pagelocked_mem, binding.unified_mem) == (pinned, unified)
    assert (host.pinned, host.mapped) == (pinned, unified)
    if unified:
        # one allocation: the device buffer is the device alias of the host buffer
        assert binding.allocation == host.device_ptr
    else:
        assert binding.allocation != host.ptr
    assert binding.host_allocation is not data
    if use_array_data:
        np.testing.assert_array_equal(binding.host_allocation, data)
        data += 100
        np.testing.assert_array_equal(binding.host_allocation, data - 100)
    # creating a binding never copies to or from the device
    assert calls.copies == []


@pytest.mark.parametrize("memory", MEMORY)
@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((4, 3), id="full"),
        pytest.param((2, 3), id="prefix"),
        pytest.param((1, 3), id="one-row"),
    ],
)
def test_stage(make_binding, calls, stream, memory, source, shape) -> None:
    """stage() copies into the leading elements and returns their device view."""
    binding = make_binding((4, 3), **memory)
    fill = np.full(binding.max_shape, -1, _F32)
    binding.stage(Buffer.wrap(fill))
    data = _data(shape)
    src = _source(source, data)
    calls.copies.clear()

    staged = binding.stage(src, stream)
    _sync(stream)

    assert calls.copies == _staging_copies(binding, src, stream)
    assert staged.is_device
    assert staged.ptr == binding.allocation
    assert staged.shape == shape
    assert (staged is binding.device) is (shape == binding.max_shape)
    np.testing.assert_array_equal(staged.numpy(), data)
    # the elements past the staged data are left untouched
    expected = fill.reshape(-1)
    expected[: data.size] = data.reshape(-1)
    np.testing.assert_array_equal(binding.fetch().array.reshape(-1), expected)


@pytest.mark.parametrize("method", ["stage", "bind"])
@pytest.mark.parametrize(
    ("data", "error"),
    [
        pytest.param(np.zeros((5, 3), _F32), "exceeds buffer", id="larger-than-binding"),
        pytest.param(np.zeros((2, 3), np.float64), "Buffer mismatch", id="other-dtype"),
    ],
)
def test_staging_rejects_mismatched_data(make_binding, calls, method, data, error) -> None:
    """Data larger than the binding or of another dtype raises before any copy."""
    binding = make_binding((4, 3))
    with pytest.raises(ValueError, match=error):
        getattr(binding, method)(Buffer.wrap(data))
    assert calls.copies == []


@pytest.mark.parametrize("memory", MEMORY)
@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize(
    "aligned", [pytest.param(True, id="aligned"), pytest.param(False, id="misaligned")]
)
def test_bind(make_binding, calls, stream, memory, source, aligned) -> None:
    """Aligned device-visible data (device or mapped host) is bound in place; the rest is staged."""
    binding = make_binding((4, 3), **memory)
    data = _data((2, 3))
    src = _source(source, data, aligned=aligned)
    assert src.device_visible is (source in ("mapped", "device"))
    if src.device_visible:
        assert (src.device_ptr % DEVICE_ALIGNMENT == 0) is aligned
    calls.copies.clear()

    bound = binding.bind(src, stream)
    _sync(stream)

    if src.device_visible and aligned:
        assert bound is src
        assert calls.copies == []
    else:
        assert bound.is_device
        assert bound.ptr == binding.allocation
        assert bound.shape == data.shape
        assert calls.copies == _staging_copies(binding, src, stream)
    np.testing.assert_array_equal(bound.numpy(), data)


@pytest.mark.parametrize("memory", MEMORY)
@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        pytest.param(None, (4, 3), id="default"),
        pytest.param((4, 3), (4, 3), id="full-tuple"),
        pytest.param([4, 3], (4, 3), id="full-list"),
        pytest.param((2, 3), (2, 3), id="prefix-tuple"),
        pytest.param([2, 3], (2, 3), id="prefix-list"),
        pytest.param((12,), (12,), id="flat"),
        pytest.param((0, 3), (0, 3), id="empty"),
    ],
)
def test_fetch(make_binding, calls, stream, memory, shape, expected) -> None:
    """fetch() copies the leading elements to the host buffer and returns that host view."""
    binding = make_binding((4, 3), **memory)
    data = _data((4, 3))
    binding.stage(Buffer.wrap(data))
    if not binding.unified_mem:
        binding.host_allocation.fill(-1)
    calls.copies.clear()

    fetched = binding.fetch(shape, stream)
    _sync(stream)

    assert calls.copies == ([] if binding.unified_mem else [("D2H", stream is not None)])
    assert fetched.is_host
    assert fetched.shape == expected
    assert fetched.ptr == binding.host.ptr
    if expected == binding.max_shape:
        assert fetched is binding.host
        assert fetched.array is binding.host_allocation
    np.testing.assert_array_equal(fetched.array.reshape(-1), data.reshape(-1)[: fetched.size])


@pytest.mark.parametrize(
    "shape", [pytest.param((5, 3), id="more-rows"), pytest.param((13,), id="flat-overflow")]
)
def test_fetch_rejects_shapes_beyond_the_allocation(make_binding, calls, shape) -> None:
    """fetch() of more elements than the allocation holds raises before any copy."""
    binding = make_binding((4, 3))
    with pytest.raises(ValueError, match="exceeds buffer"):
        binding.fetch(shape)
    assert calls.copies == []


def test_views_are_cached_for_the_last_shape(make_binding) -> None:
    """stage() and fetch() share one cached view pair, rebuilt only when the shape changes."""
    binding = make_binding((4, 3))
    first = binding.fetch((2, 3))
    assert binding.fetch([2, 3]) is first
    staged = binding.stage(Buffer.wrap(_data((2, 3))))
    assert binding.fetch((2, 3)) is first
    assert binding.stage(Buffer.wrap(_data((2, 3)))) is staged
    binding.fetch((1, 3))
    second = binding.fetch((2, 3))
    assert second is not first
    assert (second.ptr, second.shape) == (first.ptr, first.shape)
    # the full shape always hands back the binding's own buffers
    assert binding.fetch() is binding.host
    assert binding.stage(Buffer.wrap(_data((4, 3)))) is binding.device


def _hold(binding: Binding, held: str) -> object:
    """Take the reference ``held`` names; cached views are created and dropped."""
    if held == "host-array":
        return binding.host_allocation
    if held == "host-view":
        return binding.fetch((2, 3))
    if held == "device-view":
        return binding.stage(Buffer.wrap(_data((2, 3))))
    if held == "cached-views":
        binding.fetch((2, 3))
        binding.stage(Buffer.wrap(_data((1, 3))))
    return None


@pytest.mark.parametrize("memory", MEMORY)
@pytest.mark.parametrize(
    "held",
    [
        pytest.param(name, id=name)
        for name in ("nothing", "cached-views", "host-array", "host-view", "device-view")
    ],
)
def test_free(calls, memory, held) -> None:
    """free() returns the memory once nothing references it, exactly once, never earlier."""
    binding = create_binding(np.zeros((4, 3), _F32), **memory)
    binding.stage(Buffer.wrap(_data((4, 3))))
    binding.fetch()
    host_ptr, device_ptr = binding.host.ptr, binding.allocation
    if binding.unified_mem:
        owned = {("host", host_ptr)}
    elif binding.pagelocked_mem:
        owned = {("host", host_ptr), ("device", device_ptr)}
    else:
        owned = {("device", device_ptr)}
    ref = _hold(binding, held)
    # a device view of unified memory holds the host allocation it aliases
    on_device = held == "device-view" and not binding.unified_mem
    kept = (
        {e for e in owned if e[0] == ("device" if on_device else "host")}
        if ref is not None
        else set()
    )

    binding.free()
    binding.free()
    for attr in ("allocation", "host_allocation"):
        with pytest.raises(RuntimeError, match="freed"):
            getattr(binding, attr)
    gc.collect()
    assert {e for e in calls.freed if e in owned} == owned - kept

    # whatever is still referenced stays readable
    if isinstance(ref, Buffer):
        np.testing.assert_array_equal(ref.numpy(), _data((2, 3)))
    elif ref is not None:
        np.testing.assert_array_equal(ref, _data((4, 3)))
    del ref
    gc.collect()
    assert sorted(e for e in calls.freed if e in owned) == sorted(owned)


_SIMPLE_SHAPE = (1, 3, 160, 160)


@pytest.mark.parametrize("memory", [pytest.param({}, id="defaults"), *MEMORY])
def test_allocate_bindings_static(simple_engine, allocate, memory) -> None:
    """One binding per tensor at its engine shape, ordered by index, with matching allocations."""
    engine, context, _stream = simple_engine
    inputs, outputs, allocations = allocate(engine, context, **memory)
    pinned = memory.get("pagelocked_mem", True)
    unified = pinned and memory.get("unified_mem", False)

    assert [(b.index, b.name, b.is_input) for b in inputs] == [(0, "input", True)]
    assert [(b.index, b.name, b.is_input) for b in outputs] == [(1, "output", False)]
    assert allocations == [inputs[0].allocation, outputs[0].allocation]
    for binding in inputs + outputs:
        assert binding.shape == list(_SIMPLE_SHAPE)
        assert binding.min_shape == binding.max_shape == _SIMPLE_SHAPE
        assert binding.dtype == _F32
        assert binding.tensor_format == trt.TensorFormat.LINEAR
        assert (binding.pagelocked_mem, binding.unified_mem) == (pinned, unified)


@pytest.mark.parametrize(
    "profile", [pytest.param(p, id=name) for name, p in SIMPLE_DYNAMIC_PROFILES.items()]
)
def test_allocate_bindings_dynamic(build_dynamic_engine, allocate, profile) -> None:
    """Dynamic inputs, batch or not, allocate at the max profile shape and keep the min as a bound."""
    lo, _, hi = profile
    engine, context, _logger, stream = create_engine(
        build_dynamic_engine(DATA_DIR / "simple.onnx", profile)
    )
    try:
        (inp,), (out,), _ = allocate(engine, context)
        context_shape = (
            context.get_tensor_shape("input") if FLAGS.TRT_10 else context.get_binding_shape(0)
        )
        assert tuple(context_shape) == hi
        assert (inp.min_shape, inp.max_shape) == (lo, hi)
        assert (out.min_shape, out.max_shape) == (hi, hi)
        for binding in (inp, out):
            assert binding.host.shape == binding.device.shape == hi
        for shape in (lo, hi):
            inp.check(Buffer.wrap(np.zeros(shape, _F32)), "simple")
        with pytest.raises(ValueError, match="between"):
            inp.check(Buffer.wrap(np.zeros(tuple(d + 1 for d in hi), _F32)), "simple")
    finally:
        destroy_stream(stream)


class _FakeEngine:
    """
    A TensorRT engine and its execution context in one object.

    Exposes both the tensor API (TensorRT 10) and the binding API, so the
    branches of allocate_bindings run on any TensorRT version. Tensors are
    ``(name, is_input, shape, dtype)`` with dtype a ``trt.DataType`` name.
    """

    def __init__(self, tensors: list[tuple], profile: tuple | None) -> None:
        self._tensors = tensors
        self._index = {t[0]: i for i, t in enumerate(tensors)}
        self._profile = profile
        self.shapes = {t[0]: tuple(t[2]) for t in tensors}
        self.num_io_tensors = self.num_bindings = len(tensors)
        self.num_optimization_profiles = 0 if profile is None else 1

    def get_tensor_name(self, i: int) -> str:
        return self._tensors[i][0]

    def get_tensor_mode(self, name: str) -> object:
        return (
            trt.TensorIOMode.INPUT
            if self.binding_is_input(self._index[name])
            else trt.TensorIOMode.OUTPUT
        )

    def get_tensor_dtype(self, name: str) -> object:
        return self.get_binding_dtype(self._index[name])

    def get_tensor_format(self, _name: str) -> object:
        return trt.TensorFormat.LINEAR

    def get_tensor_profile_shape(self, _name: str, _profile: int) -> tuple | None:
        return self._profile

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        shape = self.shapes[name]
        if self.binding_is_input(self._index[name]):
            return shape
        # a dynamic output dim follows the batch of the first input
        batch = self.shapes[self._tensors[0][0]][0]
        return tuple(batch if dim < 0 else dim for dim in shape)

    def set_input_shape(self, name: str, shape: tuple[int, ...]) -> bool:
        if not self.binding_is_input(self._index[name]):
            err_msg = f"'{name}' is not an input"
            raise RuntimeError(err_msg)
        self.shapes[name] = tuple(shape)
        return True

    def binding_is_input(self, i: int) -> bool:
        return self._tensors[i][1]

    def get_binding_name(self, i: int) -> str:
        return self._tensors[i][0]

    def get_binding_dtype(self, i: int) -> object:
        return getattr(trt.DataType, self._tensors[i][3])

    def get_binding_format(self, _i: int) -> object:
        return trt.TensorFormat.LINEAR

    def get_profile_shape(self, _profile: int, _name: str) -> tuple | None:
        return self._profile

    def get_binding_shape(self, i: int) -> tuple[int, ...]:
        return self.get_tensor_shape(self._tensors[i][0])

    def set_binding_shape(self, i: int, shape: tuple[int, ...]) -> bool:
        return self.set_input_shape(self._tensors[i][0], shape)


@pytest.mark.usefixtures("fake_cuda", "tensor_api")
def test_allocate_bindings_partitions_interleaved_tensors(allocate, nvtx_ranges) -> None:
    """Engine order is kept, dtypes map to numpy, dynamic inputs resize, and outputs follow them."""
    fake = _FakeEngine(
        [
            ("a", True, (-1, 3), "FLOAT"),
            ("probs", False, (-1, 2), "HALF"),
            ("b", True, (2, 2), "INT32"),
        ],
        profile=((1, 3), (2, 3), (4, 3)),
    )
    inputs, outputs, allocations = allocate(fake, fake)

    assert [(b.index, b.name) for b in inputs] == [(0, "a"), (2, "b")]
    assert [(b.index, b.name) for b in outputs] == [(1, "probs")]
    assert allocations == [b.allocation for b in sorted(inputs + outputs, key=lambda b: b.index)]
    a, b = inputs
    (probs,) = outputs
    assert (a.min_shape, a.max_shape, fake.shapes["a"]) == ((1, 3), (4, 3), (4, 3))
    assert (b.min_shape, b.max_shape, fake.shapes["b"]) == ((2, 2), (2, 2), (2, 2))
    assert (probs.min_shape, probs.max_shape) == ((4, 2), (4, 2))
    assert [x.dtype for x in (a, probs, b)] == [np.float32, np.float16, np.int32]
    assert nvtx_ranges == (["core::allocate_bindings", None] if FLAGS.NVTX_ENABLED else [])


@pytest.mark.usefixtures("fake_cuda", "tensor_api")
@pytest.mark.parametrize(
    ("tensors", "profile", "error", "message"),
    [
        pytest.param(
            [("x", True, (-1, 3), "FLOAT"), ("y", False, (4, 3), "FLOAT")],
            None,
            RuntimeError,
            "No optimization profiles found",
            id="dynamic-without-profile",
        ),
        pytest.param(
            [("x", True, (-1, 3), "FLOAT"), ("y", False, (4, 3), "FLOAT")],
            ((1, 3), (4, 3)),
            RuntimeError,
            "Profile shape for tensor 'x' has 2 elements, expected 3",
            id="short-profile",
        ),
        pytest.param(
            [("y", False, (4, 3), "FLOAT")], None, ValueError, "No input tensors", id="no-inputs"
        ),
        pytest.param(
            [("x", True, (4, 3), "FLOAT")], None, ValueError, "No output tensors", id="no-outputs"
        ),
    ],
)
def test_allocate_bindings_rejects(tensors, profile, error, message, nvtx_ranges) -> None:
    """Malformed engines raise, and every exit path closes its NVTX range."""
    fake = _FakeEngine(tensors, profile)
    with pytest.raises(error, match=message):
        allocate_bindings(fake, fake)
    gc.collect()
    assert nvtx_ranges == (["core::allocate_bindings", None] if FLAGS.NVTX_ENABLED else [])
