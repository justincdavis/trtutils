# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_dla.py -- inputs a DLA reads in place at reissued addresses."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils import TRTEngine
from trtutils.compat._libs import trt
from trtutils.core import Buffer, stream_synchronize
from trtutils.core._dla import DLAMemoryGuard, dla_input_names

# CUDA hands a freed allocation this large back out with new pages, which is
# what a stale DLA read needs to show; small ones keep their pages
_ALLOC_ELEMS = 8 << 20
_SHAPE = (1, 16, 32, 32)


def _build(path, *, device: str, dtype: trt.DataType, verbosity: trt.ProfilingVerbosity):
    """Build a one-conv engine, on the DLA with no GPU layer before it when possible."""
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    x = network.add_input("input", dtype, _SHAPE)
    weights = np.random.default_rng(0).standard_normal((16, 16, 3, 3)).astype(np.float32) * 0.1
    conv = network.add_convolution_nd(
        x, 16, (3, 3), trt.Weights(weights), trt.Weights(np.zeros(16, dtype=np.float32))
    )
    conv.padding_nd = (1, 1)
    y = conv.get_output(0)
    y.name = "output"
    network.mark_output(y)
    for tensor in (x, y):
        tensor.dtype = dtype
        tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.profiling_verbosity = verbosity
    if device == "dla":
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = 0
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        if dtype == trt.float16:
            # fp16 rows of 64 bytes are valid DLA input; forbid a GPU reformat anyway
            config.set_flag(trt.BuilderFlag.DIRECT_IO)
    path.write_bytes(bytes(builder.build_serialized_network(network, config)))
    return path


@pytest.fixture(scope="module")
def dla_engine_path(tmp_path_factory):
    """A DLA engine whose fp16 input the DLA reads in place."""
    path = tmp_path_factory.mktemp("dla") / "direct.engine"
    return _build(path, device="dla", dtype=trt.float16, verbosity=trt.ProfilingVerbosity.DETAILED)


def test_guard_refuses_reissued_allocation() -> None:
    """The guard reads live allocations in place and refuses a new one at a recorded address."""
    guard = DLAMemoryGuard()
    first = Buffer.device((_ALLOC_ELEMS,), np.float16)
    other = Buffer.device((_ALLOC_ELEMS,), np.float16)
    assert guard.can_read(first.ptr)
    assert guard.can_read(first.ptr + 4096)
    assert guard.can_read(other.ptr)
    ptr = first.ptr
    first.free()
    held = []
    for _ in range(8):
        reissued = Buffer.device((_ALLOC_ELEMS,), np.float16)
        if reissued.ptr == ptr:
            break
        held.append(reissued)
    else:
        pytest.skip("CUDA never reissued the freed address")
    assert not guard.can_read(reissued.ptr)
    assert not guard.can_read(reissued.ptr + 4096)
    assert guard.can_read(other.ptr)


@pytest.mark.dla
@pytest.mark.parametrize(
    ("device", "dtype", "verbosity", "expected"),
    [
        pytest.param("dla", "float16", "DETAILED", ["input"], id="dla-reads-input"),
        pytest.param("dla", "float32", "DETAILED", [], id="dla-after-gpu-reformat"),
        pytest.param("dla", "float16", "LAYER_NAMES_ONLY", ["input"], id="dla-names-only"),
        pytest.param("dla", "float16", "NONE", ["input"], id="dla-no-layer-info"),
        pytest.param("gpu", "float16", "LAYER_NAMES_ONLY", [], id="gpu-names-only"),
        pytest.param("gpu", "float16", "DETAILED", [], id="gpu-detailed"),
    ],
)
def test_dla_input_names(tmp_path, device, dtype, verbosity, expected) -> None:
    """Inputs a DLA reads in place are found, conservatively without per-layer detail."""
    path = _build(
        tmp_path / "net.engine",
        device=device,
        dtype=getattr(trt, dtype),
        verbosity=getattr(trt.ProfilingVerbosity, verbosity),
    )
    engine = TRTEngine(path, warmup=False, cuda_graph=False)
    assert dla_input_names(engine.engine, engine.input_names) == expected
    assert [b.dla_guard is not None for b in engine.input_bindings] == [bool(expected)]


@pytest.mark.dla
@pytest.mark.regression
def test_dla_reads_input_at_reissued_address(dla_engine_path) -> None:
    """A DLA reading its input in place sees new data at an address freed and handed out again."""
    engine = TRTEngine(dla_engine_path, warmup=False, cuda_graph=False)
    rng = np.random.default_rng(0)
    seen: set[int] = set()
    reissued = 0
    for _ in range(8):
        data = rng.standard_normal(_SHAPE).astype(np.float16)
        # a host input is copied into the engine's binding, which also drops the
        # engine's reference to the previous device input so its memory is freed
        expected = engine.execute([Buffer.wrap(data)])[0]
        memory = Buffer.device((_ALLOC_ELEMS,), np.float16)
        x = memory[: data.size].reshape(_SHAPE)
        x.copy_from(Buffer.wrap(data))
        output = engine.raw_exec([x])[0]
        stream_synchronize(engine.stream)
        np.testing.assert_array_equal(output.numpy(), expected)
        reissued += memory.ptr in seen
        seen.add(memory.ptr)
        del x
        memory.free()
    if not reissued:
        pytest.skip("CUDA never reissued a freed input address")
