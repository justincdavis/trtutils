# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_dla.py -- inputs a DLA reads in place at reissued addresses."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils import TRTEngine
from trtutils.compat._libs import trt
from trtutils.core import Binding, Buffer, stream_synchronize
from trtutils.core._dla import DLAInputBinding, DLAMemoryGuard, dla_input_names

# CUDA hands a freed allocation this large back out with new pages, which is
# what a stale DLA read needs to show; small ones keep their pages
_ALLOC_ELEMS = 8 << 20
_SHAPE = (1, 16, 32, 32)


def _build(path, net: str, verbosity: trt.ProfilingVerbosity):
    """
    Build a small engine.

    ``dla``: an fp16 conv whose input the DLA reads directly. ``dla-mixed``: the
    same plus an fp32 input added to its output, which a GPU reformat feeds to
    the DLA. ``gpu-conv``: the conv on the GPU. ``gpu-layernorm``: a layer norm,
    which TensorRT compiles into a Myelin ForeignNode on the GPU.
    """
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.profiling_verbosity = verbosity
    if net == "gpu-layernorm":
        x = network.add_input("input", trt.float32, (1, 32, 64))
        scale = network.add_constant((1, 1, 64), np.ones((1, 1, 64), dtype=np.float32))
        bias = network.add_constant((1, 1, 64), np.zeros((1, 1, 64), dtype=np.float32))
        norm = network.add_normalization(x, scale.get_output(0), bias.get_output(0), 1 << 2)
        y = norm.get_output(0)
    else:
        x = network.add_input("input", trt.float16, _SHAPE)
        weights = np.random.default_rng(0).standard_normal((16, 16, 3, 3)).astype(np.float32)
        weights *= 0.1
        conv = network.add_convolution_nd(
            x, 16, (3, 3), trt.Weights(weights), trt.Weights(np.zeros(16, dtype=np.float32))
        )
        conv.padding_nd = (1, 1)
        y = conv.get_output(0)
        if net == "dla-mixed":
            other = network.add_input("other", trt.float32, _SHAPE)
            y = network.add_elementwise(y, other, trt.ElementWiseOperation.SUM).get_output(0)
        for tensor in (x, y):
            tensor.dtype = trt.float16
            tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        config.set_flag(trt.BuilderFlag.FP16)
        if net != "gpu-conv":
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = 0
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            # fp16 rows of 64 bytes are valid DLA input; forbid a GPU reformat anyway
            config.set_flag(trt.BuilderFlag.DIRECT_IO)
    y.name = "output"
    network.mark_output(y)
    path.write_bytes(bytes(builder.build_serialized_network(network, config)))
    return path


@pytest.fixture(scope="module")
def dla_engine_path(tmp_path_factory):
    """A DLA engine whose fp16 input the DLA reads in place."""
    path = tmp_path_factory.mktemp("dla") / "direct.engine"
    return _build(path, "dla", trt.ProfilingVerbosity.DETAILED)


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
    ("net", "verbosity", "dla_core", "expected"),
    [
        pytest.param("gpu-layernorm", "LAYER_NAMES_ONLY", None, [], id="gpu-myelin-names-only"),
        pytest.param("gpu-conv", "DETAILED", None, [], id="gpu-detailed"),
        pytest.param("gpu-conv", "DETAILED", 0, [], id="gpu-detailed-dla-core"),
        pytest.param("dla-mixed", "DETAILED", 0, ["input"], id="dla-detailed-dla-core"),
        pytest.param("dla-mixed", "DETAILED", None, [], id="dla-detailed-no-dla-core"),
        pytest.param(
            "dla-mixed", "LAYER_NAMES_ONLY", 0, ["input", "other"], id="dla-names-only-dla-core"
        ),
        pytest.param("dla-mixed", "NONE", 0, ["input", "other"], id="dla-no-layer-info-dla-core"),
        pytest.param("dla-mixed", "LAYER_NAMES_ONLY", None, [], id="dla-names-only-no-dla-core"),
    ],
)
def test_dla_inputs_get_guarded_bind(tmp_path, net, verbosity, dla_core, expected) -> None:
    """Exactly the inputs a DLA reads in place get a guarded bind; the rest keep Binding.bind."""
    path = _build(tmp_path / "net.engine", net, getattr(trt.ProfilingVerbosity, verbosity))
    engine = TRTEngine(path, warmup=False, cuda_graph=False, dla_core=dla_core)
    if net == "gpu-layernorm":
        # names only, a Myelin subgraph looks like a DLA one
        info = engine.engine.create_engine_inspector().get_engine_information(
            trt.LayerInformationFormat.JSON
        )
        assert "ForeignNode" in info
    assert dla_input_names(engine.engine, engine.input_names, dla_core) == expected
    guarded = [b.name for b in engine.input_bindings if isinstance(b, DLAInputBinding)]
    assert guarded == expected
    assert all(type(b) is Binding for b in engine.input_bindings if b.name not in expected)


@pytest.mark.dla
@pytest.mark.regression
def test_dla_reads_input_at_reissued_address(dla_engine_path) -> None:
    """A DLA reading its input in place sees new data at an address freed and handed out again."""
    engine = TRTEngine(dla_engine_path, warmup=False, cuda_graph=False, dla_core=0)
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
