# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Tests for I/O tensors in TensorRT formats other than LINEAR.

Bindings (src/trtutils/core/_bindings.py) are checked against what
TensorRT addresses, and TRTEngine against how it exchanges formatted I/O.
Every engine is a 3x3 convolution + ReLU built with the network API, so
each I/O format is exactly the one requested.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from tests.conftest import ENGINES_DIR
from trtutils._engine import TRTEngine
from trtutils._flags import FLAGS
from trtutils.compat._libs import trt
from trtutils.core import destroy_stream, stream_synchronize
from trtutils.core._bindings import Binding, allocate_bindings
from trtutils.core._buffer import Buffer, MemoryLocation
from trtutils.core._engine import create_engine

_SHAPE = (1, 3, 32, 48)
_WEIGHTS = (np.random.default_rng(0).standard_normal((3, 3, 3, 3)) * 0.3).astype(np.float32)
_GUARD = 4096
_SENTINEL = 0xAB

# case: (device, I/O dtype, input format, output format, input shape)
_CASES = {
    "gpu-linear": ("gpu", "FLOAT", "LINEAR", "LINEAR", _SHAPE),
    "gpu-hwc": ("gpu", "FLOAT", "HWC", "HWC", _SHAPE),
    "gpu-hwc-uint8": ("gpu", "UINT8", "HWC", "LINEAR", _SHAPE),
    "gpu-hwc8": ("gpu", "HALF", "HWC8", "HWC8", _SHAPE),
    "gpu-hwc8-output": ("gpu", "HALF", "LINEAR", "HWC8", _SHAPE),
    "gpu-chw32": ("gpu", "HALF", "CHW32", "CHW32", _SHAPE),
    "dla-linear-w48": ("dla", "HALF", "DLA_LINEAR", "DLA_LINEAR", _SHAPE),
    "dla-linear-w64": ("dla", "HALF", "DLA_LINEAR", "DLA_LINEAR", (1, 3, 32, 64)),
    "dla-linear-output": ("dla", "HALF", "LINEAR", "DLA_LINEAR", _SHAPE),
    "dla-chw16": ("dla", "HALF", "CHW16", "CHW16", _SHAPE),
    "dla-hwc4": ("dla", "HALF", "DLA_HWC4", "DLA_LINEAR", (1, 1, 32, 40)),
}


def _case(name: str, *values: object) -> object:
    marks = [pytest.mark.dla] if name.startswith("dla") else []
    return pytest.param(name, *values, id=name, marks=marks)


def _dla_core(case: str) -> int | None:
    return 0 if case.startswith("dla") else None


@pytest.fixture(scope="session")
def format_engine():
    """
    Build and cache the engine of a case in ``_CASES``, skipping cases TensorRT cannot build.

    Factory: format_engine(case).
    """

    def _build(case: str):
        device, dtype, in_fmt, out_fmt, shape = _CASES[case]
        path = ENGINES_DIR / f"io_format_{case}_{trt.__version__}.engine"
        if path.exists():
            return path
        builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        x = network.add_input("in_0", trt.float32, shape)
        # a uint8 input only feeds a cast
        first = network.add_cast(x, trt.float32).get_output(0) if dtype == "UINT8" else x
        weights = np.ascontiguousarray(_WEIGHTS[:, : shape[1]])
        conv = network.add_convolution_nd(
            first, 3, (3, 3), trt.Weights(weights), trt.Weights(np.zeros(3, np.float32))
        )
        conv.padding_nd = (1, 1)
        y = network.add_activation(conv.get_output(0), trt.ActivationType.RELU).get_output(0)
        y.name = "out_0"
        network.mark_output(y)
        x.dtype = getattr(trt.DataType, dtype)
        y.dtype = trt.float32 if dtype == "UINT8" else x.dtype
        x.allowed_formats = 1 << int(getattr(trt.TensorFormat, in_fmt))
        y.allowed_formats = 1 << int(getattr(trt.TensorFormat, out_fmt))
        config = builder.create_builder_config()
        config.builder_optimization_level = 0
        if dtype == "HALF":
            config.set_flag(trt.BuilderFlag.FP16)
        if device == "dla":
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = 0
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            pytest.skip(f"TensorRT {trt.__version__} cannot build {case}")
        ENGINES_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(serialized))
        return path

    return _build


def _offsets(binding: Binding) -> np.ndarray:
    """The scalar offset of every element of the binding's shape in its format's layout."""
    shape = binding.max_shape
    index = np.indices(shape)
    vdim, comps = binding.vectorized_dim, binding.components_per_element
    offsets = np.zeros(shape, np.int64)
    for i, stride in enumerate(binding.strides):
        if i == vdim:
            offsets += (index[i] // comps) * stride * comps + index[i] % comps
        else:
            offsets += index[i] * stride * comps
    return offsets


def _reference(x: np.ndarray) -> np.ndarray:
    """The engines' 3x3 convolution (padding 1) + ReLU on an NCHW array, in float32."""
    _, channels, height, width = x.shape
    padded = np.pad(x.astype(np.float32), ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros((x.shape[0], 3, height, width), np.float32)
    for i in range(3):
        for j in range(3):
            window = padded[:, :, i : i + height, j : j + width]
            out += np.einsum("oc,nchw->nohw", _WEIGHTS[:, :channels, i, j], window)
    return np.maximum(out, 0)


@pytest.mark.parametrize(
    "api", [pytest.param(True, id="tensor-api"), pytest.param(False, id="binding-api")]
)
@pytest.mark.parametrize(
    ("case", "formats", "nbytes"),
    [
        _case("gpu-linear", ("LINEAR", "LINEAR"), (18432, 18432)),
        _case("gpu-hwc", ("HWC", "HWC"), (18432, 18432)),
        _case("gpu-hwc-uint8", ("HWC", "LINEAR"), (4608, 18432)),
        # C=3 padded to 8 and 32 channels
        _case("gpu-hwc8", ("HWC8", "HWC8"), (24576, 24576)),
        _case("gpu-chw32", ("CHW32", "CHW32"), (98304, 98304)),
        # rows of 48 fp16 padded to 128 bytes; aligned rows of 64 are reported as LINEAR
        _case("dla-linear-w48", ("DLA_LINEAR", "DLA_LINEAR"), (12288, 12288)),
        _case("dla-linear-w64", ("LINEAR", "LINEAR"), (12288, 12288)),
        _case("dla-chw16", ("CHW16", "CHW16"), (49152, 49152)),
        # one channel, rows of 40 fp16 padded to 128 bytes
        _case("dla-hwc4", ("DLA_HWC4", "DLA_LINEAR"), (4096, 12288)),
    ],
)  # fmt: skip
def test_allocate_bindings_sizes_formatted_io(
    format_engine, monkeypatch, case, formats, nbytes, api
) -> None:
    """Bindings span the bytes TensorRT addresses in each format, flat when more than the shape."""
    engine, context, _logger, stream = create_engine(format_engine(case), dla_core=_dla_core(case))
    if not api and not hasattr(engine, "get_binding_format"):
        destroy_stream(stream)
        pytest.skip(f"TensorRT {trt.__version__} has no binding API")
    monkeypatch.setattr(FLAGS, "TRT_10", api)
    try:
        inputs, outputs, _ = allocate_bindings(engine, context)
        for binding, fmt, size in zip(inputs + outputs, formats, nbytes):
            dense = math.prod(binding.shape) * binding.dtype.itemsize
            assert binding.tensor_format == getattr(trt.TensorFormat, fmt)
            assert binding.strides == tuple(context.get_tensor_strides(binding.name))
            assert binding.nbytes == size
            alloc = tuple(binding.shape) if size == dense else (size // binding.dtype.itemsize,)
            for buffer in (binding.host, binding.device):
                assert (buffer.shape, buffer.nbytes) == (alloc, size)
            binding.free()
    finally:
        destroy_stream(stream)


@pytest.mark.parametrize("case", [_case(name) for name in _CASES])
def test_formatted_io_stays_inside_its_bindings(format_engine, case) -> None:
    """TensorRT reads and writes only Binding.nbytes of each tensor: guards past them stay intact."""
    engine, context, _logger, stream = create_engine(format_engine(case), dla_core=_dla_core(case))
    inputs, outputs, _ = allocate_bindings(engine, context)
    guarded: dict[str, Buffer] = {}
    try:
        for binding in inputs + outputs:
            # zeros within the layout; past it NaN (0xFF) for inputs, a sentinel for outputs
            raw = np.full(binding.nbytes + _GUARD, _SENTINEL, np.uint8)
            if binding.is_input:
                raw[: binding.nbytes] = 0
                raw[binding.nbytes :] = 0xFF
            guarded[binding.name] = Buffer.empty(raw.shape, np.uint8, MemoryLocation.DEVICE)
            guarded[binding.name].copy_from(Buffer.wrap(raw))
            context.set_tensor_address(binding.name, guarded[binding.name].ptr)
            binding.free()
        assert context.execute_async_v3(stream)
        stream_synchronize(stream)
        for binding in outputs:
            raw = guarded[binding.name].numpy()
            assert (raw[binding.nbytes :] == _SENTINEL).all()
            # the guard of every input is NaN, so a read past an input would surface here
            assert np.isfinite(raw[: binding.nbytes].view(binding.dtype)).all()
    finally:
        destroy_stream(stream)


@pytest.mark.parametrize(
    "unified_mem", [pytest.param(False, id="device-mem"), pytest.param(True, id="unified-mem")]
)
@pytest.mark.parametrize(
    "case",
    [_case("gpu-hwc-uint8"), _case("gpu-hwc"), _case("gpu-hwc8-output"), _case("dla-linear-output")],
)
def test_engine_exchanges_io_in_its_format(format_engine, case, unified_mem) -> None:
    """I/O is exchanged in the tensor's format; a padded output is whole only on its binding."""
    _, dtype, *_ = _CASES[case]
    engine = TRTEngine(
        format_engine(case), warmup=False, dla_core=_dla_core(case), unified_mem=unified_mem
    )
    (inp,), (out,) = engine.input_bindings, engine.output_bindings
    x = np.random.default_rng(1).integers(0, 4, _SHAPE).astype(inp.dtype)
    # the caller provides the input in its layout (HWC: channels last) under the tensor shape
    layout = np.zeros(inp.nbytes // inp.dtype.itemsize, inp.dtype)
    layout[_offsets(inp)] = x
    outputs = engine.execute([Buffer.wrap(layout.reshape(_SHAPE))])
    whole = out.fetch().array.reshape(-1)
    tol = 5e-2 if dtype == "HALF" else 1e-4
    np.testing.assert_allclose(whole[_offsets(out)], _reference(x), rtol=tol, atol=tol)
    # outputs at the tensor shape hold the leading bytes of the layout
    assert outputs[0].shape == _SHAPE
    np.testing.assert_array_equal(outputs[0].reshape(-1), whole[: math.prod(_SHAPE)])


@pytest.mark.parametrize(
    "location",
    [
        pytest.param(MemoryLocation.HOST, id="staged"),
        pytest.param(MemoryLocation.DEVICE, id="in-place"),
    ],
)
@pytest.mark.parametrize(
    ("case", "fmt", "nbytes"),
    [
        _case("gpu-hwc8", "HWC8", 24576),
        _case("dla-linear-w48", "DLA_LINEAR", 12288),
        _case("dla-chw16", "CHW16", 49152),
    ],
)  # fmt: skip
def test_engine_rejects_inputs_smaller_than_their_layout(
    format_engine, case, fmt, nbytes, location
) -> None:
    """A Buffer of the tensor shape for a padded input raises before TensorRT reads past it."""
    engine = TRTEngine(format_engine(case), warmup=False, dla_core=_dla_core(case))
    x = Buffer.from_array(np.zeros(_SHAPE, np.float16), location)
    message = f"is stored as {fmt}, which takes {nbytes} bytes at shape {_SHAPE}, but the Buffer"
    for run in (engine.execute, engine.raw_exec):
        with pytest.raises(ValueError, match=re.escape(message)):
            run([x])
