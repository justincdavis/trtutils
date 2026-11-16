# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/builder/hooks/_io.py and _ranges.py -- I/O dtypes and INT8 ranges."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from trtutils import TRTEngine
from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.builder import build_engine
from trtutils.builder.hooks import fp16_io_hook, int8_input_hook, int8_ranges_hook
from trtutils.compat._libs import trt
from trtutils.core import Buffer

_IDENTITY_SHAPE = (1, 16, 16, 64)
_SMOOTH_SHAPE = (1, 16, 32, 32)

_needs_weak_typing = pytest.mark.skipif(
    not FLAGS.WEAK_TYPING, reason="weakly-typed networks were removed in TensorRT 11"
)
_needs_implicit_int8 = pytest.mark.skipif(
    not FLAGS.DYNAMIC_RANGE or FLAGS.IS_BLACKWELL,
    reason="implicit INT8 was removed in TensorRT 11 and is not supported on Blackwell",
)
_DEVICES = [
    pytest.param("gpu", id="gpu"),
    pytest.param("dla", id="dla", marks=pytest.mark.dla),
]


def _save_onnx(path, nodes, shape_in, shape_out, initializer):
    """Save a single-input, single-output float32 ONNX graph x -> y."""
    graph = helper.make_graph(
        nodes,
        path.stem,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_in)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_out)],
        initializer=initializer,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


@pytest.fixture(scope="module")
def identity_onnx_path(tmp_path_factory):
    """3x3 convolution whose output equals its input, (1, 16, 16, 64)."""
    weights = np.zeros((16, 16, 3, 3), dtype=np.float32)
    weights[:, :, 1, 1] = np.eye(16)
    return _save_onnx(
        tmp_path_factory.mktemp("identity") / "identity.onnx",
        [helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])],
        list(_IDENTITY_SHAPE),
        list(_IDENTITY_SHAPE),
        [numpy_helper.from_array(weights, "w")],
    )


@pytest.fixture(scope="module")
def smooth_onnx_path(tmp_path_factory):
    """Depthwise 3x3 box blur minus 0.25, ReLU, then 2x2 average pooling, (1, 16, 32, 32)."""
    channels = _SMOOTH_SHAPE[1]
    weights = np.full((channels, 1, 3, 3), 1 / 9, dtype=np.float32)
    bias = np.full(channels, -0.25, dtype=np.float32)
    nodes = [
        helper.make_node("Conv", ["x", "w", "b"], ["c"], pads=[1, 1, 1, 1], group=channels),
        helper.make_node("Relu", ["c"], ["r"]),
        helper.make_node("AveragePool", ["r"], ["y"], kernel_shape=[2, 2], strides=[2, 2]),
    ]
    height, width = _SMOOTH_SHAPE[2] // 2, _SMOOTH_SHAPE[3] // 2
    return _save_onnx(
        tmp_path_factory.mktemp("smooth") / "smooth.onnx",
        nodes,
        list(_SMOOTH_SHAPE),
        [1, channels, height, width],
        [numpy_helper.from_array(weights, "w"), numpy_helper.from_array(bias, "b")],
    )


def _int8_build() -> dict:
    """INT8 build options; the GPU runs these small convolutions in fp16 unless asked for INT8."""
    return {
        "int8": True,
        "layer_precision": [(0, trt.DataType.INT8)],
        "prefer_precision_constraints": True,
    }


def _build(onnx_path, tmp_path, device, hooks, **precision: object) -> TRTEngine:
    """Build ``onnx_path`` on ``device`` with ``hooks`` and load the engine."""
    engine_path = tmp_path / f"{onnx_path.stem}.engine"
    build_engine(
        onnx_path,
        engine_path,
        default_device=device,
        gpu_fallback=True,
        hooks=hooks,
        optimization_level=1,
        **precision,
    )
    return TRTEngine(
        engine_path, warmup=False, cuda_graph=False, dla_core=0 if device == "dla" else None
    )


def _smooth_reference(x: np.ndarray) -> np.ndarray:
    """Float64 reference of the smooth_onnx_path network."""
    _, _, height, width = x.shape
    padded = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    blur = sum(padded[:, :, i : i + height, j : j + width] for i in range(3) for j in range(3))
    relu = np.maximum(blur / 9 - 0.25, 0.0)
    return relu.reshape(1, -1, height // 2, 2, width // 2, 2).mean(axis=(3, 5))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    ("io", "dtypes"),
    [
        pytest.param("fp16", (np.float16, np.float16), id="fp16-io", marks=_needs_weak_typing),
        pytest.param("int8", (np.int8, np.float16), id="int8-input", marks=_needs_implicit_int8),
    ],
)
def test_io_hooks_keep_every_int8_value(identity_onnx_path, tmp_path, io, dtypes, device) -> None:
    """The hooks set the engine I/O dtypes and all 256 int8 values pass an identity conv exactly."""
    if io == "fp16":
        hooks, precision = [fp16_io_hook], {"fp16": True}
    else:
        hooks, precision = [int8_input_hook, fp16_io_hook, int8_ranges_hook()], _int8_build()
    engine = _build(identity_onnx_path, tmp_path, device, hooks, **precision)
    assert (engine.input_dtypes[0], engine.output_dtypes[0]) == dtypes
    values = np.resize(np.arange(-128, 128), _IDENTITY_SHAPE).astype(dtypes[0])
    output = engine.execute([Buffer.wrap(values)])[0]
    np.testing.assert_array_equal(output, values.astype(np.float16))


@_needs_implicit_int8
def test_int8_ranges_hook_sets_hand_computed_ranges() -> None:
    """Each tensor gets the tightest symmetric range covering its hand-computed interval."""
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    x = network.add_input("x", trt.float32, (1, 2, 8, 8))
    kernel = np.array([[1.0, -2.0], [0.5, 0.5]], dtype=np.float32).reshape(2, 2, 1, 1)
    bias = np.array([0.0, 1.0], dtype=np.float32)
    conv = network.add_convolution_nd(x, 2, (1, 1), kernel, bias)
    relu = network.add_activation(conv.get_output(0), trt.ActivationType.RELU)
    pool = network.add_pooling_nd(relu.get_output(0), trt.PoolingType.AVERAGE, (2, 2))
    pool.stride_nd = (2, 2)
    shift, scale, power = (np.array([v], dtype=np.float32) for v in (1.0, -2.0, 1.0))
    affine = network.add_scale(pool.get_output(0), trt.ScaleMode.UNIFORM, shift, scale, power)
    concat = network.add_concatenation([pool.get_output(0), affine.get_output(0)])
    exp = network.add_unary(concat.get_output(0), trt.UnaryOperation.EXP)
    network.mark_output(exp.get_output(0))

    with pytest.raises(ValueError, match=r"no input_ranges entry for the network inputs \['x'\]"):
        int8_ranges_hook()(network)

    conv.get_output(0).set_dynamic_range(-100.0, 100.0)
    int8_ranges_hook({"x": (-1.0, 3.0)})(network)

    # channel 0 is x0 - 2 x1 in [-7, 5], channel 1 is (x0 + x1) / 2 + 1 in [0, 4]
    intervals = {
        x: (-1.0, 3.0),
        relu.get_output(0): (0.0, 5.0),
        pool.get_output(0): (0.0, 5.0),
        affine.get_output(0): (-9.0, 1.0),
        concat.get_output(0): (-9.0, 5.0),
    }
    for tensor, (lo, hi) in intervals.items():
        bound = max(hi, -lo * 127 / 128)
        assert tensor.dynamic_range == pytest.approx((-bound, bound)), tensor.name
    assert conv.get_output(0).dynamic_range == (-100.0, 100.0)
    assert exp.get_output(0).dynamic_range is None


@_needs_implicit_int8
@pytest.mark.parametrize("device", _DEVICES)
def test_int8_ranges_build_matches_fp32(smooth_onnx_path, tmp_path, device) -> None:
    """A data-free INT8 blur -> ReLU -> average pool on [0, 1] is within two INT8 steps of fp32."""
    hooks = [fp16_io_hook, int8_ranges_hook({"x": (0.0, 1.0)})]
    with patch.object(LOG, "warning", wraps=LOG.warning) as warning:
        engine = _build(smooth_onnx_path, tmp_path, device, hooks, **_int8_build())
    messages = [str(call.args[0]) for call in warning.call_args_list if call.args]
    assert not [m for m in messages if m.startswith("INT8 without calibration data")]

    x = np.random.default_rng(0).random(_SMOOTH_SHAPE).astype(np.float16)
    output = engine.execute([Buffer.wrap(x)])[0]
    # the output interval is [0, 0.75], one INT8 step is 0.75 / 127
    assert np.abs(output - _smooth_reference(x)).max() <= 2 * 0.75 / 127
