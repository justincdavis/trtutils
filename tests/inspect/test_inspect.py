# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/inspect/ -- engine and ONNX inspection utilities."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import onnx
import pytest
import tensorrt as trt
from onnx import TensorProto, helper, numpy_helper

from trtutils import FLAGS
from trtutils.builder import build_engine
from trtutils.inspect import (
    LayerInfo,
    PlacementReport,
    get_engine_names,
    inspect_engine,
    inspect_onnx_layers,
    inspect_placement,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "engine_fixture",
    [
        pytest.param("engine_path", id="simple"),
        pytest.param("yolov10_engine_path", id="yolov10"),
    ],
)
def test_inspect_engine(engine_fixture, request) -> None:
    """inspect_engine returns correct structure with typed fields."""
    result = inspect_engine(request.getfixturevalue(engine_fixture))

    assert isinstance(result, tuple)
    assert len(result) == 4

    mem_size, batch_size, inputs, outputs = result
    assert isinstance(mem_size, int)
    assert mem_size >= 0
    assert isinstance(batch_size, int)
    assert isinstance(inputs, list)
    assert isinstance(outputs, list)

    # both inputs and outputs must have at least one 4-tuple entry
    for tensor_list in (inputs, outputs):
        assert len(tensor_list) > 0
        for entry in tensor_list:
            assert isinstance(entry, tuple)
            assert len(entry) == 4
            name, _shape, _dtype, _fmt = entry
            assert isinstance(name, str)
            assert len(name) > 0


@pytest.mark.parametrize(
    "engine_fixture",
    [
        pytest.param("engine_path", id="simple"),
        pytest.param("yolov10_engine_path", id="yolov10"),
    ],
)
def test_get_engine_names(engine_fixture, request) -> None:
    """get_engine_names returns non-empty lists of input/output name strings."""
    result = get_engine_names(request.getfixturevalue(engine_fixture))

    assert isinstance(result, tuple)
    assert len(result) == 2

    input_names, output_names = result
    assert isinstance(input_names, list)
    assert isinstance(output_names, list)
    assert len(input_names) >= 1
    assert len(output_names) >= 1

    for name in input_names + output_names:
        assert isinstance(name, str)
        assert len(name) > 0


@pytest.mark.parametrize(
    "onnx_fixture",
    [
        pytest.param("simple_onnx_path", id="simple"),
        pytest.param("yolov10_onnx_path", id="yolov10"),
    ],
)
def test_inspect_onnx_layers(onnx_fixture, request) -> None:
    """inspect_onnx_layers returns well-formed layer tuples."""
    onnx_path = request.getfixturevalue(onnx_fixture)
    if not onnx_path.exists():
        pytest.skip(f"{onnx_path.name} not found")

    layers = inspect_onnx_layers(onnx_path)
    assert isinstance(layers, list)
    assert len(layers) > 0
    for layer in layers:
        assert isinstance(layer, LayerInfo)
        assert isinstance(layer.index, int)
        assert isinstance(layer.name, str)
        assert isinstance(layer.layer_type, str)
        assert isinstance(layer.input_tensor_size, int)
        assert isinstance(layer.output_tensor_size, int)
        assert isinstance(layer.dla_compatible, bool)


def test_inspect_engine_nonexistent_raises() -> None:
    """Nonexistent engine file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        inspect_engine("nonexistent_engine_abc123.engine")


def test_inspect_onnx_layers_nonexistent_raises() -> None:
    """Nonexistent ONNX file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        inspect_onnx_layers("nonexistent_model_abc123.onnx")


class _FakeEngine:
    """Explicit-batch engine as TRT 8.6 reports it: max_batch_size is always 1."""

    max_batch_size = 1
    has_implicit_batch_dimension = False
    device_memory_size = 0
    device_memory_size_v2 = 0
    num_io_tensors = 2

    def __init__(self, in_shape) -> None:
        self._in_shape = in_shape

    def get_tensor_name(self, i):
        return ("in", "out")[i]

    def get_tensor_shape(self, name):
        return self._in_shape if name == "in" else (1, 10)

    def get_tensor_dtype(self, name):
        return trt.float32

    def get_tensor_format(self, name):
        return trt.TensorFormat.LINEAR

    def get_tensor_mode(self, name):
        return trt.TensorIOMode.INPUT if name == "in" else trt.TensorIOMode.OUTPUT

    def get_tensor_profile_shape(self, name, profile):
        return [(1, 3, 640, 640), (2, 3, 640, 640), (8, 3, 640, 640)]

    def get_profile_shape(self, profile, name):
        return self.get_tensor_profile_shape(name, profile)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("in_shape", "expected"),
    [
        pytest.param((4, 3, 640, 640), 4, id="static"),
        pytest.param((-1, 3, 640, 640), 8, id="dynamic-profile-max"),
    ],
)
def test_inspect_engine_batch_size_explicit_batch(monkeypatch, in_shape, expected) -> None:
    """Batch size comes from the input shape / profile max, not max_batch_size (always 1)."""
    monkeypatch.setattr(FLAGS, "TRT_10", True)
    _, batch_size, _, _ = inspect_engine(_FakeEngine(in_shape))
    assert batch_size == expected


def _roll_onnx(path: Path) -> Path:
    """Save a float32 ONNX net rolling 'in_0' by 3 along W, times 2, into 'out_0' (1, 16, 64, 64)."""
    initializers = [
        numpy_helper.from_array(np.array(value, np.int64), name)
        for name, value in (
            ("start_0", [-3]),
            ("end_0", [64]),
            ("start_1", [0]),
            ("end_1", [-3]),
            ("axis", [3]),
        )
    ]
    initializers.append(numpy_helper.from_array(np.array(2.0, np.float32), "two"))
    nodes = [
        helper.make_node("Slice", ["in_0", "start_0", "end_0", "axis"], ["a"], name="/Slice"),
        helper.make_node("Slice", ["in_0", "start_1", "end_1", "axis"], ["b"], name="/Slice_1"),
        helper.make_node("Concat", ["a", "b"], ["cat"], name="/Concat", axis=3),
        helper.make_node("Mul", ["cat", "two"], ["out_0"], name="/Mul"),
    ]
    shape = [1, 16, 64, 64]
    graph = helper.make_graph(
        nodes,
        "roll",
        [helper.make_tensor_value_info("in_0", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("out_0", TensorProto.FLOAT, shape)],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


@pytest.mark.parametrize(
    ("net", "dla_core", "subgraphs", "transitions", "devices"),
    [
        pytest.param(
            "Relu",
            None,
            0,
            0,
            {"/c1/Conv": "gpu", "/Relu": "gpu", "/c2/Conv": "gpu"},
            id="gpu-conv",
        ),
        pytest.param(
            "Relu",
            0,
            1,
            0,
            {"/c1/Conv": "dla", "/Relu": "dla", "/c2/Conv": "dla"},
            id="dla-conv",
            marks=pytest.mark.dla,
        ),
        pytest.param(
            "Sin",
            0,
            2,
            2,
            {"/c1/Conv": "dla", "/Sin": "gpu", "/c2/Conv": "dla"},
            id="dla-sin",
            marks=pytest.mark.dla,
        ),
        pytest.param(
            "roll",
            0,
            2,
            2,
            {"/Slice": "dla", "/Slice_1": "dla", "/Concat": "gpu", "/Mul": "dla"},
            id="dla-roll-gpu-copies",
            marks=pytest.mark.dla,
        ),
    ],
)
def test_inspect_placement(
    tmp_path, conv_onnx, net, dla_core, subgraphs, transitions, devices
) -> None:
    """Each network layer is attributed to the device it runs on, in GPU and DLA engines."""
    onnx_path = _roll_onnx(tmp_path / "roll.onnx") if net == "roll" else conv_onnx(16, 64, 64, net)
    engine_path = tmp_path / f"{net}.engine"
    build_engine(
        onnx_path,
        engine_path,
        default_device=trt.DeviceType.GPU if dla_core is None else trt.DeviceType.DLA,
        dla_core=dla_core,
        gpu_fallback=dla_core is not None,
        fp16=dla_core is not None,
        profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
        optimization_level=1,
    )
    report = inspect_placement(engine_path, dla_core, inspect_onnx_layers(onnx_path))
    on_gpu = [layer for layer, device in devices.items() if device == "gpu"]
    assert len(report.dla_subgraphs) == subgraphs
    assert report.transitions == transitions
    assert report.network_devices == devices
    assert report.fully_on_dla is (subgraphs > 0 and not on_gpu)
    assert [n.name for n in report.blocking_layers] == on_gpu
    assert PlacementReport.from_dict(json.loads(json.dumps(report.to_dict()))) == report


def test_inspect_placement_requires_detailed(tmp_path, conv_onnx) -> None:
    """An engine built without ProfilingVerbosity.DETAILED raises a ValueError."""
    engine_path = tmp_path / "conv.engine"
    build_engine(
        conv_onnx(16, 64, 64),
        engine_path,
        profiling_verbosity=trt.ProfilingVerbosity.LAYER_NAMES_ONLY,
        optimization_level=1,
    )
    with pytest.raises(ValueError, match="DETAILED"):
        inspect_placement(engine_path)
