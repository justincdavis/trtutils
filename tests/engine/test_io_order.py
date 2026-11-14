# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Tests for src/trtutils/_engine.py -- engine I/O order: inputs keyed by name, outputs by output_names.

The test network has two same-shape inputs whose roles are not symmetric, and two
same-shape outputs listed in graph.output in the reverse of the order their nodes
appear, which TensorRT 8.6 builds into the engine in node order. Nothing but the
names can tell the tensors apart.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from trtutils import TRTEngine
from trtutils.builder import build_engine
from trtutils.core import Buffer, MemoryLocation, stream_synchronize

SHAPE = (1, 4, 8, 8)
ONNX_INPUTS = ["a", "b"]
# o_late is listed first, but its nodes come after the node of o_early
ONNX_OUTPUTS = ["o_late", "o_early"]


@pytest.fixture(scope="module")
def io_order_engine_path(tmp_path_factory):
    """Build the two-input, two-output test engine once per module."""

    def value(name: str) -> onnx.ValueInfoProto:
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(SHAPE))

    graph = helper.make_graph(
        [
            helper.make_node("Sub", ["b", "a"], ["o_early"]),
            helper.make_node("Add", ["a", "a"], ["t"]),
            helper.make_node("Sub", ["t", "b"], ["o_late"]),
        ],
        "io_order",
        [value(name) for name in ONNX_INPUTS],
        [value(name) for name in ONNX_OUTPUTS],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    directory = tmp_path_factory.mktemp("io_order")
    onnx_path = directory / "io_order.onnx"
    onnx.save(model, str(onnx_path))
    engine_path = directory / "io_order.engine"
    build_engine(onnx_path, engine_path, optimization_level=0)
    return engine_path


@pytest.fixture
def io_engine(io_order_engine_path):
    """Load a fresh engine per test."""
    eng = TRTEngine(io_order_engine_path, warmup=False, cuda_graph=False)
    yield eng
    del eng


@pytest.fixture
def host_data():
    """Random host data for each ONNX input, by name."""
    rng = np.random.default_rng(0)
    return {name: rng.standard_normal(SHAPE, dtype=np.float32) for name in ONNX_INPUTS}


def _assert_outputs(outputs: list[np.ndarray], names: list[str], host_data) -> None:
    a, b = host_data["a"], host_data["b"]
    expected = {"o_early": b - a, "o_late": (a + a) - b}
    got = dict(zip(names, outputs))
    assert sorted(got) == sorted(expected)
    for name, value in expected.items():
        np.testing.assert_allclose(got[name], value, rtol=1e-6, atol=1e-6)


def test_engine_reorders_outputs(io_engine) -> None:
    """The test network gets an engine output order unlike its ONNX order."""
    if io_engine.output_names == ONNX_OUTPUTS:
        pytest.skip("this TensorRT version keeps the ONNX output order")
    # TensorRT 8.6: the order the producing nodes appear in the graph
    assert io_engine.output_names == ["o_early", "o_late"]


@pytest.mark.parametrize(
    "form",
    [
        pytest.param("sequence", id="sequence"),
        pytest.param("mapping", id="mapping"),
        pytest.param("mapping-reversed", id="mapping-reversed"),
    ],
)
@pytest.mark.parametrize("entry", ["execute", "call", "mock_execute", "raw_exec"])
def test_inputs_by_name_outputs_by_output_names(io_engine, host_data, form, entry) -> None:
    """Inputs bind by name (sequences by input_names); outputs follow output_names."""
    location = MemoryLocation.DEVICE if entry == "raw_exec" else MemoryLocation.HOST
    buffers = {name: Buffer.from_array(host_data[name], location) for name in ONNX_INPUTS}
    if form == "sequence":
        data = [buffers[name] for name in io_engine.input_names]
    else:
        order = ONNX_INPUTS if form == "mapping" else ONNX_INPUTS[::-1]
        data = {name: buffers[name] for name in order}
    if entry == "raw_exec":
        device_outputs = io_engine.raw_exec(data)
        stream_synchronize(io_engine.stream)
        outputs = [o.numpy() for o in device_outputs]
    elif entry == "call":
        outputs = io_engine(data)
    elif entry == "mock_execute":
        outputs = io_engine.mock_execute(data)
    else:
        outputs = io_engine.execute(data)
    _assert_outputs(outputs, io_engine.output_names, host_data)


@pytest.mark.parametrize(
    ("keys", "wrap", "error", "match"),
    [
        pytest.param(["a"], True, ValueError, r"missing \['b'\]", id="missing"),
        pytest.param(["a", "b", "c"], True, ValueError, r"unknown \['c'\]", id="unknown"),
        pytest.param(
            ["a", "B"], True, ValueError, r"missing \['b'\]; unknown \['B'\]", id="misspelled"
        ),
        pytest.param(["a", "b"], False, TypeError, "must be Buffers", id="not-buffers"),
    ],
)
def test_mapping_errors(io_engine, host_data, keys, wrap, error, match) -> None:
    """A mapping with wrong names or non-Buffer values raises and leaves the engine usable."""
    named = {key: Buffer.wrap(host_data["a"]) if wrap else host_data["a"] for key in keys}
    with pytest.raises(error, match=match):
        io_engine.execute(named)
    good = {name: Buffer.wrap(host_data[name]) for name in ONNX_INPUTS}
    _assert_outputs(io_engine.execute(good), io_engine.output_names, host_data)
