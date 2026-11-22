# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/inspect/_placement.py -- placement reports from inspector JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trtutils.inspect import EngineLayer, LayerInfo, PlacementReport

pytestmark = pytest.mark.cpu

FIXTURES = Path(__file__).parent / "fixtures"


def _layer(
    index: int,
    name: str,
    layer_type: str,
    *,
    dla_compatible: bool = True,
    outputs: tuple[str, ...] = (),
    inputs: tuple[str, ...] = (),
) -> LayerInfo:
    return LayerInfo(index, name, layer_type, None, 0, 0, dla_compatible, outputs, inputs)


def _fixture_report(name: str) -> PlacementReport:
    """Report from TensorRT 8.6 inspector JSON and the network layers the engine was built from."""
    fixture = json.loads((FIXTURES / f"{name}.json").read_text())
    layers = [
        _layer(
            n["index"],
            n["name"],
            n["layer_type"],
            dla_compatible=n["dla_compatible"],
            outputs=tuple(n.get("outputs", ())),
            inputs=(f"{n['name']}:in",),
        )
        for n in fixture["network_layers"]
    ]
    return PlacementReport.from_inspector_json(fixture["inspector"], layers, dla_core=0)


_CONV_RELU = {"/c1/Conv": "dla", "/Relu": "dla", "/c2/Conv": "dla", "/Relu_1": "dla"}


@pytest.mark.parametrize(
    ("name", "summary", "devices", "zero_gpu"),
    [
        pytest.param(
            "pure_fp32_io",
            "1 DLA subgraph(s), 0 GPU compute layer(s), 2 GPU reformat(s) (2 at I/O), "
            "0 DLA<->GPU transition(s), coverage 100%",
            _CONV_RELU,
            False,
            id="dla-fp32-io",
        ),
        pytest.param(
            "pure_fp16_aligned",
            "1 DLA subgraph(s), 0 GPU compute layer(s), 0 GPU reformat(s) (0 at I/O), "
            "0 DLA<->GPU transition(s), coverage 100%",
            _CONV_RELU,
            True,
            id="dla-fp16-aligned",
        ),
        pytest.param(
            "pure_fp16_unaligned",
            "1 DLA subgraph(s), 0 GPU compute layer(s), 2 GPU reformat(s) (2 at I/O), "
            "0 DLA<->GPU transition(s), coverage 100%",
            _CONV_RELU,
            False,
            id="dla-fp16-unaligned",
        ),
        pytest.param(
            "mixed_sin",
            "2 DLA subgraph(s), 1 GPU compute layer(s), 2 GPU reformat(s) (2 at I/O), "
            "2 DLA<->GPU transition(s), coverage 67%",
            {"/c1/Conv": "dla", "/Sin": "gpu", "/c2/Conv": "dla"},
            False,
            id="mixed-sin",
        ),
        pytest.param(
            "roll_concat",
            "2 DLA subgraph(s), 0 GPU compute layer(s), 4 GPU reformat(s) (2 at I/O, "
            "2 between DLA subgraphs), 2 DLA<->GPU transition(s), coverage 75%",
            {"/Slice": "dla", "/Slice_1": "dla", "/Concat": "gpu", "/Mul": "dla"},
            False,
            id="roll-concat-copies",
        ),
    ],
)
def test_report_from_inspector_json(name, summary, devices, zero_gpu) -> None:
    """Inspector JSON gives the placement summary and layer devices, and a JSON round trip."""
    report = _fixture_report(name)
    on_gpu = [layer for layer, device in devices.items() if device == "gpu"]
    assert report.summary() == f"DLA core 0: {summary}"
    assert report.network_devices == devices
    assert report.fully_on_dla is (not on_gpu)
    assert report.zero_gpu is zero_gpu
    assert [n.name for n in report.blocking_layers] == on_gpu
    if on_gpu:
        assert f"on the GPU: {on_gpu[0]} (" in str(report)
    again = PlacementReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert again == report
    assert str(again) == str(report)


def test_gpu_layer_inside_dla_range_runs_on_gpu() -> None:
    """A GPU layer TensorRT moves out of the middle of a DLA subgraph's range is on the GPU."""
    report = PlacementReport(
        engine_layers=[
            EngineLayer("{ForeignNode[/Mul.../Add]}", "DLA"),
            EngineLayer("PWN(/Div)", "PointWiseV2"),
        ],
        network_layers=[
            _layer(0, "/Mul", "ELEMENTWISE"),
            _layer(1, "/Div", "ELEMENTWISE"),
            _layer(2, "/Add", "ELEMENTWISE"),
        ],
    )
    assert report.network_devices == {"/Mul": "dla", "/Div": "gpu", "/Add": "dla"}
    assert [n.name for n in report.blocking_layers] == ["/Div"]
    assert report.dla_coverage == 2 / 3


def test_gpu_subgraph_range_runs_on_gpu() -> None:
    """A Myelin subgraph is named like a DLA subgraph but runs its range on the GPU."""
    report = PlacementReport(
        engine_layers=[EngineLayer("{ForeignNode[/MatMul.../Softmax]}", "Myelin")],
        network_layers=[
            _layer(0, "/MatMul", "MATRIX_MULTIPLY", dla_compatible=False),
            _layer(1, "/Div", "ELEMENTWISE"),
            _layer(2, "/Softmax", "SOFTMAX"),
        ],
    )
    assert report.dla_subgraphs == []
    assert report.fully_on_dla is False
    assert report.network_devices == {"/MatMul": "gpu", "/Div": "gpu", "/Softmax": "gpu"}
    assert report.dla_coverage == 0.0


def test_reformat_chain_to_output_is_io() -> None:
    """Reformats chained from a DLA subgraph to an engine output are I/O reformats."""
    report = PlacementReport(
        engine_layers=[
            EngineLayer("{ForeignNode[/Conv.../Relu]}", "DLA", ("in_0",), ("t0",)),
            EngineLayer("Reformatting CopyNode for Output Tensor 0", "Reformat", ("t0",), ("t1",)),
            EngineLayer(
                "Reformatting CopyNode for Output Tensor 1", "Reformat", ("t1",), ("out_0",)
            ),
        ],
        io_names=("in_0", "out_0"),
    )
    assert report.internal_reformat_layers == []
    assert len(report.io_reformat_layers) == 2
    assert report.fully_on_dla is True
    assert report.transitions == 0


def test_concat_copies_next_to_io_run_on_gpu() -> None:
    """Concat copies reading an engine input are GPU work, not I/O reformats."""
    report = PlacementReport(
        engine_layers=[
            EngineLayer("{ForeignNode[/Conv]}", "DLA", ("in_0",), ("c",)),
            EngineLayer("in_0 copy", "Reformat", ("in_0",), ("cat",), "CONCAT"),
            EngineLayer("c copy", "Reformat", ("c",), ("cat",), "CONCAT"),
            EngineLayer("{ForeignNode[/Mul]}", "DLA", ("cat",), ("out_0",)),
        ],
        network_layers=[
            _layer(0, "/Conv", "CONVOLUTION", outputs=("c",)),
            _layer(1, "/Concat", "CONCATENATION", dla_compatible=False, outputs=("cat",)),
            _layer(2, "/Mul", "ELEMENTWISE", outputs=("out_0",)),
        ],
        io_names=("in_0", "out_0"),
    )
    assert [r.name for r in report.internal_reformat_layers] == ["in_0 copy", "c copy"]
    assert report.fully_on_dla is False
    assert report.transitions == 2
    assert report.network_devices == {"/Conv": "dla", "/Concat": "gpu", "/Mul": "dla"}


def test_internal_reformat_does_not_move_its_producer_to_gpu() -> None:
    """A REFORMAT writing a DLA layer's output tensor leaves that layer on the DLA."""
    dla_out = "Reformatted Output Tensor 0 to {ForeignNode[/c1/Conv]}"
    report = PlacementReport(
        engine_layers=[
            EngineLayer("{ForeignNode[/c1/Conv]}", "DLA", ("in_0",), (dla_out,)),
            EngineLayer("Reformatting CopyNode", "Reformat", (dla_out,), ("c1",), "REFORMAT"),
            EngineLayer("PWN(/Sin)", "PointWiseV2", ("c1",), ("s",)),
            EngineLayer("{ForeignNode[/c2/Conv]}", "DLA", ("s",), ("out_0",)),
        ],
        network_layers=[
            _layer(0, "/c1/Conv", "CONVOLUTION", outputs=("c1",)),
            _layer(1, "/Sin", "UNARY", dla_compatible=False, outputs=("s",)),
            _layer(2, "/c2/Conv", "CONVOLUTION", outputs=("out_0",)),
        ],
        io_names=("in_0", "out_0"),
    )
    assert report.network_devices == {"/c1/Conv": "dla", "/Sin": "gpu", "/c2/Conv": "dla"}
    assert [n.name for n in report.blocking_layers] == ["/Sin"]


def test_uninspected_report() -> None:
    """A report without inspector output is not fully on the DLA and survives a round trip."""
    report = PlacementReport([], dla_core=1, inspected=False)
    assert report.fully_on_dla is False
    assert report.dla_coverage is None
    assert report.summary() == (
        "DLA core 1: placement unknown, the TensorRT engine inspector returned nothing"
    )
    assert PlacementReport.from_dict(json.loads(json.dumps(report.to_dict()))) == report


def test_names_only_inspector_json_raises() -> None:
    """Inspector JSON with layer names only raises a ValueError naming DETAILED."""
    with pytest.raises(ValueError, match="DETAILED"):
        PlacementReport.from_inspector_json({"Layers": ["conv1", "relu1"], "Bindings": []})
