# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/builder/onnx/_subgraph.py -- ONNX subgraph extraction."""

from __future__ import annotations

import pytest

onnx = pytest.importorskip("onnx")
gs = pytest.importorskip("onnx_graphsurgeon")

from trtutils.builder.onnx._subgraph import (  # noqa: E402
    extract_subgraph,
    extract_subgraph_from_file,
    split_model,
    split_model_from_file,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(scope="session")
def simple_onnx_model(onnx_path) -> onnx.ModelProto:
    """Load the test ONNX model."""
    return onnx.load(str(onnx_path))


@pytest.fixture(scope="session")
def simple_gs_graph(simple_onnx_model: onnx.ModelProto) -> gs.Graph:
    """Create a GraphSurgeon graph from the test model."""
    graph = gs.import_onnx(simple_onnx_model)
    graph.cleanup().toposort()
    return graph


def test_invalid_start_idx_negative(simple_onnx_model) -> None:
    """start_idx < 0 raises ValueError."""
    with pytest.raises(ValueError, match="start_idx must be greater than 0"):
        extract_subgraph(simple_onnx_model, -1, 0)


def test_invalid_end_idx_too_large(simple_onnx_model, simple_gs_graph) -> None:
    """end_idx >= num_nodes raises ValueError."""
    with pytest.raises(ValueError, match="end_idx must be less than"):
        extract_subgraph(simple_onnx_model, 0, len(simple_gs_graph.nodes))


def test_invalid_start_greater_than_end(simple_onnx_model) -> None:
    """start_idx > end_idx raises ValueError."""
    with pytest.raises(ValueError, match="start_idx must be less than end_idx"):
        extract_subgraph(simple_onnx_model, 1, 0)


def test_extract_from_model_proto(simple_onnx_model) -> None:
    """extract_subgraph accepts an onnx.ModelProto and returns a submodel."""
    sub = extract_subgraph(simple_onnx_model, 0, 0)
    assert isinstance(sub, onnx.ModelProto)
    assert len(sub.graph.node) > 0


def test_extract_from_gs_graph(simple_gs_graph) -> None:
    """extract_subgraph accepts a gs.Graph directly."""
    sub = extract_subgraph(simple_gs_graph, 0, 0)
    assert isinstance(sub, onnx.ModelProto)


def test_split_into_two(simple_onnx_model) -> None:
    """Splitting at one index produces two subgraphs."""
    parts = split_model(simple_onnx_model, [0])
    assert len(parts) == 2
    for part in parts:
        assert isinstance(part, onnx.ModelProto)


def test_split_empty_indices_raises(simple_onnx_model) -> None:
    """Empty split_indices raises ValueError."""
    with pytest.raises(ValueError, match="non-empty list"):
        split_model(simple_onnx_model, [])


def test_split_negative_index_raises(simple_onnx_model) -> None:
    """Negative split index raises ValueError."""
    with pytest.raises(ValueError, match="greater than 0"):
        split_model(simple_onnx_model, [-1])


def test_split_index_too_large_raises(simple_onnx_model, simple_gs_graph) -> None:
    """Split index >= num_nodes raises ValueError."""
    with pytest.raises(ValueError, match="less than the number of nodes"):
        split_model(simple_onnx_model, [len(simple_gs_graph.nodes)])


def test_extract_subgraph_from_file(onnx_path, tmp_path) -> None:
    """extract_subgraph_from_file writes a valid ONNX file."""
    output = tmp_path / "sub.onnx"
    extract_subgraph_from_file(onnx_path, output, 0, 0)
    assert output.exists()
    loaded = onnx.load(str(output))
    assert isinstance(loaded, onnx.ModelProto)


def test_split_model_from_file(onnx_path, tmp_path) -> None:
    """split_model_from_file writes multiple ONNX files."""
    out_dir = tmp_path / "splits"
    out_dir.mkdir()
    split_model_from_file(onnx_path, out_dir, [0])

    assert (out_dir / "sub_0.onnx").exists()
    assert (out_dir / "sub_1.onnx").exists()


def test_split_pieces_chain_and_validate(simple_onnx_model, simple_gs_graph) -> None:
    """Split pieces cover all nodes, chain by tensor names, and pass the ONNX checker."""
    parts = split_model(simple_onnx_model, [0])
    for part in parts:
        onnx.checker.check_model(part)
    assert sum(len(p.graph.node) for p in parts) == len(simple_gs_graph.nodes)
    out_names = {o.name for o in parts[0].graph.output}
    in_names = {i.name for i in parts[1].graph.input}
    assert out_names & in_names
