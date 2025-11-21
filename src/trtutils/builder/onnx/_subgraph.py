# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import onnx
import onnx_graphsurgeon as gs

if TYPE_CHECKING:
    from pathlib import Path


def extract_subgraph(
    model: onnx.ModelProto | gs.Graph,
    start_idx: int,
    end_idx: int,
) -> onnx.ModelProto:
    """
    Extract a subgraph from an ONNX model using layer indices.

    Parameters
    ----------
    model : onnx.ModelProto | gs.Graph
        The ONNX model or GraphSurgeon graph to extract the subgraph from.
    start_idx : int
        The index of the first node in the subgraph.
    end_idx : int
        The index of the last node in the subgraph.

    Returns
    -------
    onnx.ModelProto
        The extracted subgraph.

    Raises
    ------
    ValueError
        If the start_idx is less than 0.
    ValueError
        If the end_idx is greater than the number of nodes in the graph.
    ValueError
        If the start_idx is greater than the end_idx.

    """
    if isinstance(model, onnx.ModelProto):
        graph = gs.import_onnx(model)
        graph.cleanup().toposort()
    else:
        graph = model
    nodes = graph.nodes

    # validate the indices
    if start_idx < 0:
        err_msg = "start_idx must be greater than 0"
        raise ValueError(err_msg)
    if end_idx >= len(nodes):
        err_msg = "end_idx must be less than the number of nodes in the graph"
        raise ValueError(err_msg)
    if start_idx > end_idx:
        err_msg = "start_idx must be less than end_idx"
        raise ValueError(err_msg)

    # extract the subgraph with inputs/outputs
    slice_nodes = nodes[start_idx : end_idx + 1]
    produced = {t for n in slice_nodes for t in n.outputs}
    consumed = {t for n in slice_nodes for t in n.inputs if t is not None}
    inputs = [t for t in consumed if t not in produced]
    outputs = []
    for t in produced:
        for n in nodes[end_idx + 1 :]:
            if t in n.inputs:
                outputs.append(t)
                break
    if not outputs:
        outputs = list(produced)

    # create the new subgraph
    sub_g = gs.Graph(nodes=slice_nodes, inputs=inputs, outputs=outputs)
    subgraph = sub_g.cleanup().toposort()

    # export and return
    return gs.export_onnx(subgraph)


def split_model(
    model: onnx.ModelProto,
    split_indices: list[int],
) -> list[onnx.ModelProto]:
    """
    Split an ONNX model into sequential pipeline subgraphs.

    Parameters
    ----------
    model : onnx.ModelProto
        The ONNX model to split.
    split_indices : list[int]
        The indices of the nodes to split the model at.

    Returns
    -------
    list[onnx.ModelProto]

    Raises
    ------
    ValueError
        If the split_indices are not a non-empty list.
    ValueError
        If the split_indices are not greater than 0.
    ValueError
        If the split_indices are not less than the number of nodes in the graph.

    """
    graph = gs.import_onnx(model)
    graph.cleanup().toposort()
    nodes = graph.nodes

    # validate the split indices
    split_indices = sorted(split_indices)
    if len(split_indices) == 0:
        err_msg = "split_indices must be a non-empty list"
        raise ValueError(err_msg)
    if split_indices[0] < 0:
        err_msg = "split_indices must be greater than 0"
        raise ValueError(err_msg)
    if split_indices[-1] >= len(nodes):
        err_msg = "split_indices must be less than the number of nodes in the graph"
        raise ValueError(err_msg)

    # create the boundaries
    boundaries = []
    prev = 0
    for idx in split_indices:
        boundaries.append((prev, idx))
        prev = idx + 1
    boundaries.append((prev, len(nodes) - 1))

    # extract the subgraphs
    subgraphs: list[onnx.ModelProto] = []
    for start, end in boundaries:
        subgraphs.append(extract_subgraph(model, start, end))
    return subgraphs


def extract_subgraph_from_file(
    model_path: Path,
    output_path: Path,
    start_idx: int,
    end_idx: int,
) -> None:
    """
    Extract a subgraph from an ONNX model file using layer indices.

    Parameters
    ----------
    model_path : Path
        The path to the ONNX model file.
    output_path : Path
        The path to save the subgraph to.
    start_idx : int
        The index of the first node in the subgraph.
    end_idx : int
        The index of the last node in the subgraph.

    """
    model = onnx.load(model_path)
    subgraph = extract_subgraph(model, start_idx, end_idx)
    onnx.save(subgraph, output_path)


def split_model_from_file(
    model_path: Path,
    output_dir: Path,
    split_indices: list[int],
) -> None:
    """
    Split an ONNX model file into sequential pipeline subgraphs.

    Parameters
    ----------
    model_path : Path
        The path to the ONNX model file.
    output_dir : Path
        The directory to save the subgraphs to.
    split_indices : list[int]
        The indices of the nodes to split the model at.

    """
    model = onnx.load(model_path)
    subgraphs = split_model(model, split_indices)
    for i, subgraph in enumerate(subgraphs):
        subgraph_path = output_dir / f"sub_{i}.onnx"
        onnx.save(subgraph, subgraph_path)
