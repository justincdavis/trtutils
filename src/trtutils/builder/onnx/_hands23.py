# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""ONNX graph pass that removes the NonZero ops from an exported Hands23 model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
from onnx import helper, numpy_helper

from trtutils._log import LOG

if TYPE_CHECKING:
    from pathlib import Path

# score given to rpn proposals that fail the zero-area filter
_INVALID_SCORE = -1e4


def _producer(
    by_out: dict[str, onnx.NodeProto],
    tensor: str,
    op_types: tuple[str, ...],
) -> onnx.NodeProto | None:
    node = by_out.get(tensor)
    if node is None or node.op_type != op_types[0]:
        return None
    if len(op_types) == 1:
        return node
    return _producer(by_out, node.input[0], op_types[1:])


def _nonzero_cond(by_out: dict[str, onnx.NodeProto], index: str) -> str | None:
    # index is Transpose(NonZero(cond)); returns the cond tensor name
    nonzero = _producer(by_out, index, ("Transpose", "NonZero"))
    return None if nonzero is None else nonzero.input[0]


def _match_rois(
    by_out: dict[str, onnx.NodeProto],
    node: onnx.NodeProto,
) -> tuple[str, str] | None:
    # rois are Slice(Gather(boxes, Squeeze(Transpose(NonZero(level == i)))))
    gather = _producer(by_out, node.input[1], ("Slice", "Gather"))
    if gather is None:
        return None
    squeeze = _producer(by_out, gather.input[1], ("Squeeze",))
    mask = None if squeeze is None else _nonzero_cond(by_out, squeeze.input[0])
    return None if mask is None else (mask, gather.name)


def _match_rpn(
    graph: onnx.GraphProto,
    by_out: dict[str, onnx.NodeProto],
    nms: onnx.NodeProto,
) -> dict[str, str]:
    # rpn nms scores are Unsqueeze(GatherND(scores, Transpose(NonZero(And)))); every GatherND
    # on that index drops the invalid proposals: the scores get the mask, the rest pass through
    score_gather = _producer(by_out, nms.input[1], ("Unsqueeze", "GatherND"))
    if score_gather is None:
        return {}
    valid = _nonzero_cond(by_out, score_gather.input[1])
    if valid is None or by_out[valid].op_type != "And":
        return {}
    return {
        n.name: valid if n is score_gather else ""
        for n in graph.node
        if n.op_type == "GatherND" and n.input[1] == score_gather.input[1]
    }


def _match(
    graph: onnx.GraphProto,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], set[str]] | None:
    """
    Find the rewrite sites, or None if the graph is not the Hands23 export pattern.

    Returns ``(roi_masks, scatter_masks, rpn_where, bypass)``: ROIAlign node name -> level
    mask, ScatterND node name -> level mask, RPN GatherND node name -> valid mask or "" for
    the ones that become Identity, and the Gather node names whose Slice users read the full
    box tensor instead.
    """
    by_out = {o: n for n in graph.node for o in n.output}
    roi_masks: dict[str, str] = {}
    scatter_masks: dict[str, str] = {}
    rpn_where: dict[str, str] = {}
    bypass: set[str] = set()

    for node in graph.node:
        if node.op_type == "ROIAlign_TRT":
            found = _match_rois(by_out, node)
            if found is None:
                return None
            roi_masks[node.name] = found[0]
            bypass.add(found[1])
        elif node.op_type == "ScatterND":
            mask = _nonzero_cond(by_out, node.input[1])
            if mask is None:
                return None
            scatter_masks[node.name] = mask
        elif node.op_type == "NonMaxSuppression":
            rpn_where.update(_match_rpn(graph, by_out, node))

    if not rpn_where or not roi_masks or len(roi_masks) != len(scatter_masks):
        return None
    return roi_masks, scatter_masks, rpn_where, bypass


def _set_live_nodes(graph: onnx.GraphProto, nodes: list[onnx.NodeProto]) -> None:
    # drop what the rewrite left dead (NonZero, Transpose, Squeeze, bypassed Gather)
    live = {o.name for o in graph.output}
    kept: list[onnx.NodeProto] = []
    for node in reversed(nodes):
        if any(o in live for o in node.output):
            kept.append(node)
            live.update(node.input)
    del graph.node[:]
    graph.node.extend(reversed(kept))


def _make_mask(
    cond: str,
    axes: str,
    out: str,
) -> onnx.NodeProto:
    return helper.make_node("Unsqueeze", [cond, axes], [out], name=out)


def remove_hands23_nonzero(model: onnx.ModelProto) -> bool:
    """
    Remove the NonZero ops from an exported Hands23 ONNX model, in place.

    TensorRT runs every NonZero as a blocking, data-dependent-shape copy; Hands23 has nine.
    This pass replaces them with fixed-shape equivalents:

    - the box pooler (two poolers, four FPN levels each) runs ROIAlign over all rois with the
      rois of other levels zeroed, then merges levels with ``Where(level_mask, pooled, prev)``
      instead of ``ScatterND`` on ``NonZero`` indices
    - the RPN zero-area filter becomes ``Where(valid, score, -1e4)`` ahead of the RPN NMS,
      instead of gathering the valid proposals

    Outputs match the original graph. The RPN rewrite assumes at least 1000 proposals are
    valid per image (the graph keeps the top 1000 after NMS); fewer would let invalid boxes
    through. Hands23 at 800 px yields 1700+ in practice. A model that is not the Hands23
    export pattern, or that was already rewritten, is left untouched.

    Parameters
    ----------
    model : onnx.ModelProto
        The exported Hands23 model. Modified in place.

    Returns
    -------
    bool
        True if the model was rewritten, False if it was left untouched.

    """
    graph = model.graph
    match = _match(graph)
    if match is None:
        LOG.debug("hands23 pass: no NonZero pooler pattern found, skipping")
        return False
    roi_masks, scatter_masks, rpn_where, bypass = match

    axes_roi = "hands23_unsqueeze_1"
    axes_pool = "hands23_unsqueeze_123"
    zero = "hands23_zero"
    invalid = "hands23_invalid_score"
    graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([1], np.int64), axes_roi),
            numpy_helper.from_array(np.array([1, 2, 3], np.int64), axes_pool),
            numpy_helper.from_array(np.array(0, np.float32), zero),
            numpy_helper.from_array(np.array(_INVALID_SCORE, np.float32), invalid),
        ]
    )

    # slices read the full box tensor instead of the per-level gather
    gather_src = {n.name: n.input[0] for n in graph.node if n.name in bypass}
    gather_out = {o: gather_src[n.name] for n in graph.node if n.name in bypass for o in n.output}

    nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if node.op_type == "Slice" and node.input[0] in gather_out:
            node.input[0] = gather_out[node.input[0]]
        if node.op_type == "ROIAlign_TRT":
            # new nodes go right before their user, so the node order stays topological
            mask, masked = f"{node.output[0]}_level", f"{node.output[0]}_rois"
            nodes.append(_make_mask(roi_masks[node.name], axes_roi, mask))
            nodes.append(
                helper.make_node("Where", [mask, node.input[1], zero], [masked], name=masked)
            )
            node.input[1] = masked
        elif node.op_type == "ScatterND":
            mask = f"{node.output[0]}_level"
            nodes.append(_make_mask(scatter_masks[node.name], axes_pool, mask))
            prev, _, updates = node.input
            replacement = helper.make_node(
                "Where", [mask, updates, prev], list(node.output), name=node.name
            )
            node.CopyFrom(replacement)
        elif node.op_type == "GatherND" and node.name in rpn_where:
            valid, data = rpn_where[node.name], node.input[0]
            if valid:
                replacement = helper.make_node(
                    "Where", [valid, data, invalid], list(node.output), name=node.name
                )
            else:
                replacement = helper.make_node("Identity", [data], list(node.output), name=node.name)
            node.CopyFrom(replacement)
        nodes.append(node)

    _set_live_nodes(graph, nodes)

    if any(n.op_type == "NonZero" for n in graph.node):
        err_msg = "Hands23 pass left a NonZero in the graph"
        raise RuntimeError(err_msg)
    return True


def fix_hands23_onnx(onnx_path: Path | str, output: Path | str | None = None) -> bool:
    """
    Remove the NonZero ops from a Hands23 ONNX file, see :func:`remove_hands23_nonzero`.

    Parameters
    ----------
    onnx_path : Path | str
        Path to an exported Hands23 ONNX model.
    output : Path | str | None
        Where to write the rewritten model. By default the input file is overwritten.

    Returns
    -------
    bool
        True if the model was rewritten and saved, False if it was left untouched (not a
        Hands23 export, or already rewritten). Nothing is written in that case.

    """
    model = onnx.load(str(onnx_path))
    if not remove_hands23_nonzero(model):
        return False
    onnx.save(model, str(output if output is not None else onnx_path))
    return True
