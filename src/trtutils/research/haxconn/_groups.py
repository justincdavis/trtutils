# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Layer grouping for HaX-CoNN: where a DNN may move between accelerators."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

    from trtutils.inspect._types import LayerInfo

_FOREIGN_NODE = "{ForeignNode["


class LayerGroup(NamedTuple):
    """
    A contiguous range of network layers that always runs on one accelerator.

    Attributes
    ----------
    first : int
        The index of the first network layer.
    last : int
        The index of the last network layer.
    dla : bool
        Whether every data layer of the group can run on a DLA.

    """

    first: int
    last: int
    dla: bool


def constant_layers(
    layer_types: Sequence[str],
    inputs: Sequence[Sequence[str]],
    outputs: Sequence[Sequence[str]],
) -> set[int]:
    """
    Find the layers whose outputs depend only on constants.

    Parameters
    ----------
    layer_types : Sequence[str]
        The TensorRT layer type of each layer, e.g. ``"CONSTANT"``.
    inputs : Sequence[Sequence[str]]
        The names of the tensors each layer reads.
    outputs : Sequence[Sequence[str]]
        The names of the tensors each layer writes.

    Returns
    -------
    set[int]
        The indices of the constant layers: CONSTANT layers, layers without inputs,
        and layers that read only constant tensors. Network layers are not in
        topological order, so this iterates to a fixed point.

    """
    constant = {i for i, t in enumerate(layer_types) if t == "CONSTANT" or not inputs[i]}
    tensors = {name for i in constant for name in outputs[i]}
    changed = True
    while changed:
        changed = False
        for i, names in enumerate(inputs):
            if i not in constant and all(name in tensors for name in names):
                constant.add(i)
                tensors.update(outputs[i])
                changed = True
    return constant


def match_engine_layers(
    engine_layers: Sequence[str],
    layers: Sequence[LayerInfo],
) -> list[list[int]]:
    """
    Find the network layers each engine layer runs.

    Parameters
    ----------
    engine_layers : Sequence[str]
        The layer names of a built engine, e.g. from :func:`profile_engine`. TensorRT
        8.6 writes fusions as ``A + B``, ``A || B`` or ``PWN(PWN(A), B)``.
    layers : Sequence[LayerInfo]
        The network layers the engine was built from.

    Returns
    -------
    list[list[int]]
        Per engine layer, the indices of the network layers whose names it contains as
        whole names; a ``{ForeignNode[first...last]}`` subgraph runs every layer from
        ``first`` to ``last``.

    """
    patterns = [
        # whole names only; "..." separates the ends of a subgraph name
        (
            layer.index,
            re.compile(r"(?<![\w/])(?<![^.]\.)" + re.escape(layer.name) + r"(?![\w/]|\.(?!\.\.))"),
        )
        for layer in layers
    ]
    matches: list[list[int]] = []
    for name in engine_layers:
        matched = [index for index, pattern in patterns if pattern.search(name)]
        # a subgraph is named after its first and last layers and runs everything between
        if name.startswith(_FOREIGN_NODE) and matched:
            matched = list(range(min(matched), max(matched) + 1))
        matches.append(matched)
    return matches


def fused_spans(engine_layers: Sequence[str], layers: Sequence[LayerInfo]) -> list[tuple[int, int]]:
    """
    Find the network layer ranges TensorRT fused into one engine layer.

    Parameters
    ----------
    engine_layers : Sequence[str]
        The layer names of a built engine, e.g. from :func:`profile_engine`.
    layers : Sequence[LayerInfo]
        The network layers the engine was built from.

    Returns
    -------
    list[tuple[int, int]]
        The (lowest, highest) network layer index named by each engine layer that
        names more than one network layer.

    """
    return _spans(match_engine_layers(engine_layers, layers))


def _spans(matches: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    # the fused ranges in the output of match_engine_layers
    return [(min(matched), max(matched)) for matched in matches if len(matched) > 1]


def _constants(layers: Sequence[LayerInfo]) -> set[int]:
    return constant_layers(
        [layer.layer_type for layer in layers],
        [layer.inputs for layer in layers],
        [layer.outputs for layer in layers],
    )


def transition_points(
    layers: Sequence[LayerInfo],
    spans: Sequence[tuple[int, int]] = (),
) -> list[int]:
    """
    Find the boundaries where a DNN may move to the other accelerator.

    Boundary ``k`` lies between layers ``k - 1`` and ``k``. It is a transition point
    when exactly one data tensor (not derived from constants) crosses it, from a
    layer before it to a layer after it, and no fused engine layer spans it.

    Parameters
    ----------
    layers : Sequence[LayerInfo]
        The network layers, from :func:`inspect_onnx_layers`.
    spans : Sequence[tuple[int, int]], optional
        The fused ranges, from :func:`fused_spans`. Default is ().

    Returns
    -------
    list[int]
        The transition points, ascending.

    """
    return _points(layers, spans, _constants(layers))


def _points(
    layers: Sequence[LayerInfo], spans: Sequence[tuple[int, int]], constant: set[int]
) -> list[int]:
    producer = {name: layer.index for layer in layers for name in layer.outputs}
    # every data edge as (producer, consumer, tensor); network inputs come from -1
    edges = [
        (producer.get(name, -1), layer.index, name)
        for layer in layers
        if layer.index not in constant
        for name in layer.inputs
        if producer.get(name, -1) not in constant
    ]
    points: list[int] = []
    for k in range(1, len(layers)):
        if any(lo < k <= hi for lo, hi in spans):
            continue
        crossing = {(p, name) for p, c, name in edges if min(p, c) < k <= max(p, c)}
        if len({name for _, name in crossing}) == 1 and all(p < k for p, _ in crossing):
            points.append(k)
    return points


def group_layers(
    layers: Sequence[LayerInfo],
    weights: Sequence[float] | None = None,
    spans: Sequence[tuple[int, int]] = (),
    max_groups: int | None = None,
) -> list[LayerGroup]:
    """
    Split a network into layer groups at its transition points (HaX-CoNN Sec. 3.1).

    A group is DLA-capable when every layer of it that is not derived from constants
    passes TensorRT's DLA check. Groups without data layers join their neighbour.
    With ``max_groups``, the adjacent pair with the smallest combined weight is merged
    until the cap holds, preferring pairs that agree on DLA capability.

    Parameters
    ----------
    layers : Sequence[LayerInfo]
        The network layers, from :func:`inspect_onnx_layers`.
    weights : Sequence[float], optional
        A weight per layer for merging, e.g. its GPU time. Default is None (1 each).
    spans : Sequence[tuple[int, int]], optional
        The fused ranges, from :func:`fused_spans`. Default is ().
    max_groups : int, optional
        The most groups to return. Default is None (no cap).

    Returns
    -------
    list[LayerGroup]
        The groups, in network order, covering every layer.

    Raises
    ------
    ValueError
        If there are no layers or ``max_groups`` is below 1.

    """
    if len(layers) == 0:
        err_msg = "group_layers needs at least one layer"
        raise ValueError(err_msg)
    if max_groups is not None and max_groups < 1:
        err_msg = f"max_groups must be at least 1, got {max_groups}"
        raise ValueError(err_msg)
    weight = list(weights) if weights is not None else [1.0] * len(layers)
    constant = _constants(layers)
    bounds = [0, *_points(layers, spans, constant), len(layers)]
    ranges = [(a, b - 1) for a, b in zip(bounds, bounds[1:])]

    # groups of constants only join the next group (the last joins the previous one)
    merged: list[tuple[int, int]] = []
    carry: int | None = None
    for first, last in ranges:
        start = carry if carry is not None else first
        if all(i in constant for i in range(first, last + 1)):
            carry = start
            continue
        merged.append((start, last))
        carry = None
    if carry is not None:
        if merged:
            merged[-1] = (merged[-1][0], len(layers) - 1)
        else:
            merged.append((0, len(layers) - 1))

    def _dla(first: int, last: int) -> bool:
        data = [layers[i] for i in range(first, last + 1) if i not in constant]
        return all(layer.dla_compatible for layer in data)

    groups = [LayerGroup(first, last, _dla(first, last)) for first, last in merged]
    while max_groups is not None and len(groups) > max_groups:
        costs = [
            (
                a.dla != b.dla,
                sum(weight[a.first : b.last + 1]),
                k,
            )
            for k, (a, b) in enumerate(zip(groups, groups[1:]))
        ]
        _, _, k = min(costs)
        first, last = groups[k].first, groups[k + 1].last
        groups[k : k + 2] = [LayerGroup(first, last, _dla(first, last))]
    return groups
