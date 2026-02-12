# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Layer grouping for HaX-CoNN optimization (Section 3.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._types import Layer, LayerGroup


def identify_layer_groups(
    layers: list[Layer],
    dnn_id: int,
) -> list[LayerGroup]:
    """
    Group contiguous layers with the same DLA compatibility into LayerGroups.

    A new group starts whenever ``can_run_on_dla`` changes between adjacent
    layers. Each group becomes an atomic scheduling unit for HaX-CoNN.

    Parameters
    ----------
    layers : list[Layer]
        Layers from a single DNN, sorted by index.
    dnn_id : int
        The DNN identifier to assign to each group.

    Returns
    -------
    list[LayerGroup]
        List of LayerGroup objects representing atomic scheduling units.

    """
    from ._types import LayerGroup  # noqa: PLC0415

    if not layers:
        return []

    groups: list[LayerGroup] = []
    group_id = 0

    current_dla = layers[0].can_run_on_dla
    current_indices: list[int] = [layers[0].index]
    current_output_size = layers[0].output_tensor_size
    current_input_size = layers[0].input_tensor_size

    for layer in layers[1:]:
        if layer.can_run_on_dla != current_dla:
            # Finish current group
            groups.append(
                LayerGroup(
                    group_id=group_id,
                    dnn_id=dnn_id,
                    layer_indices=current_indices,
                    can_run_on_dla=current_dla,
                    total_output_tensor_size=current_output_size,
                    total_input_tensor_size=current_input_size,
                )
            )
            group_id += 1
            current_dla = layer.can_run_on_dla
            current_indices = [layer.index]
            current_output_size = layer.output_tensor_size
            current_input_size = layer.input_tensor_size
        else:
            current_indices.append(layer.index)
            current_output_size += layer.output_tensor_size
            current_input_size += layer.input_tensor_size

    # Finish last group
    groups.append(
        LayerGroup(
            group_id=group_id,
            dnn_id=dnn_id,
            layer_indices=current_indices,
            can_run_on_dla=current_dla,
            total_output_tensor_size=current_output_size,
            total_input_tensor_size=current_input_size,
        )
    )

    return groups


def assign_groups_to_layers(
    layers: list[Layer],
    groups: list[LayerGroup],
) -> list[Layer]:
    """
    Set the ``group_id`` on each Layer based on the identified groups.

    Parameters
    ----------
    layers : list[Layer]
        Layers to update.
    groups : list[LayerGroup]
        Layer groups to assign from.

    Returns
    -------
    list[Layer]
        The same layers with ``group_id`` set.

    """
    # Build index -> group_id mapping
    idx_to_group: dict[int, int] = {}
    for group in groups:
        for idx in group.layer_indices:
            idx_to_group[idx] = group.group_id

    for layer in layers:
        layer.group_id = idx_to_group.get(layer.index)

    return layers
