# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Cost models for AxoNN optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._types import AxoNNConfig, ProcessorType, TransitionCost

if TYPE_CHECKING:
    from ._types import Layer, LayerCost, Schedule


def estimate_transition_cost(
    layer: Layer,
    from_processor: ProcessorType,
    to_processor: ProcessorType,
    config: AxoNNConfig,
) -> TransitionCost:
    """
    Estimate the cost of transitioning between processors.

    The transition cost includes:
    - Cache flush overhead from source processor
    - Cold cache miss penalty on destination processor
    - Memory transfer overhead based on tensor size

    Parameters
    ----------
    layer : Layer
        The layer whose output is being transitioned.
    from_processor : ProcessorType
        The source processor.
    to_processor : ProcessorType
        The destination processor.
    config : AxoNNConfig
        Configuration with transition cost parameters.

    Returns
    -------
    TransitionCost
        The estimated transition cost.

    """
    if from_processor == to_processor:
        return TransitionCost(
            from_processor=from_processor,
            to_processor=to_processor,
            time_ms=0.0,
            energy_mj=0.0,
        )

    # Convert bytes to megabytes
    tensor_size_mb = layer.output_tensor_size / (1024 * 1024)

    # Estimate time: base overhead + size-dependent overhead
    time_ms = config.transition_time_base_ms + (tensor_size_mb * config.transition_time_per_mb)

    # Estimate energy: base overhead + size-dependent overhead
    energy_mj = config.transition_energy_base_mj + (tensor_size_mb * config.transition_energy_per_mb)

    return TransitionCost(
        from_processor=from_processor,
        to_processor=to_processor,
        time_ms=time_ms,
        energy_mj=energy_mj,
    )


def compute_layer_time(
    layer_cost: LayerCost,
    processor: ProcessorType,
) -> float:
    """
    Get the execution time for a layer on a specific processor.

    Parameters
    ----------
    layer_cost : LayerCost
        The cost data for the layer.
    processor : ProcessorType
        The processor to get time for.

    Returns
    -------
    float
        Execution time in milliseconds.

    Raises
    ------
    ValueError
        If DLA time requested but layer is not DLA-compatible.

    """
    if processor == ProcessorType.GPU:
        return layer_cost.gpu_time_ms
    if layer_cost.dla_time_ms is None:
        err_msg = f"Layer {layer_cost.layer_idx} ({layer_cost.layer_name}) is not DLA-compatible"
        raise ValueError(err_msg)
    return layer_cost.dla_time_ms


def compute_layer_energy(
    layer_cost: LayerCost,
    processor: ProcessorType,
) -> float:
    """
    Get the energy consumption for a layer on a specific processor.

    Parameters
    ----------
    layer_cost : LayerCost
        The cost data for the layer.
    processor : ProcessorType
        The processor to get energy for.

    Returns
    -------
    float
        Energy consumption in millijoules.

    Raises
    ------
    ValueError
        If DLA energy requested but layer is not DLA-compatible.

    """
    if processor == ProcessorType.GPU:
        return layer_cost.gpu_energy_mj
    if layer_cost.dla_energy_mj is None:
        err_msg = f"Layer {layer_cost.layer_idx} ({layer_cost.layer_name}) is not DLA-compatible"
        raise ValueError(err_msg)
    return layer_cost.dla_energy_mj


def compute_total_time(
    layers: list[Layer],
    costs: list[LayerCost],
    schedule: Schedule,
    config: AxoNNConfig,
) -> float:
    """
    Compute the total execution time for a schedule.

    Total time includes:
    - Layer execution times on assigned processors
    - Transition costs between processors

    Parameters
    ----------
    layers : list[Layer]
        The network layers.
    costs : list[LayerCost]
        Cost data for each layer.
    schedule : Schedule
        The layer-to-processor assignments.
    config : AxoNNConfig
        Configuration with transition cost parameters.

    Returns
    -------
    float
        Total execution time in milliseconds.

    """
    # Build lookup by layer index
    cost_by_idx = {c.layer_idx: c for c in costs}
    layer_by_idx = {l.index: l for l in layers}

    total_time = 0.0
    sorted_indices = sorted(schedule.assignments.keys())

    for i, layer_idx in enumerate(sorted_indices):
        processor = schedule.assignments[layer_idx]
        layer_cost = cost_by_idx[layer_idx]

        # Add layer execution time
        total_time += compute_layer_time(layer_cost, processor)

        # Add transition cost if processor changes
        if i > 0:
            prev_idx = sorted_indices[i - 1]
            prev_processor = schedule.assignments[prev_idx]
            if prev_processor != processor:
                layer = layer_by_idx[prev_idx]
                transition = estimate_transition_cost(layer, prev_processor, processor, config)
                total_time += transition.time_ms

    return total_time


def compute_total_energy(
    layers: list[Layer],
    costs: list[LayerCost],
    schedule: Schedule,
    config: AxoNNConfig,
) -> float:
    """
    Compute the total energy consumption for a schedule.

    Total energy includes:
    - Layer energy consumption on assigned processors
    - Transition energy costs

    Parameters
    ----------
    layers : list[Layer]
        The network layers.
    costs : list[LayerCost]
        Cost data for each layer.
    schedule : Schedule
        The layer-to-processor assignments.
    config : AxoNNConfig
        Configuration with transition cost parameters.

    Returns
    -------
    float
        Total energy consumption in millijoules.

    """
    # Build lookup by layer index
    cost_by_idx = {c.layer_idx: c for c in costs}
    layer_by_idx = {l.index: l for l in layers}

    total_energy = 0.0
    sorted_indices = sorted(schedule.assignments.keys())

    for i, layer_idx in enumerate(sorted_indices):
        processor = schedule.assignments[layer_idx]
        layer_cost = cost_by_idx[layer_idx]

        # Add layer energy consumption
        total_energy += compute_layer_energy(layer_cost, processor)

        # Add transition cost if processor changes
        if i > 0:
            prev_idx = sorted_indices[i - 1]
            prev_processor = schedule.assignments[prev_idx]
            if prev_processor != processor:
                layer = layer_by_idx[prev_idx]
                transition = estimate_transition_cost(layer, prev_processor, processor, config)
                total_energy += transition.energy_mj

    return total_energy


def compute_gpu_only_costs(
    costs: list[LayerCost],
) -> tuple[float, float]:
    """
    Compute total time and energy if all layers run on GPU.

    Parameters
    ----------
    costs : list[LayerCost]
        Cost data for each layer.

    Returns
    -------
    tuple[float, float]
        Total time (ms) and total energy (mJ) for GPU-only execution.

    """
    total_time = sum(c.gpu_time_ms for c in costs)
    total_energy = sum(c.gpu_energy_mj for c in costs)
    return total_time, total_energy


def compute_dla_only_costs(
    costs: list[LayerCost],
) -> tuple[float, float]:
    """
    Compute total time and energy if all DLA-compatible layers run on DLA.

    Layers without DLA support are assumed to run on GPU.

    Parameters
    ----------
    costs : list[LayerCost]
        Cost data for each layer.

    Returns
    -------
    tuple[float, float]
        Total time (ms) and total energy (mJ) for DLA-preferred execution.

    """
    total_time = 0.0
    total_energy = 0.0

    for c in costs:
        if c.dla_time_ms is not None:
            total_time += c.dla_time_ms
            total_energy += c.dla_energy_mj if c.dla_energy_mj is not None else c.gpu_energy_mj
        else:
            total_time += c.gpu_time_ms
            total_energy += c.gpu_energy_mj

    return total_time, total_energy


def create_gpu_only_schedule(layers: list[Layer]) -> Schedule:
    """
    Create a schedule with all layers on GPU.

    Parameters
    ----------
    layers : list[Layer]
        The network layers.

    Returns
    -------
    Schedule
        A schedule with all layers assigned to GPU.

    """
    schedule = Schedule()
    for layer in layers:
        schedule.set_processor(layer.index, ProcessorType.GPU)
    schedule.count_transitions()
    return schedule


def create_dla_preferred_schedule(layers: list[Layer]) -> Schedule:
    """
    Create a schedule with DLA-compatible layers on DLA, others on GPU.

    Parameters
    ----------
    layers : list[Layer]
        The network layers.

    Returns
    -------
    Schedule
        A schedule with DLA-compatible layers on DLA.

    """
    schedule = Schedule()
    for layer in layers:
        if layer.can_run_on_dla:
            schedule.set_processor(layer.index, ProcessorType.DLA)
        else:
            schedule.set_processor(layer.index, ProcessorType.GPU)
    schedule.count_transitions()
    return schedule
