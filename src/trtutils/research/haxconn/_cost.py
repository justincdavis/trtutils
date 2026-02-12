# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Cost models for HaX-CoNN optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._types import DNNSchedule, MultiSchedule, ProcessorType

if TYPE_CHECKING:
    from ._types import HaxconnConfig, LayerGroup, LayerGroupCost


def estimate_transition_cost(
    group: LayerGroup,
    from_proc: ProcessorType,
    to_proc: ProcessorType,
    config: HaxconnConfig,
) -> float:
    """
    Estimate transition time in ms when switching a group between processors.

    The transition cost is based on the group output tensor size, modeling
    the MarkOutput/addInput overhead for TensorRT engine boundaries.

    Parameters
    ----------
    group : LayerGroup
        The group whose output is being transitioned.
    from_proc : ProcessorType
        The source processor.
    to_proc : ProcessorType
        The destination processor.
    config : HaxconnConfig
        Configuration with transition cost parameters.

    Returns
    -------
    float
        Estimated transition time in milliseconds.

    """
    if from_proc == to_proc:
        return 0.0

    tensor_size_mb = group.total_output_tensor_size / (1024 * 1024)
    return config.transition_time_base_ms + (tensor_size_mb * config.transition_time_per_mb)


def compute_group_time(
    cost: LayerGroupCost,
    processor: ProcessorType,
    group_id: int | None = None,
) -> float:
    """
    Get the execution time for a group on a specific processor.

    Parameters
    ----------
    cost : LayerGroupCost
        The cost data for the group.
    processor : ProcessorType
        The processor to get time for.
    group_id : int | None, optional
        Group ID for error messages.

    Returns
    -------
    float
        Execution time in milliseconds.

    Raises
    ------
    ValueError
        If DLA time requested but group is not DLA-compatible.

    """
    if processor == ProcessorType.GPU:
        return cost.gpu_time_ms
    if cost.dla_time_ms is None:
        id_str = f"Group {group_id}" if group_id is not None else "Group"
        err_msg = f"{id_str} is not DLA-compatible"
        raise ValueError(err_msg)
    return cost.dla_time_ms


def compute_dnn_time(
    groups: list[LayerGroup],
    costs: dict[int, LayerGroupCost],
    schedule: DNNSchedule,
    config: HaxconnConfig,
) -> float:
    """
    Compute total DNN latency from Eq. 2 (standalone + transitions).

    This is the pre-contention time: sum of group execution times plus
    transition costs between consecutive groups on different processors.

    Parameters
    ----------
    groups : list[LayerGroup]
        Layer groups for this DNN, sorted by group_id.
    costs : dict[int, LayerGroupCost]
        Cost data keyed by group_id.
    schedule : DNNSchedule
        The group-to-processor assignments.
    config : HaxconnConfig
        Configuration with transition cost parameters.

    Returns
    -------
    float
        Total DNN latency in milliseconds.

    """
    sorted_groups = sorted(groups, key=lambda g: g.group_id)
    total_time = 0.0

    for i, group in enumerate(sorted_groups):
        proc = schedule.get_processor(group.group_id)
        total_time += compute_group_time(costs[group.group_id], proc, group.group_id)

        # Add transition cost
        if i > 0:
            prev_group = sorted_groups[i - 1]
            prev_proc = schedule.get_processor(prev_group.group_id)
            if prev_proc != proc:
                total_time += estimate_transition_cost(prev_group, prev_proc, proc, config)

    return total_time


def compute_dnn_timing(
    groups: list[LayerGroup],
    costs: dict[int, LayerGroupCost],
    schedule: DNNSchedule,
    config: HaxconnConfig,
) -> DNNSchedule:
    """
    Compute per-group start/end times for a DNN schedule.

    Populates ``group_start_times`` and ``group_end_times`` on the schedule.

    Parameters
    ----------
    groups : list[LayerGroup]
        Layer groups for this DNN, sorted by group_id.
    costs : dict[int, LayerGroupCost]
        Cost data keyed by group_id.
    schedule : DNNSchedule
        The group-to-processor assignments (modified in-place).
    config : HaxconnConfig
        Configuration with transition cost parameters.

    Returns
    -------
    DNNSchedule
        The same schedule with timing populated.

    """
    sorted_groups = sorted(groups, key=lambda g: g.group_id)
    current_time = 0.0

    for i, group in enumerate(sorted_groups):
        proc = schedule.get_processor(group.group_id)

        # Add transition cost
        if i > 0:
            prev_group = sorted_groups[i - 1]
            prev_proc = schedule.get_processor(prev_group.group_id)
            if prev_proc != proc:
                current_time += estimate_transition_cost(prev_group, prev_proc, proc, config)

        schedule.group_start_times[group.group_id] = current_time
        group_time = compute_group_time(costs[group.group_id], proc, group.group_id)
        current_time += group_time
        schedule.group_end_times[group.group_id] = current_time

    schedule.total_time_ms = current_time
    return schedule


def compute_throughput_objective(multi_schedule: MultiSchedule) -> float:
    """
    Compute the throughput objective (Eq. 10): sum of all DNN latencies.

    Parameters
    ----------
    multi_schedule : MultiSchedule
        The combined multi-DNN schedule.

    Returns
    -------
    float
        Sum of all DNN total times in milliseconds.

    """
    return sum(sched.total_time_ms for sched in multi_schedule.dnn_schedules)


def compute_max_latency_objective(multi_schedule: MultiSchedule) -> float:
    """
    Compute the max-latency objective (Eq. 11): maximum DNN latency.

    Parameters
    ----------
    multi_schedule : MultiSchedule
        The combined multi-DNN schedule.

    Returns
    -------
    float
        Maximum DNN total time in milliseconds.

    """
    if not multi_schedule.dnn_schedules:
        return 0.0
    return max(sched.total_time_ms for sched in multi_schedule.dnn_schedules)


def create_naive_schedule(
    all_groups: list[list[LayerGroup]],
    all_costs: list[dict[int, LayerGroupCost]],
    config: HaxconnConfig,
) -> MultiSchedule:
    """
    Create an all-GPU baseline schedule for all DNNs.

    Parameters
    ----------
    all_groups : list[list[LayerGroup]]
        Groups for each DNN, indexed by DNN id.
    all_costs : list[dict[int, LayerGroupCost]]
        Costs for each DNN's groups, indexed by DNN id.
    config : HaxconnConfig
        Configuration with transition cost parameters.

    Returns
    -------
    MultiSchedule
        An all-GPU schedule with timing populated.

    """
    multi = MultiSchedule()

    for dnn_id, (groups, costs) in enumerate(zip(all_groups, all_costs)):
        sched = DNNSchedule(dnn_id=dnn_id)
        for group in groups:
            sched.set_processor(group.group_id, ProcessorType.GPU)

        compute_dnn_timing(groups, costs, sched, config)
        multi.dnn_schedules.append(sched)

    multi.throughput_objective = compute_throughput_objective(multi)
    multi.max_latency_objective = compute_max_latency_objective(multi)
    return multi
