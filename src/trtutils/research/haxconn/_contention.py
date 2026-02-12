# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""PCCS contention model for HaX-CoNN (Section 3.3)."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from trtutils._log import LOG

from ._types import ProcessorType

if TYPE_CHECKING:
    from ._types import DNNSchedule, HaxconnConfig, LayerGroup, LayerGroupCost, MultiSchedule


def compute_contention_slowdown(
    throughput_a: float,
    throughput_b: float,
    config: HaxconnConfig,
) -> float:
    """
    Compute the PCCS contention slowdown multiplier (Eq. 7).

    When two layer groups from different DNNs run concurrently on different
    accelerators (one on GPU, one on DLA), they contend for shared memory
    bandwidth. The PCCS model estimates the resulting slowdown.

    Parameters
    ----------
    throughput_a : float
        Memory throughput of the first group in MB/s.
    throughput_b : float
        Memory throughput of the second group in MB/s.
    config : HaxconnConfig
        Configuration with PCCS alpha, beta, and max_bandwidth.

    Returns
    -------
    float
        Slowdown multiplier >= 1.0.

    """
    if config.max_bandwidth_mbps <= 0:
        return 1.0

    combined_ratio = (throughput_a + throughput_b) / config.max_bandwidth_mbps
    slowdown = 1.0 + config.pccs_alpha * (combined_ratio**config.pccs_beta)
    return max(1.0, slowdown)


def compute_contention_interval(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    """
    Compute the overlap interval between two execution windows (Eq. 8).

    Parameters
    ----------
    start_a : float
        Start time of the first execution window.
    end_a : float
        End time of the first execution window.
    start_b : float
        Start time of the second execution window.
    end_b : float
        End time of the second execution window.

    Returns
    -------
    float
        Duration of overlap (>= 0).

    """
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _get_group_mem_throughput(
    group: LayerGroup,
    costs: dict[int, LayerGroupCost],
    *,
    gpu: bool,
) -> float:
    """Get memory throughput for a group on its assigned processor."""
    cost = costs[group.group_id]
    if gpu:
        return cost.gpu_mem_throughput_mbps
    return cost.dla_mem_throughput_mbps if cost.dla_mem_throughput_mbps is not None else 0.0


def estimate_contention_for_schedule(
    all_groups: list[list[LayerGroup]],
    all_costs: list[dict[int, LayerGroupCost]],
    multi_schedule: MultiSchedule,
    config: HaxconnConfig,
    *,
    verbose: bool | None = None,
) -> MultiSchedule:
    """
    Adjust execution times in a MultiSchedule to account for memory contention.

    For each pair of overlapping groups from different DNNs on different
    accelerators, applies the PCCS model to compute slowdown and adjusts
    execution times. Iterates until convergence (max time delta < epsilon).

    Parameters
    ----------
    all_groups : list[list[LayerGroup]]
        Groups for each DNN, indexed by DNN id.
    all_costs : list[dict[int, LayerGroupCost]]
        Costs for each DNN's groups, indexed by DNN id.
    multi_schedule : MultiSchedule
        The schedule to adjust (will be deep-copied).
    config : HaxconnConfig
        Configuration with PCCS parameters and convergence thresholds.
    verbose : bool | None, optional
        Whether to print verbose output.

    Returns
    -------
    MultiSchedule
        A new MultiSchedule with contention-adjusted times.

    """
    result = copy.deepcopy(multi_schedule)
    num_dnns = result.num_dnns

    _min_dnns_for_contention = 2
    if num_dnns < _min_dnns_for_contention:
        return result

    # Build flat list of (dnn_id, group) pairs for cross-DNN comparison
    dnn_groups: list[tuple[int, LayerGroup]] = [
        (group.dnn_id, group) for groups in all_groups for group in groups
    ]

    for iteration in range(config.contention_max_iters):
        max_delta = 0.0

        for i in range(len(dnn_groups)):
            dnn_i, group_i = dnn_groups[i]
            sched_i = result.get_schedule(dnn_i)

            if group_i.group_id not in sched_i.group_start_times:
                continue

            proc_i = sched_i.get_processor(group_i.group_id)
            start_i = sched_i.group_start_times[group_i.group_id]
            end_i = sched_i.group_end_times[group_i.group_id]

            for j in range(i + 1, len(dnn_groups)):
                dnn_j, group_j = dnn_groups[j]

                # Only cross-DNN contention
                if dnn_i == dnn_j:
                    continue

                sched_j = result.get_schedule(dnn_j)

                if group_j.group_id not in sched_j.group_start_times:
                    continue

                proc_j = sched_j.get_processor(group_j.group_id)

                # Contention only when on different accelerators
                if proc_i == proc_j:
                    continue

                start_j = sched_j.group_start_times[group_j.group_id]
                end_j = sched_j.group_end_times[group_j.group_id]

                overlap = compute_contention_interval(start_i, end_i, start_j, end_j)
                if overlap <= 0:
                    continue

                # Get memory throughputs
                tput_i = _get_group_mem_throughput(
                    group_i,
                    all_costs[dnn_i],
                    gpu=proc_i == ProcessorType.GPU,
                )
                tput_j = _get_group_mem_throughput(
                    group_j,
                    all_costs[dnn_j],
                    gpu=proc_j == ProcessorType.GPU,
                )

                slowdown = compute_contention_slowdown(tput_i, tput_j, config)

                if slowdown <= 1.0:
                    continue

                # Apply slowdown to the overlapping portion (Eq. 2)
                extra_time_i = overlap * (slowdown - 1.0)
                extra_time_j = overlap * (slowdown - 1.0)

                # Adjust end times for group_i
                sched_i.group_end_times[group_i.group_id] += extra_time_i
                max_delta = max(max_delta, extra_time_i)

                # Cascade to subsequent groups in DNN i
                _cascade_timing(
                    sched_i,
                    all_groups[dnn_i],
                    group_i.group_id,
                    extra_time_i,
                )

                # Adjust end times for group_j
                sched_j.group_end_times[group_j.group_id] += extra_time_j
                max_delta = max(max_delta, extra_time_j)

                # Cascade to subsequent groups in DNN j
                _cascade_timing(
                    sched_j,
                    all_groups[dnn_j],
                    group_j.group_id,
                    extra_time_j,
                )

        if verbose:
            LOG.info(f"Contention iteration {iteration + 1}: max_delta={max_delta:.4f}ms")

        if max_delta < config.contention_epsilon:
            break

    # Recompute total times
    for sched in result.dnn_schedules:
        if sched.group_end_times:
            sched.total_time_ms = max(sched.group_end_times.values())

    return result


def _cascade_timing(
    schedule: DNNSchedule,
    groups: list[LayerGroup],
    after_group_id: int,
    delta: float,
) -> None:
    """Cascade a timing change to all subsequent groups in a DNN schedule."""
    # Find groups that come after the given group
    sorted_groups = sorted(groups, key=lambda g: g.group_id)
    found = False

    for group in sorted_groups:
        if group.group_id == after_group_id:
            found = True
            continue
        if found and group.group_id in schedule.group_start_times:
            schedule.group_start_times[group.group_id] += delta
            schedule.group_end_times[group.group_id] += delta
