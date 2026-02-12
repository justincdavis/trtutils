# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Z3-based solver for HaX-CoNN multi-DNN scheduling (Section 3.4)."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from trtutils._log import LOG

from ._contention import estimate_contention_for_schedule
from ._cost import (
    compute_dnn_timing,
    compute_max_latency_objective,
    compute_throughput_objective,
    create_naive_schedule,
    estimate_transition_cost,
)
from ._types import DNNSchedule, HaxconnConfig, MultiSchedule, Objective, ProcessorType

if TYPE_CHECKING:
    from ._types import LayerGroup, LayerGroupCost

# Try to import z3, but make it optional
try:
    from z3 import And, If, Int, Optimize, Real, sat

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


def _check_z3_available() -> None:
    """Check if Z3 is available and raise an error if not."""
    if not Z3_AVAILABLE:
        err_msg = (
            "Z3 solver is not available. Install with: pip install z3-solver "
            "or pip install trtutils[haxconn]"
        )
        raise ImportError(err_msg)


def _build_multi_schedule(
    all_groups: list[list[LayerGroup]],
    all_costs: list[dict[int, LayerGroupCost]],
    assignments: list[dict[int, ProcessorType]],
    config: HaxconnConfig,
) -> MultiSchedule:
    """Build a MultiSchedule from processor assignments and compute timing."""
    multi = MultiSchedule()

    for dnn_id, (groups, costs) in enumerate(zip(all_groups, all_costs)):
        sched = DNNSchedule(dnn_id=dnn_id, assignments=assignments[dnn_id])
        compute_dnn_timing(groups, costs, sched, config)
        multi.dnn_schedules.append(sched)

    multi.throughput_objective = compute_throughput_objective(multi)
    multi.max_latency_objective = compute_max_latency_objective(multi)
    return multi


def solve_schedule(
    all_groups: list[list[LayerGroup]],
    all_costs: list[dict[int, LayerGroupCost]],
    config: HaxconnConfig,
    *,
    verbose: bool | None = None,
) -> MultiSchedule | None:
    """
    Find optimal multi-DNN schedule using Z3 SMT solver.

    Formulates the HaX-CoNN scheduling problem with:
    - Decision vars: ``proc[n][g]`` in {0=GPU, 1=DLA} per group per DNN
    - Constraints: DLA-incompatible groups forced to GPU, sequential within DNN
    - Objective: minimize sum(DNN_time) or minimize max(DNN_time)

    Uses iterative linearization for contention: solves without contention,
    applies PCCS model, re-solves with adjusted times, repeats until convergence.

    Parameters
    ----------
    all_groups : list[list[LayerGroup]]
        Groups for each DNN.
    all_costs : list[dict[int, LayerGroupCost]]
        Cost data for each DNN.
    config : HaxconnConfig
        Configuration with objective, PCCS params, Z3 timeout.
    verbose : bool | None, optional
        Whether to print verbose output.

    Returns
    -------
    MultiSchedule | None
        The optimal schedule, or None if no feasible solution found.

    """
    _check_z3_available()

    num_dnns = len(all_groups)

    if verbose:
        LOG.info(f"Solving for {num_dnns} DNNs with Z3 ({config.objective.value})")

    # Iterative linearization: start with standalone times, iterate with contention
    adjusted_costs = [dict(costs) for costs in all_costs]
    best_schedule: MultiSchedule | None = None

    for lin_iter in range(3):
        if verbose:
            LOG.info(f"Linearization iteration {lin_iter + 1}")

        opt = Optimize()
        opt.set("timeout", config.z3_timeout_ms)

        # Decision variables: processor per group per DNN
        proc_vars: list[list[Int]] = []  # type: ignore[type-arg]
        for n in range(num_dnns):
            dnn_vars = []
            for _g, group in enumerate(all_groups[n]):
                var = Int(f"proc_{n}_{group.group_id}")
                opt.add(And(var >= 0, var <= 1))

                # DLA-incompatible groups must use GPU
                if not group.can_run_on_dla:
                    opt.add(var == 0)

                dnn_vars.append(var)
            proc_vars.append(dnn_vars)

        # Time variables and expressions per DNN
        dnn_time_vars = []

        for n in range(num_dnns):
            groups = all_groups[n]
            costs = adjusted_costs[n]
            sorted_groups = sorted(groups, key=lambda g: g.group_id)

            layer_times = []
            transition_times = []

            for i, group in enumerate(sorted_groups):
                cost = costs[group.group_id]
                gpu_time = cost.gpu_time_ms
                dla_time = cost.dla_time_ms if cost.dla_time_ms is not None else gpu_time

                group_time = If(
                    proc_vars[n][i] == 0,
                    gpu_time,
                    dla_time,
                )
                layer_times.append(group_time)

                # Transition cost between consecutive groups
                if i > 0:
                    prev_group = sorted_groups[i - 1]
                    # Estimate max transition cost (when processors differ)
                    trans_time = estimate_transition_cost(
                        prev_group, ProcessorType.GPU, ProcessorType.DLA, config
                    )
                    # Add transition if processors differ
                    trans_expr = If(
                        proc_vars[n][i - 1] != proc_vars[n][i],
                        trans_time,
                        0.0,
                    )
                    transition_times.append(trans_expr)

            total_time_expr = sum(layer_times)
            if transition_times:
                total_time_expr = total_time_expr + sum(transition_times)

            dnn_time = Real(f"dnn_time_{n}")
            opt.add(dnn_time == total_time_expr)
            dnn_time_vars.append(dnn_time)

        # Objective
        if config.objective == Objective.MAX_THROUGHPUT:
            obj_var = Real("objective")
            opt.add(obj_var == sum(dnn_time_vars))
            opt.minimize(obj_var)
        else:
            # MIN_MAX_LATENCY
            max_var = Real("max_latency")
            for dnn_time in dnn_time_vars:
                opt.add(max_var >= dnn_time)
            opt.minimize(max_var)

        if verbose:
            LOG.info("Running Z3 solver...")

        if opt.check() != sat:
            if verbose:
                LOG.warning("No feasible schedule found")
            return best_schedule

        model = opt.model()

        # Extract assignments
        assignments: list[dict[int, ProcessorType]] = []
        for n in range(num_dnns):
            dnn_assignments: dict[int, ProcessorType] = {}
            sorted_groups = sorted(all_groups[n], key=lambda g: g.group_id)
            for i, group in enumerate(sorted_groups):
                proc_val = model[proc_vars[n][i]].as_long()
                processor = ProcessorType.GPU if proc_val == 0 else ProcessorType.DLA
                dnn_assignments[group.group_id] = processor
            assignments.append(dnn_assignments)

        # Build schedule with timing
        schedule = _build_multi_schedule(all_groups, all_costs, assignments, config)

        # Apply contention model
        schedule = estimate_contention_for_schedule(
            all_groups, all_costs, schedule, config, verbose=verbose
        )

        # Recompute objectives
        schedule.throughput_objective = compute_throughput_objective(schedule)
        schedule.max_latency_objective = compute_max_latency_objective(schedule)

        if verbose:
            LOG.info(
                f"Schedule: throughput={schedule.throughput_objective:.2f}ms, "
                f"max_latency={schedule.max_latency_objective:.2f}ms"
            )

        # Check convergence against previous iteration
        if best_schedule is not None:
            if config.objective == Objective.MAX_THROUGHPUT:
                delta = abs(best_schedule.throughput_objective - schedule.throughput_objective)
            else:
                delta = abs(best_schedule.max_latency_objective - schedule.max_latency_objective)

            if delta < config.contention_epsilon:
                if verbose:
                    LOG.info("Contention linearization converged")
                best_schedule = schedule
                break

        best_schedule = schedule

    return best_schedule


def solve_schedule_greedy(
    all_groups: list[list[LayerGroup]],
    all_costs: list[dict[int, LayerGroupCost]],
    config: HaxconnConfig,
    *,
    verbose: bool | None = None,
) -> MultiSchedule:
    """
    Find a multi-DNN schedule using a greedy algorithm.

    Algorithm:
    1. Start with all-GPU schedule
    2. For each DNN, sort DLA-compatible groups by (DLA speedup * mem throughput)
    3. Greedily move groups to DLA if the objective improves
    4. Evaluate contention after each move

    Parameters
    ----------
    all_groups : list[list[LayerGroup]]
        Groups for each DNN.
    all_costs : list[dict[int, LayerGroupCost]]
        Cost data for each DNN.
    config : HaxconnConfig
        Configuration with objective and PCCS params.
    verbose : bool | None, optional
        Whether to print verbose output.

    Returns
    -------
    MultiSchedule
        The best schedule found.

    """
    # Start with all-GPU baseline
    best = create_naive_schedule(all_groups, all_costs, config)
    best = estimate_contention_for_schedule(all_groups, all_costs, best, config, verbose=verbose)
    best.throughput_objective = compute_throughput_objective(best)
    best.max_latency_objective = compute_max_latency_objective(best)

    def _get_objective(sched: MultiSchedule) -> float:
        if config.objective == Objective.MAX_THROUGHPUT:
            return sched.throughput_objective
        return sched.max_latency_objective

    if verbose:
        LOG.info(f"Greedy baseline: {_get_objective(best):.2f}ms")

    # Collect all DLA-compatible groups with their speedup scores
    candidates: list[tuple[int, int, float]] = []  # (dnn_id, group_id, score)

    for dnn_id, (groups, costs) in enumerate(zip(all_groups, all_costs)):
        for group in groups:
            if not group.can_run_on_dla:
                continue
            cost = costs[group.group_id]
            if cost.dla_time_ms is None:
                continue

            # Score: time savings * memory intensity (favor high-bandwidth groups for DLA)
            speedup = cost.gpu_time_ms - cost.dla_time_ms
            mem_intensity = cost.gpu_mem_throughput_mbps / max(config.max_bandwidth_mbps, 1.0)
            score = speedup * (1.0 + mem_intensity)
            candidates.append((dnn_id, group.group_id, score))

    # Sort by score (best first)
    candidates.sort(key=lambda x: x[2], reverse=True)

    for dnn_id, group_id, _score in candidates:
        # Try moving this group to DLA
        test = copy.deepcopy(best)
        test.get_schedule(dnn_id).set_processor(group_id, ProcessorType.DLA)

        # Recompute timing
        groups = all_groups[dnn_id]
        costs = all_costs[dnn_id]
        compute_dnn_timing(groups, costs, test.get_schedule(dnn_id), config)

        # Apply contention
        test = estimate_contention_for_schedule(all_groups, all_costs, test, config)
        test.throughput_objective = compute_throughput_objective(test)
        test.max_latency_objective = compute_max_latency_objective(test)

        if _get_objective(test) < _get_objective(best):
            best = test
            if verbose:
                LOG.info(
                    f"Moved DNN {dnn_id} group {group_id} to DLA, "
                    f"objective: {_get_objective(best):.2f}ms"
                )

    if verbose:
        LOG.info(f"Greedy schedule: {best}")

    return best


def find_optimal_schedule(
    all_groups: list[list[LayerGroup]],
    all_costs: list[dict[int, LayerGroupCost]],
    config: HaxconnConfig | None = None,
    *,
    use_z3: bool = True,
    verbose: bool | None = None,
) -> MultiSchedule:
    """
    Find optimal multi-DNN schedule using Z3 if available, otherwise greedy.

    Parameters
    ----------
    all_groups : list[list[LayerGroup]]
        Groups for each DNN.
    all_costs : list[dict[int, LayerGroupCost]]
        Cost data for each DNN.
    config : HaxconnConfig | None, optional
        Configuration for optimization. Uses defaults if None.
    use_z3 : bool, optional
        Whether to try Z3 first. Default True.
    verbose : bool | None, optional
        Whether to print verbose output.

    Returns
    -------
    MultiSchedule
        The best schedule found.

    """
    if config is None:
        config = HaxconnConfig()

    # Try Z3 first if requested and available
    if use_z3 and Z3_AVAILABLE:
        if verbose:
            LOG.info("Using Z3 solver for multi-DNN optimization")
        schedule = solve_schedule(all_groups, all_costs, config, verbose=verbose)
        if schedule is not None:
            return schedule
        if verbose:
            LOG.info("Z3 found no solution, falling back to greedy algorithm")

    # Fall back to greedy
    if verbose:
        LOG.info("Using greedy algorithm for multi-DNN optimization")
    return solve_schedule_greedy(all_groups, all_costs, config, verbose=verbose)
