# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Z3-based solver for AxoNN optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils._log import LOG

from ._cost import (
    compute_gpu_only_costs,
    compute_total_energy,
    compute_total_time,
    estimate_transition_cost,
)
from ._types import AxoNNConfig, ProcessorType, Schedule

if TYPE_CHECKING:
    from ._types import Layer, LayerCost

try:
    from z3 import Bool, If, Not, Solver, Xor, sat

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

# Timeout for each individual Z3 satisfiability check (milliseconds)
_Z3_CHECK_TIMEOUT_MS = 30000


def _build_and_check(
    layers: list[Layer],
    costs: dict[int, LayerCost],
    config: AxoNNConfig,
    energy_target: float,
    time_budget: float,
) -> Schedule | None:
    """
    Check if a feasible schedule exists within time and energy budgets.

    Uses Z3 Solver (satisfiability) with Bool variables for efficiency.

    Parameters
    ----------
    layers : list[Layer]
        Network layers.
    costs : dict[int, LayerCost]
        Cost data keyed by layer index.
    config : AxoNNConfig
        AxoNN configuration.
    energy_target : float
        Energy constraint in millijoules.
    time_budget : float
        Time budget in milliseconds (upper bound).

    Returns
    -------
    Schedule | None
        A feasible schedule, or None if infeasible.

    """
    n_layers = len(layers)

    solver = Solver()
    solver.set("timeout", _Z3_CHECK_TIMEOUT_MS)

    # Bool decision variables: True = DLA, False = GPU
    use_dla = [Bool(f"dla_{i}") for i in range(n_layers)]

    # GPU-only layers must be False
    for layer in layers:
        if not layer.can_run_on_dla:
            solver.add(Not(use_dla[layer.index]))

    # Transition counting via Xor of adjacent variables
    # Xor(a, b) is True when a != b (i.e. a transition happens)
    trans_exprs = [Xor(use_dla[i], use_dla[i + 1]) for i in range(n_layers - 1)]

    # Limit transitions: sum of bools <= max_transitions
    # Express bool-to-int via If(b, 1, 0) and use a running sum
    if trans_exprs:
        trans_ints = [If(t, 1, 0) for t in trans_exprs]
        solver.add(sum(trans_ints) <= config.max_transitions)

    # Build time and energy expressions as linear combinations
    # time = sum(If(dla_i, dla_time_i, gpu_time_i)) + transition_times
    # energy = sum(If(dla_i, dla_energy_i, gpu_energy_i)) + transition_energies
    time_terms = []
    energy_terms = []

    for layer in layers:
        cost = costs[layer.index]
        gpu_t = cost.gpu_time_ms
        dla_t = cost.dla_time_ms if cost.dla_time_ms is not None else gpu_t
        gpu_e = cost.gpu_energy_mj
        dla_e = cost.dla_energy_mj if cost.dla_energy_mj is not None else gpu_e

        time_terms.append(If(use_dla[layer.index], dla_t, gpu_t))
        energy_terms.append(If(use_dla[layer.index], dla_e, gpu_e))

    # Add transition costs (only when trans_expr is True)
    for i in range(n_layers - 1):
        layer = layers[i]
        trans_time_ms, trans_energy_mj = estimate_transition_cost(
            layer,
            ProcessorType.GPU,
            ProcessorType.DLA,
            config,
        )
        time_terms.append(If(trans_exprs[i], trans_time_ms, 0.0))
        energy_terms.append(If(trans_exprs[i], trans_energy_mj, 0.0))

    total_time = sum(time_terms)
    total_energy = sum(energy_terms)

    # Constraints (small epsilon for floating-point vs Z3 rational precision)
    _eps = 1e-6
    solver.add(total_energy <= energy_target + _eps)
    solver.add(total_time <= time_budget + _eps)

    result = solver.check()
    if result == sat:
        model = solver.model()
        schedule = Schedule()
        for layer in layers:
            is_dla = bool(model[use_dla[layer.index]])
            processor = ProcessorType.DLA if is_dla else ProcessorType.GPU
            schedule.set_processor(layer.index, processor)
        schedule.total_time_ms = compute_total_time(layers, costs, schedule, config)
        schedule.total_energy_mj = compute_total_energy(layers, costs, schedule, config)
        return schedule

    return None


def solve_schedule(
    layers: list[Layer],
    costs: dict[int, LayerCost],
    config: AxoNNConfig | None = None,
    *,
    verbose: bool | None = None,
) -> Schedule | None:
    """
    Find optimal layer schedule using Z3 SMT solver.

    Uses binary search on execution time with Z3 satisfiability checks
    to find the minimum-time schedule within the energy constraint.

    This function formulates the AxoNN optimization problem as:
    - Objective: Minimize total execution time
    - Constraints:
        - Total energy <= energy target
        - Number of transitions <= max_transitions
        - DLA-incompatible layers must use GPU

    Parameters
    ----------
    layers : list[Layer]
        Network layers with metadata.
    costs : dict[int, LayerCost]
        Cost data keyed by layer index.
    config : AxoNNConfig | None, optional
        Configuration for optimization. Uses defaults if None.
    verbose : bool | None, optional
        Whether to print verbose output.

    Returns
    -------
    Schedule | None
        The optimal schedule, or None if no feasible solution found.

    """
    if not Z3_AVAILABLE:
        if verbose:
            LOG.warning("Z3 not available, cannot use Z3 solver")
        return None

    if config is None:
        config = AxoNNConfig()

    # Compute energy target if not specified
    energy_target = config.energy_target_mj
    if energy_target is None:
        _, gpu_energy = compute_gpu_only_costs(costs)
        energy_target = gpu_energy * config.energy_target_ratio
        if verbose:
            LOG.info(
                f"Using energy target: {energy_target:.2f}mJ "
                f"({config.energy_target_ratio * 100}% of GPU)"
            )

    n_layers = len(layers)

    if verbose:
        LOG.info(f"Solving for {n_layers} layers with energy target {energy_target:.2f}mJ")

    # Compute bounds for binary search on time
    gpu_time, _ = compute_gpu_only_costs(costs)
    # Lower bound: minimum possible time (each layer on its fastest processor)
    min_time = 0.0
    for layer in layers:
        cost = costs[layer.index]
        gpu_t = cost.gpu_time_ms
        dla_t = cost.dla_time_ms if cost.dla_time_ms is not None else gpu_t
        min_time += min(gpu_t, dla_t)
    # Upper bound: GPU-only time (always feasible if energy allows)
    max_time = gpu_time

    if verbose:
        LOG.info(f"Time bounds: [{min_time:.4f}, {max_time:.4f}] ms")
        LOG.info("Running binary search with Z3 satisfiability checks...")

    # First check: is any schedule feasible at all?
    best_schedule = _build_and_check(
        layers,
        costs,
        config,
        energy_target,
        max_time,
    )
    if best_schedule is None:
        if verbose:
            LOG.warning("No feasible schedule found within energy constraint")
            LOG.warning("Try relaxing energy_target or increasing max_transitions")
        return None

    # Binary search to minimize time
    lo, hi = min_time, max_time
    tolerance = 0.001  # 1 microsecond
    max_iterations = 30

    for iteration in range(max_iterations):
        if hi - lo < tolerance:
            break

        mid = (lo + hi) / 2.0
        candidate = _build_and_check(
            layers,
            costs,
            config,
            energy_target,
            mid,
        )

        if candidate is not None:
            best_schedule = candidate
            hi = mid
            if verbose:
                LOG.info(
                    f"  Iteration {iteration + 1}: time_budget={mid:.4f}ms -> "
                    f"feasible (time={candidate.total_time_ms:.4f}ms)"
                )
        else:
            lo = mid
            if verbose:
                LOG.info(f"  Iteration {iteration + 1}: time_budget={mid:.4f}ms -> infeasible")

    if verbose and best_schedule is not None:
        LOG.info(f"Found optimal schedule: {best_schedule}")
        LOG.info(f"  DLA layers: {best_schedule.get_dla_layers()}")
        LOG.info(f"  GPU layers: {best_schedule.get_gpu_layers()}")

    return best_schedule
