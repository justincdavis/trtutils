# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
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
            "or pip install trtutils[axonn]"
        )
        raise ImportError(err_msg)


def solve_schedule(
    layers: list[Layer],
    costs: dict[int, LayerCost],
    config: AxoNNConfig | None = None,
    *,
    verbose: bool | None = None,
) -> Schedule | None:
    """
    Find optimal layer schedule using Z3 SMT solver.

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

    Raises
    ------
    ImportError
        If Z3 is not installed.

    """
    _check_z3_available()

    if config is None:
        config = AxoNNConfig()

    # Compute energy target if not specified
    energy_target = config.energy_target_mj
    if energy_target is None:
        _, gpu_energy = compute_gpu_only_costs(costs)
        energy_target = gpu_energy * config.energy_target_ratio
        if verbose:
            LOG.info(
                f"Using energy target: {energy_target:.2f}mJ ({config.energy_target_ratio * 100}% of GPU)"
            )

    n_layers = len(layers)

    if verbose:
        LOG.info(f"Solving for {n_layers} layers with energy target {energy_target:.2f}mJ")

    # Create Z3 optimizer
    opt = Optimize()

    # Decision variables: processor for each layer (0 = GPU, 1 = DLA)
    proc_vars = [Int(f"proc_{i}") for i in range(n_layers)]

    # Constrain to valid processor values (0 or 1)
    for _i, var in enumerate(proc_vars):
        opt.add(And(var >= 0, var <= 1))

    # Layer compatibility constraints: GPU-only layers must use GPU
    for layer in layers:
        if not layer.can_run_on_dla:
            opt.add(proc_vars[layer.index] == 0)

    # Transition indicators
    trans_vars = []
    for i in range(n_layers - 1):
        trans_var = Int(f"trans_{i}")
        # trans_var = 1 if processor changes, 0 otherwise
        opt.add(trans_var == If(proc_vars[i] != proc_vars[i + 1], 1, 0))
        trans_vars.append(trans_var)

    # Limit number of transitions
    if trans_vars:
        opt.add(sum(trans_vars) <= config.max_transitions)

    # Build time expression
    # For each layer: time = if proc==0 then gpu_time else dla_time
    time_expr = Real("total_time")
    layer_times = []
    for layer in layers:
        cost = costs[layer.index]
        gpu_time = cost.gpu_time_ms
        dla_time = cost.dla_time_ms if cost.dla_time_ms is not None else gpu_time

        layer_time = If(
            proc_vars[layer.index] == 0,
            gpu_time,
            dla_time,
        )
        layer_times.append(layer_time)

    # Add transition times
    transition_times = []
    for i in range(n_layers - 1):
        layer = layers[i]
        # Estimate transition cost for this layer
        trans_time_ms, _ = estimate_transition_cost(
            layer,
            ProcessorType.GPU,  # Doesn't matter for cost estimation
            ProcessorType.DLA,
            config,
        )
        # Transition time added only when trans_var == 1
        trans_time = If(trans_vars[i] == 1, trans_time_ms, 0.0)
        transition_times.append(trans_time)

    total_time_expr = (
        sum(layer_times) + sum(transition_times) if transition_times else sum(layer_times)
    )
    opt.add(time_expr == total_time_expr)

    # Build energy expression
    energy_expr = Real("total_energy")
    layer_energies = []
    for layer in layers:
        cost = costs[layer.index]
        gpu_energy = cost.gpu_energy_mj
        dla_energy = cost.dla_energy_mj if cost.dla_energy_mj is not None else gpu_energy

        layer_energy = If(
            proc_vars[layer.index] == 0,
            gpu_energy,
            dla_energy,
        )
        layer_energies.append(layer_energy)

    # Add transition energies
    transition_energies = []
    for i in range(n_layers - 1):
        layer = layers[i]
        _, trans_energy_mj = estimate_transition_cost(
            layer,
            ProcessorType.GPU,
            ProcessorType.DLA,
            config,
        )
        trans_energy = If(trans_vars[i] == 1, trans_energy_mj, 0.0)
        transition_energies.append(trans_energy)

    total_energy_expr = (
        sum(layer_energies) + sum(transition_energies)
        if transition_energies
        else sum(layer_energies)
    )
    opt.add(energy_expr == total_energy_expr)

    # Energy constraint
    opt.add(energy_expr <= energy_target)

    # Objective: minimize time
    opt.minimize(time_expr)

    if verbose:
        LOG.info("Running Z3 solver...")

    # Solve
    if opt.check() == sat:
        model = opt.model()

        # Extract schedule
        schedule = Schedule()
        for layer in layers:
            proc_val = model[proc_vars[layer.index]].as_long()
            processor = ProcessorType.GPU if proc_val == 0 else ProcessorType.DLA
            schedule.set_processor(layer.index, processor)

        # Compute actual costs
        schedule.total_time_ms = compute_total_time(layers, costs, schedule, config)
        schedule.total_energy_mj = compute_total_energy(layers, costs, schedule, config)

        if verbose:
            LOG.info(f"Found optimal schedule: {schedule}")
            LOG.info(f"  DLA layers: {schedule.get_dla_layers()}")
            LOG.info(f"  GPU layers: {schedule.get_gpu_layers()}")

        return schedule

    if verbose:
        LOG.warning("No feasible schedule found within constraints")
        LOG.warning("Try relaxing energy_target or increasing max_transitions")

    return None


def solve_schedule_greedy(
    layers: list[Layer],
    costs: dict[int, LayerCost],
    config: AxoNNConfig | None = None,
    *,
    verbose: bool | None = None,
) -> Schedule:
    """
    Find a schedule using a greedy algorithm (fallback when Z3 unavailable).

    This algorithm:
    1. Starts with all layers on GPU
    2. Greedily moves DLA-compatible layer chunks to DLA
    3. Stops when energy target is met or no improvement possible

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
    Schedule
        The best schedule found.

    """
    if config is None:
        config = AxoNNConfig()

    # Compute energy target if not specified
    energy_target = config.energy_target_mj
    if energy_target is None:
        _, gpu_energy = compute_gpu_only_costs(costs)
        energy_target = gpu_energy * config.energy_target_ratio

    # Start with all GPU schedule
    schedule = Schedule()
    for layer in layers:
        schedule.set_processor(layer.index, ProcessorType.GPU)

    # Find contiguous chunks of DLA-compatible layers
    chunks: list[list[int]] = []
    current_chunk: list[int] = []

    for layer in layers:
        if layer.can_run_on_dla:
            current_chunk.append(layer.index)
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
    if current_chunk:
        chunks.append(current_chunk)

    if verbose:
        LOG.info(f"Found {len(chunks)} DLA-compatible chunks")

    # Sort chunks by energy savings (biggest first)
    def chunk_energy_savings(chunk: list[int]) -> float:
        savings = 0.0
        for idx in chunk:
            cost = costs[idx]
            if cost.dla_energy_mj is not None:
                savings += cost.gpu_energy_mj - cost.dla_energy_mj
        return savings

    chunks.sort(key=chunk_energy_savings, reverse=True)

    # Greedily assign chunks to DLA
    transitions_used = 0
    for chunk in chunks:
        if transitions_used + 2 > config.max_transitions:
            # Adding this chunk would exceed transition limit
            # (each chunk adds at most 2 transitions: entry and exit)
            break

        # Try assigning this chunk to DLA
        test_schedule = Schedule(assignments=dict(schedule.assignments))
        for idx in chunk:
            test_schedule.set_processor(idx, ProcessorType.DLA)

        test_energy = compute_total_energy(layers, costs, test_schedule, config)

        if test_energy <= energy_target:
            # Accept this chunk
            schedule = test_schedule
            transitions_used = schedule.num_transitions
            if verbose:
                LOG.info(f"Assigned chunk {chunk} to DLA, energy: {test_energy:.2f}mJ")

    # Compute final costs
    schedule.total_time_ms = compute_total_time(layers, costs, schedule, config)
    schedule.total_energy_mj = compute_total_energy(layers, costs, schedule, config)

    if verbose:
        LOG.info(f"Greedy schedule: {schedule}")

    return schedule


def find_optimal_schedule(
    layers: list[Layer],
    costs: dict[int, LayerCost],
    config: AxoNNConfig | None = None,
    *,
    use_z3: bool = True,
    verbose: bool | None = None,
) -> Schedule:
    """
    Find optimal schedule using Z3 if available, otherwise greedy algorithm.

    Parameters
    ----------
    layers : list[Layer]
        Network layers with metadata.
    costs : dict[int, LayerCost]
        Cost data keyed by layer index.
    config : AxoNNConfig | None, optional
        Configuration for optimization. Uses defaults if None.
    use_z3 : bool, optional
        Whether to try Z3 first. Default True.
    verbose : bool | None, optional
        Whether to print verbose output.

    Returns
    -------
    Schedule
        The best schedule found.

    """
    if config is None:
        config = AxoNNConfig()

    # Try Z3 first if requested and available
    if use_z3 and Z3_AVAILABLE:
        if verbose:
            LOG.info("Using Z3 solver for optimization")
        schedule = solve_schedule(layers, costs, config, verbose=verbose)
        if schedule is not None:
            return schedule
        if verbose:
            LOG.info("Z3 found no solution, falling back to greedy algorithm")

    # Fall back to greedy
    if verbose:
        LOG.info("Using greedy algorithm for optimization")
    return solve_schedule_greedy(layers, costs, config, verbose=verbose)
