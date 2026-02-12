# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""D-HaX-CoNN: dynamic runtime schedule improvement (Section 3.5)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from trtutils._log import LOG

from ._cost import (
    compute_max_latency_objective,
    compute_throughput_objective,
    create_naive_schedule,
)
from ._solver import find_optimal_schedule
from ._types import Objective

if TYPE_CHECKING:
    from pathlib import Path

    from trtutils.builder._batcher import AbstractBatcher

    from ._types import (
        HaxconnConfig,
        LayerGroup,
        LayerGroupCost,
        MultiSchedule,
    )


def run_dynamic_haxconn(
    onnx_paths: list[Path],
    output_paths: list[Path],
    calibration_batchers: list[AbstractBatcher],
    config: HaxconnConfig,
    all_groups: list[list[LayerGroup]],
    all_costs: list[dict[int, LayerGroupCost]],
    all_layers_count: list[int],
    workspace: float = 4.0,
    timing_cache: Path | str | None = None,
    calibration_cache: Path | str | None = None,
    shapes: list[list[tuple[str, tuple[int, ...]]] | None] | None = None,
    optimization_level: int = 3,
    *,
    cache: bool | None = None,
    verbose: bool | None = None,
) -> tuple[MultiSchedule, list[Path]]:
    """
    D-HaX-CoNN: progressively improve schedule at runtime.

    Algorithm:
    1. Create naive all-GPU schedule and build initial engines
    2. Run solver for improved schedule
    3. If improved, rebuild engines with new assignments
    4. Repeat until budget or max rounds exhausted, or improvement < threshold

    Parameters
    ----------
    onnx_paths : list[Path]
        ONNX model paths for each DNN.
    output_paths : list[Path]
        Output engine paths for each DNN.
    calibration_batchers : list[AbstractBatcher]
        One calibration batcher per DNN.
    config : HaxconnConfig
        HaX-CoNN configuration.
    all_groups : list[list[LayerGroup]]
        Groups for each DNN.
    all_costs : list[dict[int, LayerGroupCost]]
        Costs for each DNN.
    all_layers_count : list[int]
        Number of layers per DNN.
    workspace : float, optional
        Workspace size in GB.
    timing_cache : Path | str | None, optional
        Path to timing cache.
    calibration_cache : Path | str | None, optional
        Path to calibration cache.
    shapes : list[list[tuple[str, tuple[int, ...]]] | None] | None, optional
        Per-DNN input shapes.
    optimization_level : int, optional
        TensorRT optimization level.
    cache : bool | None, optional
        Cache the engines.
    verbose : bool | None, optional
        Whether to print verbose output.

    Returns
    -------
    tuple[MultiSchedule, list[Path]]
        The best schedule found and paths to the built engines.

    """
    # Lazy import to avoid circular dependency
    from ._build import _build_engines_from_schedule  # noqa: PLC0415

    def _get_objective(sched: MultiSchedule) -> float:
        if config.objective == Objective.MAX_THROUGHPUT:
            return sched.throughput_objective
        return sched.max_latency_objective

    start_time = time.monotonic()

    # Step 1: Start with naive all-GPU schedule
    current_schedule = create_naive_schedule(all_groups, all_costs, config)

    if verbose:
        LOG.info(f"D-HaX-CoNN: initial all-GPU objective = {_get_objective(current_schedule):.2f}ms")

    # Build initial engines
    engine_paths = _build_engines_from_schedule(
        onnx_paths=onnx_paths,
        output_paths=output_paths,
        calibration_batchers=calibration_batchers,
        multi_schedule=current_schedule,
        all_groups=all_groups,
        all_layers_count=all_layers_count,
        config=config,
        workspace=workspace,
        timing_cache=timing_cache,
        calibration_cache=calibration_cache,
        shapes=shapes,
        optimization_level=optimization_level,
        cache=cache,
        verbose=verbose,
    )

    # Step 2-4: Iterative improvement
    for round_num in range(config.dynamic_max_rounds):
        elapsed = time.monotonic() - start_time
        if elapsed >= config.dynamic_budget_s:
            if verbose:
                LOG.info(f"D-HaX-CoNN: time budget exhausted after {elapsed:.1f}s")
            break

        if verbose:
            LOG.info(f"D-HaX-CoNN round {round_num + 1}: solving for improved schedule...")

        # Find improved schedule
        new_schedule = find_optimal_schedule(
            all_groups=all_groups,
            all_costs=all_costs,
            config=config,
            verbose=verbose,
        )

        new_obj = _get_objective(new_schedule)
        current_obj = _get_objective(current_schedule)

        if current_obj <= 0:
            break

        improvement = (current_obj - new_obj) / current_obj

        if verbose:
            LOG.info(
                f"D-HaX-CoNN round {round_num + 1}: "
                f"new={new_obj:.2f}ms, current={current_obj:.2f}ms, "
                f"improvement={improvement * 100:.1f}%"
            )

        if improvement < config.dynamic_improvement_threshold:
            if verbose:
                LOG.info("D-HaX-CoNN: improvement below threshold, stopping")
            break

        # Rebuild engines with improved schedule
        current_schedule = new_schedule
        engine_paths = _build_engines_from_schedule(
            onnx_paths=onnx_paths,
            output_paths=output_paths,
            calibration_batchers=calibration_batchers,
            multi_schedule=current_schedule,
            all_groups=all_groups,
            all_layers_count=all_layers_count,
            config=config,
            workspace=workspace,
            timing_cache=timing_cache,
            calibration_cache=calibration_cache,
            shapes=shapes,
            optimization_level=optimization_level,
            cache=cache,
            verbose=verbose,
        )

    elapsed = time.monotonic() - start_time

    # Recompute objectives
    current_schedule.throughput_objective = compute_throughput_objective(current_schedule)
    current_schedule.max_latency_objective = compute_max_latency_objective(current_schedule)

    if verbose:
        LOG.info(f"D-HaX-CoNN complete in {elapsed:.1f}s: {current_schedule}")

    return current_schedule, engine_paths
