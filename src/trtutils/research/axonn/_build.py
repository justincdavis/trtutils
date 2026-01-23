# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine building for AxoNN optimization."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):
    import tensorrt as trt

from trtutils._log import LOG
from trtutils.builder._build import build_engine

from ._cost import compute_gpu_only_costs, compute_total_energy, compute_total_time
from ._profile import extract_layer_info, profile_for_axonn
from ._solver import find_optimal_schedule
from ._types import AxoNNConfig, ProcessorType, Schedule

if TYPE_CHECKING:
    from trtutils.builder._batcher import AbstractBatcher


def schedule_to_layer_assignments(
    schedule: Schedule,
    num_layers: int,
) -> tuple[list[tuple[int, trt.DataType | None]], list[tuple[int, trt.DeviceType | None]]]:
    """
    Convert a Schedule to layer precision and device assignments.

    Parameters
    ----------
    schedule : Schedule
        The schedule with processor assignments.
    num_layers : int
        Total number of layers in the network.

    Returns
    -------
    tuple[list[tuple[int, trt.DataType | None]], list[tuple[int, trt.DeviceType | None]]]
        Layer precision assignments and layer device assignments.

    """
    layer_precision: list[tuple[int, trt.DataType | None]] = []
    layer_device: list[tuple[int, trt.DeviceType | None]] = []

    # Layers that should not have explicit precision set
    # These are handled automatically by TensorRT
    skip_precision_types = ["CONSTANT", "SHUFFLE"]

    for layer_idx in range(num_layers):
        if layer_idx in schedule.assignments:
            processor = schedule.assignments[layer_idx]

            if processor == ProcessorType.DLA:
                # DLA requires INT8 precision
                layer_precision.append((layer_idx, trt.DataType.INT8))
                layer_device.append((layer_idx, trt.DeviceType.DLA))
            else:
                # GPU uses FP16
                layer_precision.append((layer_idx, trt.DataType.HALF))
                layer_device.append((layer_idx, trt.DeviceType.GPU))
        else:
            # Default to GPU/FP16 if not in schedule
            layer_precision.append((layer_idx, trt.DataType.HALF))
            layer_device.append((layer_idx, trt.DeviceType.GPU))

    return layer_precision, layer_device


def build_axonn_engine(
    onnx: Path | str,
    output: Path | str,
    calibration_batcher: AbstractBatcher,
    schedule: Schedule | None = None,
    config: AxoNNConfig | None = None,
    workspace: float = 4.0,
    timing_cache: Path | str | None = None,
    calibration_cache: Path | str | None = None,
    shapes: list[tuple[str, tuple[int, ...]]] | None = None,
    optimization_level: int = 3,
    *,
    direct_io: bool = False,
    prefer_precision_constraints: bool = False,
    reject_empty_algorithms: bool = False,
    ignore_timing_mismatch: bool = False,
    cache: bool | None = None,
    verbose: bool | None = None,
) -> Schedule:
    """
    Build an optimized TensorRT engine using AxoNN algorithm.

    If a schedule is not provided, this function will:
    1. Profile the model on GPU and DLA
    2. Use the AxoNN solver to find optimal layer assignments
    3. Build the engine with the optimal schedule

    Parameters
    ----------
    onnx : Path | str
        Path to the ONNX model.
    output : Path | str
        Path to save the TensorRT engine.
    calibration_batcher : AbstractBatcher
        Data batcher for INT8 calibration.
    schedule : Schedule | None, optional
        Pre-computed schedule. If None, will be computed.
    config : AxoNNConfig | None, optional
        AxoNN configuration. Uses defaults if None.
    workspace : float, optional
        Workspace size in GB. Default 4.0.
    timing_cache : Path | str | None, optional
        Path to timing cache.
    calibration_cache : Path | str | None, optional
        Path to calibration cache.
    shapes : list[tuple[str, tuple[int, ...]]] | None, optional
        Fixed input shapes.
    optimization_level : int, optional
        TensorRT optimization level (0-5). Default 3.
    direct_io : bool, optional
        Use direct I/O. Default False.
    prefer_precision_constraints : bool, optional
        Prefer precision constraints. Default False.
    reject_empty_algorithms : bool, optional
        Reject empty algorithms. Default False.
    ignore_timing_mismatch : bool, optional
        Ignore timing cache mismatches. Default False.
    cache : bool | None, optional
        Cache the engine. Default None.
    verbose : bool | None, optional
        Verbose output. Default None.

    Returns
    -------
    Schedule
        The schedule used to build the engine.

    """
    if config is None:
        config = AxoNNConfig()

    onnx_path = Path(onnx)
    output_path = Path(output)

    if verbose:
        LOG.info(f"Building AxoNN-optimized engine from {onnx_path}")

    # Get layer information
    layers = extract_layer_info(onnx_path, verbose=verbose)
    num_layers = len(layers)

    # If no schedule provided, profile and optimize
    if schedule is None:
        if verbose:
            LOG.info("No schedule provided, profiling model...")

        layers, costs = profile_for_axonn(
            onnx=onnx_path,
            calibration_batcher=calibration_batcher,
            config=config,
            workspace=workspace,
            timing_cache=timing_cache,
            calibration_cache=calibration_cache,
            verbose=verbose,
        )

        if verbose:
            gpu_time, gpu_energy = compute_gpu_only_costs(costs)
            LOG.info(f"GPU-only baseline: {gpu_time:.2f}ms, {gpu_energy:.2f}mJ")

        schedule = find_optimal_schedule(
            layers=layers,
            costs=costs,
            config=config,
            verbose=verbose,
        )

        if verbose:
            LOG.info(f"Optimized schedule: {schedule}")

    # Convert schedule to layer assignments
    layer_precision, layer_device = schedule_to_layer_assignments(schedule, num_layers)

    # Determine if we need DLA
    has_dla = any(proc == ProcessorType.DLA for proc in schedule.assignments.values())

    if verbose:
        dla_count = len(schedule.get_dla_layers())
        gpu_count = len(schedule.get_gpu_layers())
        LOG.info(f"Building engine with {gpu_count} GPU layers, {dla_count} DLA layers")

    # Build the engine
    build_engine(
        onnx=onnx_path,
        output=output_path,
        default_device=trt.DeviceType.DLA if has_dla else trt.DeviceType.GPU,
        timing_cache=timing_cache,
        workspace=workspace,
        calibration_cache=calibration_cache,
        data_batcher=calibration_batcher,
        layer_precision=layer_precision,
        layer_device=layer_device,
        dla_core=config.dla_core if has_dla else None,
        shapes=shapes,
        optimization_level=optimization_level,
        profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
        gpu_fallback=True,
        direct_io=direct_io,
        prefer_precision_constraints=prefer_precision_constraints,
        reject_empty_algorithms=reject_empty_algorithms,
        ignore_timing_mismatch=ignore_timing_mismatch,
        fp16=True,
        int8=True,
        cache=cache,
        verbose=verbose,
    )

    if verbose:
        LOG.info(f"Engine saved to {output_path}")
        LOG.info(f"Expected performance: {schedule.total_time_ms:.2f}ms, {schedule.total_energy_mj:.2f}mJ")

    return schedule


def optimize_and_build(
    onnx: Path | str,
    output: Path | str,
    calibration_batcher: AbstractBatcher,
    energy_target_mj: float | None = None,
    energy_target_ratio: float = 0.8,
    max_transitions: int = 3,
    dla_core: int = 0,
    workspace: float = 4.0,
    timing_cache: Path | str | None = None,
    calibration_cache: Path | str | None = None,
    shapes: list[tuple[str, tuple[int, ...]]] | None = None,
    optimization_level: int = 3,
    profile_iterations: int = 1000,
    warmup_iterations: int = 50,
    *,
    direct_io: bool = False,
    prefer_precision_constraints: bool = False,
    reject_empty_algorithms: bool = False,
    ignore_timing_mismatch: bool = False,
    cache: bool | None = None,
    verbose: bool | None = None,
) -> Schedule:
    """
    One-shot function to profile, optimize, and build an AxoNN engine.

    This is a convenience function that combines profiling, optimization,
    and building into a single call.

    Parameters
    ----------
    onnx : Path | str
        Path to the ONNX model.
    output : Path | str
        Path to save the TensorRT engine.
    calibration_batcher : AbstractBatcher
        Data batcher for INT8 calibration.
    energy_target_mj : float | None, optional
        Target energy in millijoules. If None, uses ratio of GPU energy.
    energy_target_ratio : float, optional
        Ratio of GPU energy to use as target. Default 0.8.
    max_transitions : int, optional
        Maximum GPU<->DLA transitions. Default 3.
    dla_core : int, optional
        DLA core to use. Default 0.
    workspace : float, optional
        Workspace size in GB. Default 4.0.
    timing_cache : Path | str | None, optional
        Path to timing cache.
    calibration_cache : Path | str | None, optional
        Path to calibration cache.
    shapes : list[tuple[str, tuple[int, ...]]] | None, optional
        Fixed input shapes.
    optimization_level : int, optional
        TensorRT optimization level. Default 3.
    profile_iterations : int, optional
        Number of profiling iterations. Default 1000.
    warmup_iterations : int, optional
        Number of warmup iterations. Default 50.
    direct_io : bool, optional
        Use direct I/O. Default False.
    prefer_precision_constraints : bool, optional
        Prefer precision constraints. Default False.
    reject_empty_algorithms : bool, optional
        Reject empty algorithms. Default False.
    ignore_timing_mismatch : bool, optional
        Ignore timing cache mismatches. Default False.
    cache : bool | None, optional
        Cache the engine. Default None.
    verbose : bool | None, optional
        Verbose output. Default None.

    Returns
    -------
    Schedule
        The schedule used to build the engine.

    """
    config = AxoNNConfig(
        energy_target_mj=energy_target_mj,
        energy_target_ratio=energy_target_ratio,
        max_transitions=max_transitions,
        dla_core=dla_core,
        profile_iterations=profile_iterations,
        warmup_iterations=warmup_iterations,
    )

    return build_axonn_engine(
        onnx=onnx,
        output=output,
        calibration_batcher=calibration_batcher,
        schedule=None,
        config=config,
        workspace=workspace,
        timing_cache=timing_cache,
        calibration_cache=calibration_cache,
        shapes=shapes,
        optimization_level=optimization_level,
        direct_io=direct_io,
        prefer_precision_constraints=prefer_precision_constraints,
        reject_empty_algorithms=reject_empty_algorithms,
        ignore_timing_mismatch=ignore_timing_mismatch,
        cache=cache,
        verbose=verbose,
    )
