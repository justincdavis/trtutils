# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine building for AxoNN optimization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import tensorrt as trt

from trtutils._log import LOG
from trtutils.builder._build import build_engine as trt_build_engine

from ._cost import compute_gpu_only_costs
from ._profile import profile_for_axonn
from ._solver import solve_schedule
from ._types import AxoNNConfig, ProcessorType, Schedule

if TYPE_CHECKING:
    from trtutils.builder._batcher import AbstractBatcher


def _schedule_to_layer_assignments(
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

    for layer_idx in range(num_layers):
        if layer_idx in schedule.assignments:
            processor = schedule.assignments[layer_idx]

            if processor == ProcessorType.DLA:
                # DLA requires INT8 precision
                layer_precision.append((layer_idx, trt.DataType.INT8))
                layer_device.append((layer_idx, trt.DeviceType.DLA))
            else:
                # GPU - don't lock precision, let TensorRT optimize
                layer_precision.append((layer_idx, None))
                layer_device.append((layer_idx, trt.DeviceType.GPU))
        else:
            # Default to GPU if not in schedule, no precision lock
            layer_precision.append((layer_idx, None))
            layer_device.append((layer_idx, trt.DeviceType.GPU))

    return layer_precision, layer_device


def build_engine(
    onnx: Path | str,
    output: Path | str,
    calibration_batcher: AbstractBatcher,
    *,
    energy_target: float | None = None,
    energy_ratio: float = 0.8,
    max_transitions: int = 1,
    dla_core: int = 0,
    profile_iterations: int = 1000,
    warmup_iterations: int = 50,
    workspace: float = 4.0,
    timing_cache: Path | str | None = None,
    calibration_cache: Path | str | None = None,
    shapes: list[tuple[str, tuple[int, ...]]] | None = None,
    optimization_level: int = 3,
    direct_io: bool = False,
    prefer_precision_constraints: bool = False,
    reject_empty_algorithms: bool = False,
    ignore_timing_mismatch: bool = False,
    cache: bool | None = None,
    verbose: bool | None = None,
) -> tuple[float, float, int, int, int]:
    """
    Build an energy-optimized TensorRT engine using the AxoNN algorithm.

    AxoNN finds optimal layer-to-accelerator mappings that minimize execution
    time while staying under an Energy Consumption Target (ECT). It profiles
    each layer on GPU and DLA, then uses constraint optimization to find
    the best schedule.

    Reference: AxoNN: Energy-Aware Execution of Neural Network Inference on
    Multi-Accelerator Heterogeneous SoCs (DAC 2022)
    https://doi.org/10.1145/3489517.3530572

    Parameters
    ----------
    onnx : Path | str
        Path to the ONNX model.
    output : Path | str
        Path to save the TensorRT engine.
    calibration_batcher : AbstractBatcher
        Data batcher for INT8 calibration (required for DLA).
    energy_target : float | None, optional
        Explicit Energy Consumption Target (ECT) in millijoules per inference.
        When set, the solver constrains the schedule so total energy per
        inference does not exceed this value. If None (default), the ECT is
        derived automatically from ``energy_ratio``.
    energy_ratio : float, optional
        Fraction of the GPU-only baseline energy to use as the ECT. Default
        0.8. Only used when ``energy_target`` is None. The model is first
        profiled running entirely on GPU to measure baseline energy. The ECT
        is then set to ``energy_ratio * gpu_baseline_energy``. For example,
        ``energy_ratio=0.8`` means "find a schedule that uses at most 80% of
        the energy that GPU-only execution consumes." Lower values impose a
        tighter energy budget, pushing more layers to DLA (potentially slower
        but more energy-efficient). Higher values relax the constraint,
        keeping more layers on GPU (faster but higher energy).
    max_transitions : int, optional
        Maximum number of GPU<->DLA device transitions allowed in the
        schedule. Each transition incurs overhead from memory transfers
        between accelerators. Default 1.
    dla_core : int, optional
        DLA core to use (0 or 1). Default 0.
    profile_iterations : int, optional
        Number of inference iterations for profiling each layer/engine
        configuration. More iterations yield more stable timing and power
        measurements at the cost of longer profiling time. Default 1000.
    warmup_iterations : int, optional
        Number of warmup iterations run before profiling begins, to ensure
        GPU/DLA clocks are stable. Default 50.
    workspace : float, optional
        TensorRT workspace size in GB. Default 4.0.
    timing_cache : Path | str | None, optional
        Path to timing cache file.
    calibration_cache : Path | str | None, optional
        Path to calibration cache file.
    shapes : list[tuple[str, tuple[int, ...]]] | None, optional
        Fixed input shapes as list of (name, dims) tuples.
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
        Enable verbose output. Default None.

    Returns
    -------
    tuple[float, float, int, int, int]
        A tuple of (total_time_ms, total_energy_mj, num_transitions,
        num_gpu_layers, num_dla_layers).

    Examples
    --------
    >>> from trtutils.builder import ImageBatcher
    >>> from trtutils.research.axonn import build_engine
    >>>
    >>> batcher = ImageBatcher(
    ...     image_dir="calibration_images/",
    ...     shape=(640, 640, 3),
    ...     dtype=np.float32,
    ... )
    >>>
    >>> time_ms, energy_mj, transitions, gpu_layers, dla_layers = build_engine(
    ...     onnx="model.onnx",
    ...     output="model.engine",
    ...     calibration_batcher=batcher,
    ...     energy_ratio=0.8,  # ECT = 80% of GPU-only baseline energy
    ...     max_transitions=1,
    ...     verbose=True,
    ... )
    >>> print(f"Time: {time_ms:.2f}ms, Energy: {energy_mj:.2f}mJ")

    """
    # Create internal config
    config = AxoNNConfig(
        energy_target_mj=energy_target,
        energy_target_ratio=energy_ratio,
        max_transitions=max_transitions,
        dla_core=dla_core,
        profile_iterations=profile_iterations,
        warmup_iterations=warmup_iterations,
    )

    onnx_path = Path(onnx)
    output_path = Path(output)

    if verbose:
        LOG.info(f"Building AxoNN-optimized engine from {onnx_path}")
        LOG.info(f"Energy target: {energy_target}mJ (ratio={energy_ratio})")
        LOG.info(f"Max transitions: {max_transitions}, DLA core: {dla_core}")

    if verbose:
        LOG.info("Profiling model on GPU and DLA...")

    # Profile the model (also extracts layer information)
    layers, costs = profile_for_axonn(
        onnx=onnx_path,
        calibration_batcher=calibration_batcher,
        config=config,
        workspace=workspace,
        timing_cache=timing_cache,
        calibration_cache=calibration_cache,
        verbose=verbose,
    )
    num_layers = len(layers)

    # Get GPU baseline
    gpu_time, gpu_energy = compute_gpu_only_costs(costs)

    if verbose:
        LOG.info(f"GPU-only baseline: {gpu_time:.2f}ms, {gpu_energy:.2f}mJ")

    # Find optimal schedule via Z3 solver
    schedule = solve_schedule(
        layers=layers,
        costs=costs,
        config=config,
        verbose=verbose,
    )

    if schedule is None:
        # Z3 found no feasible solution — fall back to GPU-only schedule
        if verbose:
            LOG.warning("No feasible AxoNN schedule found, using GPU-only schedule")
        schedule = Schedule()
        for layer in layers:
            schedule.set_processor(layer.index, ProcessorType.GPU)
        schedule.total_time_ms = gpu_time
        schedule.total_energy_mj = gpu_energy

    if verbose:
        LOG.info(f"Optimal schedule: {schedule}")

    # Convert schedule to layer assignments
    layer_precision, layer_device = _schedule_to_layer_assignments(schedule, num_layers)

    # Determine if we need DLA
    has_dla = any(proc == ProcessorType.DLA for proc in schedule.assignments.values())

    dla_count = len(schedule.get_dla_layers())
    gpu_count = len(schedule.get_gpu_layers())

    if verbose:
        LOG.info(f"Building engine with {gpu_count} GPU layers, {dla_count} DLA layers")

    # Build the engine
    trt_build_engine(
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
        LOG.info(
            f"Expected performance: {schedule.total_time_ms:.2f}ms, {schedule.total_energy_mj:.2f}mJ"
        )

    return (
        schedule.total_time_ms,
        schedule.total_energy_mj,
        schedule.num_transitions,
        gpu_count,
        dla_count,
    )
