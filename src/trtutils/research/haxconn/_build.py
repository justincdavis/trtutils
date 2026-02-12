# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine building orchestration for HaX-CoNN optimization."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):
    import tensorrt as trt

from trtutils._log import LOG
from trtutils.builder._build import build_engine as trt_build_engine

from ._executor import HaxconnExecutor
from ._profile import profile_for_haxconn
from ._solver import find_optimal_schedule
from ._types import HaxconnConfig, MultiSchedule, Objective, ProcessorType

if TYPE_CHECKING:
    from trtutils.builder._batcher import AbstractBatcher

    from ._types import DNNSchedule, LayerGroup


def _schedule_to_layer_assignments(
    schedule: DNNSchedule,
    groups: list[LayerGroup],
    num_layers: int,
) -> tuple[list[tuple[int, trt.DataType | None]], list[tuple[int, trt.DeviceType | None]]]:
    """
    Convert a DNNSchedule to layer precision and device assignments.

    Parameters
    ----------
    schedule : DNNSchedule
        The group-to-processor schedule.
    groups : list[LayerGroup]
        Layer groups for this DNN.
    num_layers : int
        Total number of layers in the network.

    Returns
    -------
    tuple[list[tuple[int, trt.DataType | None]], list[tuple[int, trt.DeviceType | None]]]
        Layer precision assignments and layer device assignments.

    """
    # Build group_id -> processor mapping
    group_proc: dict[int, ProcessorType] = {}
    for group in groups:
        if group.group_id in schedule.assignments:
            group_proc[group.group_id] = schedule.assignments[group.group_id]

    # Build layer_index -> group_id mapping
    layer_to_group: dict[int, int] = {}
    for group in groups:
        for idx in group.layer_indices:
            layer_to_group[idx] = group.group_id

    layer_precision: list[tuple[int, trt.DataType | None]] = []
    layer_device: list[tuple[int, trt.DeviceType | None]] = []

    for layer_idx in range(num_layers):
        gid = layer_to_group.get(layer_idx)
        proc = group_proc.get(gid) if gid is not None else None

        if proc == ProcessorType.DLA:
            layer_precision.append((layer_idx, trt.DataType.INT8))
            layer_device.append((layer_idx, trt.DeviceType.DLA))
        else:
            layer_precision.append((layer_idx, None))
            layer_device.append((layer_idx, trt.DeviceType.GPU))

    return layer_precision, layer_device


def _build_engines_from_schedule(
    onnx_paths: list[Path],
    output_paths: list[Path],
    calibration_batchers: list[AbstractBatcher],
    multi_schedule: MultiSchedule,
    all_groups: list[list[LayerGroup]],
    all_layers_count: list[int],
    config: HaxconnConfig,
    workspace: float = 4.0,
    timing_cache: Path | str | None = None,
    calibration_cache: Path | str | None = None,
    shapes: list[list[tuple[str, tuple[int, ...]]] | None] | None = None,
    optimization_level: int = 3,
    *,
    cache: bool | None = None,
    verbose: bool | None = None,
) -> list[Path]:
    """
    Build TensorRT engines from a multi-DNN schedule.

    Parameters
    ----------
    onnx_paths : list[Path]
        ONNX model paths for each DNN.
    output_paths : list[Path]
        Output engine paths for each DNN.
    calibration_batchers : list[AbstractBatcher]
        One calibration batcher per DNN.
    multi_schedule : MultiSchedule
        The multi-DNN schedule with group-to-processor assignments.
    all_groups : list[list[LayerGroup]]
        Layer groups for each DNN.
    all_layers_count : list[int]
        Number of layers per DNN.
    config : HaxconnConfig
        HaX-CoNN configuration.
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
    list[Path]
        Paths to the built engine files.

    """
    built_paths: list[Path] = []

    for dnn_id in range(len(onnx_paths)):
        schedule = multi_schedule.get_schedule(dnn_id)
        groups = all_groups[dnn_id]
        num_layers = all_layers_count[dnn_id]

        layer_precision, layer_device = _schedule_to_layer_assignments(schedule, groups, num_layers)

        has_dla = any(proc == ProcessorType.DLA for proc in schedule.assignments.values())

        dnn_shapes = None
        if shapes is not None and dnn_id < len(shapes):
            dnn_shapes = shapes[dnn_id]

        if verbose:
            dla_count = len(schedule.get_dla_groups())
            gpu_count = len(schedule.get_gpu_groups())
            LOG.info(f"Building DNN {dnn_id} engine: {gpu_count} GPU groups, {dla_count} DLA groups")

        trt_build_engine(
            onnx=onnx_paths[dnn_id],
            output=output_paths[dnn_id],
            default_device=trt.DeviceType.DLA if has_dla else trt.DeviceType.GPU,
            timing_cache=timing_cache,
            workspace=workspace,
            calibration_cache=calibration_cache,
            data_batcher=calibration_batchers[dnn_id],
            layer_precision=layer_precision,
            layer_device=layer_device,
            dla_core=config.dla_core if has_dla else None,
            shapes=dnn_shapes,
            optimization_level=optimization_level,
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
            gpu_fallback=True,
            fp16=True,
            int8=True,
            cache=cache,
            verbose=verbose,
        )

        built_paths.append(output_paths[dnn_id])

        if verbose:
            LOG.info(f"DNN {dnn_id} engine saved to {output_paths[dnn_id]}")

    return built_paths


def build_engines(
    models: list[tuple[Path | str, Path | str]],
    calibration_batchers: list[AbstractBatcher],
    *,
    objective: Objective = Objective.MAX_THROUGHPUT,
    dynamic: bool = False,
    # HaX-CoNN parameters
    dla_core: int = 0,
    pccs_alpha: float = 0.5,
    pccs_beta: float = 1.5,
    max_bandwidth_mbps: float = 25600.0,
    # Profiling parameters
    profile_iterations: int = 1000,
    warmup_iterations: int = 50,
    # D-HaX-CoNN parameters
    dynamic_budget_s: float = 30.0,
    dynamic_max_rounds: int = 10,
    # TensorRT build parameters
    workspace: float = 4.0,
    timing_cache: Path | str | None = None,
    calibration_cache: Path | str | None = None,
    shapes: list[list[tuple[str, tuple[int, ...]]] | None] | None = None,
    optimization_level: int = 3,
    cache: bool | None = None,
    verbose: bool | None = None,
) -> tuple[MultiSchedule, HaxconnExecutor]:
    """
    Build contention-aware multi-DNN TensorRT engines using HaX-CoNN.

    HaX-CoNN schedules multiple concurrent DNNs across GPU and DLA,
    modeling shared memory contention via the PCCS model and optimizing
    for either maximum throughput or minimum max-latency.

    Reference: HaX-CoNN: Shared Memory Multi-Accelerator Execution of
    Concurrent DNN Workloads (PPoPP 2024)

    Parameters
    ----------
    models : list[tuple[Path | str, Path | str]]
        List of (onnx_path, output_engine_path) tuples, one per DNN.
    calibration_batchers : list[AbstractBatcher]
        One data batcher per DNN for INT8 calibration.
    objective : Objective, optional
        Optimization objective. Default MAX_THROUGHPUT.
    dynamic : bool, optional
        If True, use D-HaX-CoNN for runtime schedule improvement. Default False.
    dla_core : int, optional
        DLA core to use (0 or 1). Default 0.
    pccs_alpha : float, optional
        PCCS model alpha parameter. Default 0.5.
    pccs_beta : float, optional
        PCCS model beta parameter. Default 1.5.
    max_bandwidth_mbps : float, optional
        Maximum platform memory bandwidth in MB/s. Default 25600.0.
    profile_iterations : int, optional
        Number of profiling iterations. Default 1000.
    warmup_iterations : int, optional
        Warmup iterations before profiling. Default 50.
    dynamic_budget_s : float, optional
        Time budget for D-HaX-CoNN dynamic improvement. Default 30.0.
    dynamic_max_rounds : int, optional
        Maximum D-HaX-CoNN improvement rounds. Default 10.
    workspace : float, optional
        TensorRT workspace size in GB. Default 4.0.
    timing_cache : Path | str | None, optional
        Path to timing cache file.
    calibration_cache : Path | str | None, optional
        Path to calibration cache file.
    shapes : list[list[tuple[str, tuple[int, ...]]] | None] | None, optional
        Per-DNN fixed input shapes.
    optimization_level : int, optional
        TensorRT optimization level (0-5). Default 3.
    cache : bool | None, optional
        Cache the engines. Default None.
    verbose : bool | None, optional
        Enable verbose output. Default None.

    Returns
    -------
    tuple[MultiSchedule, HaxconnExecutor]
        The optimized schedule and a ready-to-use concurrent executor.

    Examples
    --------
    >>> from trtutils.builder import ImageBatcher
    >>> from trtutils.research.haxconn import build_engines, Objective
    >>>
    >>> batcher_a = ImageBatcher("calib_a/", (640, 640, 3), np.float32)
    >>> batcher_b = ImageBatcher("calib_b/", (640, 640, 3), np.float32)
    >>>
    >>> schedule, executor = build_engines(
    ...     models=[("model_a.onnx", "model_a.engine"),
    ...             ("model_b.onnx", "model_b.engine")],
    ...     calibration_batchers=[batcher_a, batcher_b],
    ...     objective=Objective.MAX_THROUGHPUT,
    ...     verbose=True,
    ... )
    >>>
    >>> results = executor.execute([input_a, input_b])

    """
    if len(models) != len(calibration_batchers):
        err_msg = (
            f"Number of models ({len(models)}) must match "
            f"number of calibration batchers ({len(calibration_batchers)})"
        )
        raise ValueError(err_msg)

    # Create internal config
    config = HaxconnConfig(
        objective=objective,
        dla_core=dla_core,
        profile_iterations=profile_iterations,
        warmup_iterations=warmup_iterations,
        pccs_alpha=pccs_alpha,
        pccs_beta=pccs_beta,
        max_bandwidth_mbps=max_bandwidth_mbps,
        dynamic_budget_s=dynamic_budget_s,
        dynamic_max_rounds=dynamic_max_rounds,
    )

    onnx_paths = [Path(m[0]) for m in models]
    output_paths = [Path(m[1]) for m in models]

    if verbose:
        LOG.info(f"Building HaX-CoNN-optimized engines for {len(models)} DNNs")
        LOG.info(f"Objective: {objective.value}")
        LOG.info(f"PCCS params: alpha={pccs_alpha}, beta={pccs_beta}")

    # Profile all DNNs
    if verbose:
        LOG.info("Profiling all DNNs on GPU and DLA...")

    all_layers, all_groups, all_costs = profile_for_haxconn(
        onnx_paths=onnx_paths,
        calibration_batchers=calibration_batchers,
        config=config,
        workspace=workspace,
        timing_cache=timing_cache,
        calibration_cache=calibration_cache,
        verbose=verbose,
    )

    all_layers_count = [len(layers) for layers in all_layers]

    if dynamic:
        # D-HaX-CoNN: progressive improvement at runtime
        from ._dynamic import run_dynamic_haxconn  # noqa: PLC0415

        multi_schedule, engine_paths = run_dynamic_haxconn(
            onnx_paths=onnx_paths,
            output_paths=output_paths,
            calibration_batchers=calibration_batchers,
            config=config,
            all_groups=all_groups,
            all_costs=all_costs,
            all_layers_count=all_layers_count,
            workspace=workspace,
            timing_cache=timing_cache,
            calibration_cache=calibration_cache,
            shapes=shapes,
            optimization_level=optimization_level,
            cache=cache,
            verbose=verbose,
        )
    else:
        # Static HaX-CoNN: find optimal schedule
        if verbose:
            LOG.info("Finding optimal multi-DNN schedule...")

        multi_schedule = find_optimal_schedule(
            all_groups=all_groups,
            all_costs=all_costs,
            config=config,
            verbose=verbose,
        )

        if verbose:
            LOG.info(f"Optimal schedule: {multi_schedule}")

        # Build engines from schedule
        engine_paths = _build_engines_from_schedule(
            onnx_paths=onnx_paths,
            output_paths=output_paths,
            calibration_batchers=calibration_batchers,
            multi_schedule=multi_schedule,
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

    # Create executor
    executor = HaxconnExecutor(
        engine_paths=engine_paths,
        dla_core=config.dla_core,
        warmup=True,
        warmup_iterations=config.warmup_iterations,
        verbose=verbose,
    )

    if verbose:
        LOG.info(f"HaX-CoNN build complete: {multi_schedule}")
        for i, sched in enumerate(multi_schedule.dnn_schedules):
            LOG.info(f"  DNN {i}: {sched}")

    return multi_schedule, executor
