# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Core AxoNN algorithm: profile, solve, and build."""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from trtutils._log import LOG
from trtutils.builder._build import build_engine as trt_build_engine
from trtutils.compat._libs import trt
from trtutils.inspect._onnx import inspect_onnx_layers
from trtutils.jetson._profile import profile_engine as jetson_profile_engine
from trtutils.profiling._fusion import build_fused_layer_map, resolve_fused_layer_value

from ._types import (
    LayerCost,
    Processor,
    Schedule,
    TransitionCosts,
)

if TYPE_CHECKING:
    from trtutils.builder._batcher import AbstractBatcher
    from trtutils.inspect._types import LayerInfo

from z3 import Bool, If, Not, Solver, Xor, sat


def _compute_schedule_costs(
    layers: list[LayerInfo],
    costs: dict[int, LayerCost],
    schedule: Schedule,
    tc: TransitionCosts,
) -> tuple[float, float]:
    """Compute total (time_ms, energy_mj) for a schedule including transitions."""
    layer_by_idx = {layer.index: layer for layer in layers}
    total_time = 0.0
    total_energy = 0.0
    sorted_indices = sorted(schedule.assignments.keys())

    for i, layer_idx in enumerate(sorted_indices):
        proc = schedule.assignments[layer_idx]
        cost = costs[layer_idx]

        if proc == Processor.DLA and cost.dla_time_ms is not None:
            total_time += cost.dla_time_ms
            total_energy += (
                cost.dla_energy_mj if cost.dla_energy_mj is not None else cost.gpu_energy_mj
            )
        else:
            total_time += cost.gpu_time_ms
            total_energy += cost.gpu_energy_mj

        # transition cost
        if i > 0:
            prev_idx = sorted_indices[i - 1]
            if schedule.assignments[prev_idx] != proc:
                tensor_mb = layer_by_idx[prev_idx].output_tensor_size / (1024 * 1024)
                total_time += tc.time_base_ms + tensor_mb * tc.time_per_mb
                total_energy += tc.energy_base_mj + tensor_mb * tc.energy_per_mb

    return total_time, total_energy


def _profile(
    onnx: Path,
    calibration_batcher: AbstractBatcher,
    *,
    dla_core: int = 0,
    profile_iterations: int = 1000,
    warmup_iterations: int = 50,
    workspace: float = 4.0,
    timing_cache: Path | str | None = None,
    calibration_cache: Path | str | None = None,
    cuda_graph: bool = False,
    verbose: bool | None = None,
) -> tuple[list[LayerInfo], dict[int, LayerCost]]:
    """Profile an ONNX model on GPU and DLA, return layers and per-layer costs."""
    if verbose:
        LOG.info(f"Extracting layer information from {onnx}")

    layers = inspect_onnx_layers(onnx, verbose=verbose)

    if verbose:
        dla_count = sum(1 for layer in layers if layer.dla_compatible)
        LOG.info(f"Found {len(layers)} layers, {dla_count} are DLA-compatible")

    has_dla_layers = any(layer.dla_compatible for layer in layers)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # -- GPU profiling --
        gpu_engine_path = temp_path / "gpu_engine.engine"
        if verbose:
            LOG.info("Building GPU-only FP16 engine for profiling...")

        trt_build_engine(
            onnx=onnx,
            output=gpu_engine_path,
            workspace=workspace,
            timing_cache=timing_cache,
            fp16=True,
            int8=False,
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
            verbose=verbose,
        )

        if verbose:
            LOG.info("Profiling GPU engine...")

        gpu_profile = jetson_profile_engine(
            engine=gpu_engine_path,
            iterations=profile_iterations,
            warmup_iterations=warmup_iterations,
            tegra_interval=5,
            dla_core=None,
            warmup=True,
            cuda_graph=cuda_graph,
            verbose=verbose,
        )

        # build gpu layer data: name -> (time, energy)
        # distribute total energy proportionally by execution time
        gpu_total_energy = gpu_profile.energy.mean
        gpu_total_time = sum(li.mean for li in gpu_profile.layers)
        gpu_times: dict[str, float] = {}
        gpu_energies: dict[str, float] = {}
        for li in gpu_profile.layers:
            frac = li.mean / gpu_total_time if gpu_total_time > 0 else 0.0
            gpu_times[li.name] = li.mean
            gpu_energies[li.name] = gpu_total_energy * frac

        gc.collect()

        # -- DLA profiling --
        dla_times: dict[str, float] = {}
        dla_energies: dict[str, float] = {}

        if has_dla_layers:
            dla_engine_path = temp_path / "dla_engine.engine"
            if verbose:
                LOG.info("Building DLA-enabled INT8 engine for profiling...")

            layer_precision: list[tuple[int, trt.DataType | None]] = []
            layer_device: list[tuple[int, trt.DeviceType | None]] = []
            for layer in layers:
                if layer.dla_compatible:
                    layer_precision.append((layer.index, trt.DataType.INT8))
                    layer_device.append((layer.index, trt.DeviceType.DLA))
                else:
                    layer_precision.append((layer.index, None))
                    layer_device.append((layer.index, trt.DeviceType.GPU))

            try:
                trt_build_engine(
                    onnx=onnx,
                    output=dla_engine_path,
                    default_device=trt.DeviceType.DLA,
                    workspace=workspace,
                    timing_cache=timing_cache,
                    calibration_cache=calibration_cache,
                    data_batcher=calibration_batcher,
                    layer_precision=layer_precision,
                    layer_device=layer_device,
                    dla_core=dla_core,
                    gpu_fallback=True,
                    fp16=True,
                    int8=True,
                    profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
                    verbose=verbose,
                )

                if verbose:
                    LOG.info("Profiling DLA engine...")

                dla_profile = jetson_profile_engine(
                    engine=dla_engine_path,
                    iterations=profile_iterations,
                    warmup_iterations=warmup_iterations,
                    tegra_interval=5,
                    dla_core=dla_core,
                    warmup=True,
                    cuda_graph=False,
                    verbose=verbose,
                )

                dla_total_energy = dla_profile.energy.mean
                dla_total_time = sum(li.mean for li in dla_profile.layers)
                for li in dla_profile.layers:
                    frac = li.mean / dla_total_time if dla_total_time > 0 else 0.0
                    dla_times[li.name] = li.mean
                    dla_energies[li.name] = dla_total_energy * frac

            except (RuntimeError, OSError) as e:
                if verbose:
                    LOG.warning(f"Failed to build DLA engine: {e}")
                    LOG.warning("Continuing with GPU-only costs for DLA-compatible layers")

        # resolve fused layer mappings
        onnx_names = [layer.name for layer in layers]
        gpu_fused_map = build_fused_layer_map(list(gpu_times.keys()), onnx_names)
        dla_fused_map = build_fused_layer_map(list(dla_times.keys()), onnx_names)

        # resolve GPU costs per layer
        gpu_per_layer: dict[int, tuple[float, float]] = {}
        for layer in layers:
            t = resolve_fused_layer_value(
                layer.name, gpu_times, gpu_fused_map, verbose=verbose, label="GPU"
            )
            e = resolve_fused_layer_value(
                layer.name, gpu_energies, gpu_fused_map, verbose=verbose, label="GPU"
            )
            gpu_per_layer[layer.index] = (
                t if t is not None else 0.01,
                e if e is not None else 0.001,
            )

        # resolve DLA costs with proportional blob distribution
        dla_per_layer: dict[int, tuple[float, float]] = {}
        blob_members: dict[str, list[int]] = {}
        for layer in layers:
            if not layer.dla_compatible:
                continue
            if layer.name in dla_times:
                dla_per_layer[layer.index] = (dla_times[layer.name], dla_energies[layer.name])
                continue
            if layer.name in dla_fused_map:
                fused_name, _ = dla_fused_map[layer.name]
                blob_members.setdefault(fused_name, []).append(layer.index)

        for fused_name, member_indices in blob_members.items():
            blob_time = dla_times[fused_name]
            blob_energy = dla_energies[fused_name]
            total_gpu_t = sum(gpu_per_layer[i][0] for i in member_indices)
            total_gpu_e = sum(gpu_per_layer[i][1] for i in member_indices)
            for idx in member_indices:
                gpu_t, gpu_e = gpu_per_layer[idx]
                t_w = gpu_t / total_gpu_t if total_gpu_t > 0 else 1.0 / len(member_indices)
                e_w = gpu_e / total_gpu_e if total_gpu_e > 0 else 1.0 / len(member_indices)
                dla_per_layer[idx] = (blob_time * t_w, blob_energy * e_w)

        # build LayerCost objects
        costs: dict[int, LayerCost] = {}
        for layer in layers:
            gpu_time, gpu_energy = gpu_per_layer[layer.index]
            dla_time: float | None = None
            dla_energy: float | None = None

            if layer.dla_compatible:
                if layer.index in dla_per_layer:
                    dla_time, dla_energy = dla_per_layer[layer.index]
                else:
                    if verbose:
                        LOG.warning(f"Layer {layer.name} DLA profile missing, using estimate")
                    dla_time = gpu_time * 1.5
                    dla_energy = gpu_energy * 0.5

            costs[layer.index] = LayerCost(
                gpu_time_ms=gpu_time,
                gpu_energy_mj=gpu_energy,
                dla_time_ms=dla_time,
                dla_energy_mj=dla_energy,
            )

    if verbose:
        total_gpu_time = sum(c.gpu_time_ms for c in costs.values())
        total_gpu_energy = sum(c.gpu_energy_mj for c in costs.values())
        LOG.info(f"Total GPU time: {total_gpu_time:.2f}ms, energy: {total_gpu_energy:.2f}mJ")

    return layers, costs


def _solve(
    layers: list[LayerInfo],
    costs: dict[int, LayerCost],
    *,
    energy_target_mj: float | None = None,
    energy_ratio: float = 0.8,
    max_transitions: int = 1,
    tc: TransitionCosts,
    z3_timeout_ms: int = 30000,
    search_tolerance: float = 0.001,
    verbose: bool | None = None,
) -> Schedule | None:
    """Find optimal layer schedule using Z3 binary search."""
    # compute energy target
    if energy_target_mj is None:
        gpu_energy = sum(c.gpu_energy_mj for c in costs.values())
        energy_target_mj = gpu_energy * energy_ratio
        if verbose:
            LOG.info(f"Using energy target: {energy_target_mj:.2f}mJ ({energy_ratio * 100}% of GPU)")

    n = len(layers)
    if verbose:
        LOG.info(f"Solving for {n} layers with energy target {energy_target_mj:.2f}mJ")

    # time bounds for binary search
    gpu_time = sum(c.gpu_time_ms for c in costs.values())
    min_time = 0.0
    for layer in layers:
        cost = costs[layer.index]
        gpu_t = cost.gpu_time_ms
        dla_t = cost.dla_time_ms if cost.dla_time_ms is not None else gpu_t
        min_time += min(gpu_t, dla_t)

    def _check_feasible(time_budget: float) -> Schedule | None:
        solver = Solver()
        solver.set("timeout", z3_timeout_ms)

        use_dla = [Bool(f"dla_{i}") for i in range(n)]

        # GPU-only layers must stay on GPU
        for layer in layers:
            if not layer.dla_compatible:
                solver.add(Not(use_dla[layer.index]))

        # transition counting
        trans_exprs = [Xor(use_dla[i], use_dla[i + 1]) for i in range(n - 1)]
        if trans_exprs:
            solver.add(sum(If(t, 1, 0) for t in trans_exprs) <= max_transitions)

        # time and energy expressions
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

        # transition costs
        for i in range(n - 1):
            tensor_mb = layers[i].output_tensor_size / (1024 * 1024)
            t_cost = tc.time_base_ms + tensor_mb * tc.time_per_mb
            e_cost = tc.energy_base_mj + tensor_mb * tc.energy_per_mb
            time_terms.append(If(trans_exprs[i], t_cost, 0.0))
            energy_terms.append(If(trans_exprs[i], e_cost, 0.0))

        eps = 1e-6
        solver.add(sum(energy_terms) <= energy_target_mj + eps)
        solver.add(sum(time_terms) <= time_budget + eps)

        if solver.check() != sat:
            return None

        model = solver.model()
        schedule = Schedule()
        for layer in layers:
            is_dla = bool(model[use_dla[layer.index]])
            schedule.assignments[layer.index] = Processor.DLA if is_dla else Processor.GPU
        schedule.total_time_ms, schedule.total_energy_mj = _compute_schedule_costs(
            layers,
            costs,
            schedule,
            tc,
        )
        return schedule

    if verbose:
        LOG.info(f"Time bounds: [{min_time:.4f}, {gpu_time:.4f}] ms")
        LOG.info("Running binary search with Z3 satisfiability checks...")

    # initial feasibility check at upper bound
    best = _check_feasible(gpu_time)
    if best is None:
        if verbose:
            LOG.warning("No feasible schedule found within energy constraint")
        return None

    # binary search to minimize time
    lo, hi = min_time, gpu_time
    for iteration in range(30):
        if hi - lo < search_tolerance:
            break
        mid = (lo + hi) / 2.0
        candidate = _check_feasible(mid)
        if candidate is not None:
            best = candidate
            hi = mid
            if verbose:
                LOG.info(
                    f"  Iteration {iteration + 1}: budget={mid:.4f}ms -> "
                    f"feasible (time={candidate.total_time_ms:.4f}ms)"
                )
        else:
            lo = mid
            if verbose:
                LOG.info(f"  Iteration {iteration + 1}: budget={mid:.4f}ms -> infeasible")

    return best


def build_engine(
    onnx: Path | str,
    output: Path | str,
    calibration_batcher: AbstractBatcher,
    *,
    energy_target: float | None = None,
    energy_ratio: float = 0.8,
    max_transitions: int = 1,
    transition_time_base_ms: float = 0.1,
    transition_time_per_mb: float = 0.05,
    transition_energy_base_mj: float = 0.01,
    transition_energy_per_mb: float = 0.005,
    z3_timeout_ms: int = 30000,
    search_tolerance: float = 0.001,
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
        0.8. Only used when ``energy_target`` is None.
    max_transitions : int, optional
        Maximum number of GPU<->DLA device transitions allowed. Default 1.
    transition_time_base_ms : float, optional
        Base time overhead per GPU<->DLA transition in ms. Default 0.1.
    transition_time_per_mb : float, optional
        Additional time per MB of tensor data transferred. Default 0.05.
    transition_energy_base_mj : float, optional
        Base energy overhead per transition in mJ. Default 0.01.
    transition_energy_per_mb : float, optional
        Additional energy per MB of tensor data transferred. Default 0.005.
    z3_timeout_ms : int, optional
        Timeout per Z3 satisfiability check in milliseconds. Default 30000.
    search_tolerance : float, optional
        Binary search convergence tolerance in ms. Default 0.001.
    dla_core : int, optional
        DLA core to use (0 or 1). Default 0.
    profile_iterations : int, optional
        Number of inference iterations for profiling. Default 1000.
    warmup_iterations : int, optional
        Number of warmup iterations before profiling. Default 50.
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

    """
    onnx_path = Path(onnx)
    output_path = Path(output)

    if verbose:
        LOG.info(f"Building AxoNN-optimized engine from {onnx_path}")
        LOG.info(f"Energy target: {energy_target}mJ (ratio={energy_ratio})")
        LOG.info(f"Max transitions: {max_transitions}, DLA core: {dla_core}")

    # profile
    layers, costs = _profile(
        onnx=onnx_path,
        calibration_batcher=calibration_batcher,
        dla_core=dla_core,
        profile_iterations=profile_iterations,
        warmup_iterations=warmup_iterations,
        workspace=workspace,
        timing_cache=timing_cache,
        calibration_cache=calibration_cache,
        verbose=verbose,
    )

    gpu_time = sum(c.gpu_time_ms for c in costs.values())

    if verbose:
        gpu_energy = sum(c.gpu_energy_mj for c in costs.values())
        LOG.info(f"GPU-only baseline: {gpu_time:.2f}ms, {gpu_energy:.2f}mJ")

    # solve
    tc = TransitionCosts(
        time_base_ms=transition_time_base_ms,
        time_per_mb=transition_time_per_mb,
        energy_base_mj=transition_energy_base_mj,
        energy_per_mb=transition_energy_per_mb,
    )
    schedule = _solve(
        layers=layers,
        costs=costs,
        energy_target_mj=energy_target,
        energy_ratio=energy_ratio,
        max_transitions=max_transitions,
        tc=tc,
        z3_timeout_ms=z3_timeout_ms,
        search_tolerance=search_tolerance,
        verbose=verbose,
    )

    if schedule is None:
        if verbose:
            LOG.warning("No feasible AxoNN schedule found, using GPU-only schedule")
        schedule = Schedule(
            assignments={layer.index: Processor.GPU for layer in layers},
            total_time_ms=gpu_time,
            total_energy_mj=sum(c.gpu_energy_mj for c in costs.values()),
        )

    # convert schedule to layer assignments
    num_layers = len(layers)
    layer_precision = [
        (i, trt.DataType.INT8 if schedule.assignments.get(i) == Processor.DLA else None)
        for i in range(num_layers)
    ]
    layer_device = [
        (
            i,
            trt.DeviceType.DLA
            if schedule.assignments.get(i) == Processor.DLA
            else trt.DeviceType.GPU,
        )
        for i in range(num_layers)
    ]

    has_dla = any(p == Processor.DLA for p in schedule.assignments.values())
    dla_count = sum(1 for p in schedule.assignments.values() if p == Processor.DLA)
    gpu_count = sum(1 for p in schedule.assignments.values() if p == Processor.GPU)

    if verbose:
        LOG.info(f"Building engine with {gpu_count} GPU layers, {dla_count} DLA layers")

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
        dla_core=dla_core if has_dla else None,
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
