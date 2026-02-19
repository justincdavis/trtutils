# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Profiling utilities for AxoNN optimization."""

from __future__ import annotations

import contextlib
import gc
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):
    import tensorrt as trt

from trtutils._log import LOG
from trtutils.builder._build import build_engine
from trtutils.inspect._onnx import inspect_onnx_layers
from trtutils.jetson._profile import profile_engine as jetson_profile_engine
from trtutils.profiling._fusion import build_fused_layer_map

from ._types import AxoNNConfig, Layer, LayerCost

if TYPE_CHECKING:
    from trtutils.builder._batcher import AbstractBatcher


def extract_layer_info(
    onnx: Path | str | trt.INetworkDefinition,
    config: trt.IBuilderConfig | None = None,
    *,
    verbose: bool | None = None,
) -> list[Layer]:
    """
    Extract layer information from an ONNX model.

    Delegates to :func:`trtutils.inspect.inspect_onnx_layers` and converts
    the result to AxoNN :class:`Layer` objects.

    Parameters
    ----------
    onnx : Path | str | trt.INetworkDefinition
        Path to the ONNX model or a pre-made TensorRT network.
    config : trt.IBuilderConfig | None, optional
        The TensorRT builder config. Required if onnx is a network.
    verbose : bool | None, optional
        Whether to print verbose output.

    Returns
    -------
    list[Layer]
        List of Layer objects with metadata.

    Raises
    ------
    ValueError
        If config is not provided when onnx is a network.

    """
    layer_infos = inspect_onnx_layers(onnx, config, verbose=verbose)
    return [Layer.from_layer_info(info) for info in layer_infos]


def _resolve_layer_cost(
    layer_name: str,
    layer_data: dict[str, tuple[float, float]],
    fused_map: dict[str, tuple[str, int]],
    *,
    verbose: bool | None = None,
    label: str = "",
) -> tuple[float, float] | None:
    """
    Look up profiling cost for a layer, handling TensorRT layer fusion.

    Resolution order:
    1. Exact match in ``layer_data``
    2. Constituent of a fused layer (cost split evenly among constituents)
    3. ``None`` — no data available

    Parameters
    ----------
    layer_name : str
        The ONNX layer name to look up.
    layer_data : dict[str, tuple[float, float]]
        Profiled (time_ms, energy_mj) keyed by TRT layer name.
    fused_map : dict[str, tuple[str, int]]
        Mapping from constituent names to (fused_name, num_parts).
    verbose : bool | None, optional
        Log warnings for unresolved layers.
    label : str, optional
        Label for log messages (e.g. "GPU" or "DLA").

    Returns
    -------
    tuple[float, float] | None
        (time_ms, energy_mj) or None if no data found.

    """
    # 1. Exact match
    if layer_name in layer_data:
        return layer_data[layer_name]

    # 2. Fused layer match — split cost evenly
    if layer_name in fused_map:
        fused_name, num_parts = fused_map[layer_name]
        total_time, total_energy = layer_data[fused_name]
        return (total_time / num_parts, total_energy / num_parts)

    # 3. No data
    if verbose:
        LOG.warning(f"Layer {layer_name} not found in {label} profile, using estimate")
    return None


def profile_for_axonn(
    onnx: Path | str,
    calibration_batcher: AbstractBatcher,
    config: AxoNNConfig | None = None,
    workspace: float = 4.0,
    timing_cache: Path | str | None = None,
    calibration_cache: Path | str | None = None,
    *,
    cuda_graph: bool = False,
    verbose: bool | None = None,
) -> tuple[list[Layer], dict[int, LayerCost]]:
    """
    Profile an ONNX model for AxoNN optimization.

    This function:
    1. Extracts layer information and DLA compatibility
    2. Builds a GPU-only FP16 engine and profiles it
    3. Builds a DLA-enabled INT8 engine and profiles it
    4. Returns layer info and cost data

    Parameters
    ----------
    onnx : Path | str
        Path to the ONNX model.
    calibration_batcher : AbstractBatcher
        Data batcher for INT8 calibration.
    config : AxoNNConfig | None, optional
        AxoNN configuration. If None, uses defaults.
    workspace : float, optional
        Workspace size in GB for engine building.
    timing_cache : Path | str | None, optional
        Path to timing cache.
    calibration_cache : Path | str | None, optional
        Path to calibration cache.
    cuda_graph : bool, optional
        Enable CUDA graph capture for GPU profiling. Default False.
    verbose : bool | None, optional
        Whether to print verbose output.

    Returns
    -------
    tuple[list[Layer], dict[int, LayerCost]]
        Layer information and cost data keyed by layer index.

    """
    if config is None:
        config = AxoNNConfig()

    onnx_path = Path(onnx)

    if verbose:
        LOG.info(f"Extracting layer information from {onnx_path}")

    # Extract layer information
    layers = extract_layer_info(onnx_path, verbose=verbose)

    if verbose:
        dla_layers = sum(1 for layer in layers if layer.can_run_on_dla)
        LOG.info(f"Found {len(layers)} layers, {dla_layers} are DLA-compatible")

    # Check if any layers can run on DLA
    has_dla_layers = any(layer.can_run_on_dla for layer in layers)

    # Create temporary directory for engine files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Build and profile GPU-only engine
        gpu_engine_path = temp_path / "gpu_engine.engine"

        if verbose:
            LOG.info("Building GPU-only FP16 engine for profiling...")

        build_engine(
            onnx=onnx_path,
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
            iterations=config.profile_iterations,
            warmup_iterations=config.warmup_iterations,
            tegra_interval=5,
            dla_core=None,
            warmup=True,
            cuda_graph=cuda_graph,
            verbose=verbose,
        )

        # Build layer name to GPU timing/energy mapping
        # Use total inference energy distributed proportionally by execution time.
        # Per-layer tegrastats energy is unreliable because individual layers
        # execute in microseconds while tegrastats samples every 5ms.
        gpu_total_energy = gpu_profile.energy.mean
        gpu_total_time = sum(li.mean for li in gpu_profile.layers)
        gpu_layer_data: dict[str, tuple[float, float]] = {}
        for layer_info in gpu_profile.layers:
            time_frac = layer_info.mean / gpu_total_time if gpu_total_time > 0 else 0.0
            layer_energy = gpu_total_energy * time_frac
            gpu_layer_data[layer_info.name] = (layer_info.mean, layer_energy)

        # ensure GPU engine resources are freed before DLA profiling
        gc.collect()

        # Build and profile DLA engine if there are DLA-compatible layers
        dla_layer_data: dict[str, tuple[float, float]] = {}

        if has_dla_layers:
            dla_engine_path = temp_path / "dla_engine.engine"

            if verbose:
                LOG.info("Building DLA-enabled INT8 engine for profiling...")

            # Build layer assignments from already-computed layer info
            layer_precision: list[tuple[int, trt.DataType | None]] = []
            layer_device: list[tuple[int, trt.DeviceType | None]] = []

            for layer in layers:
                if layer.can_run_on_dla:
                    layer_precision.append((layer.index, trt.DataType.INT8))
                    layer_device.append((layer.index, trt.DeviceType.DLA))
                else:
                    # GPU - don't lock precision, let TensorRT optimize
                    layer_precision.append((layer.index, None))
                    layer_device.append((layer.index, trt.DeviceType.GPU))

            try:
                build_engine(
                    onnx=onnx_path,
                    output=dla_engine_path,
                    default_device=trt.DeviceType.DLA,
                    workspace=workspace,
                    timing_cache=timing_cache,
                    calibration_cache=calibration_cache,
                    data_batcher=calibration_batcher,
                    layer_precision=layer_precision,
                    layer_device=layer_device,
                    dla_core=config.dla_core,
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
                    iterations=config.profile_iterations,
                    warmup_iterations=config.warmup_iterations,
                    tegra_interval=5,
                    dla_core=config.dla_core,
                    warmup=True,
                    cuda_graph=False,
                    verbose=verbose,
                )

                dla_total_energy = dla_profile.energy.mean
                dla_total_time = sum(li.mean for li in dla_profile.layers)
                for layer_info in dla_profile.layers:
                    time_frac = layer_info.mean / dla_total_time if dla_total_time > 0 else 0.0
                    layer_energy = dla_total_energy * time_frac
                    dla_layer_data[layer_info.name] = (layer_info.mean, layer_energy)

            except (RuntimeError, OSError) as e:
                if verbose:
                    LOG.warning(f"Failed to build DLA engine: {e}")
                    LOG.warning("Continuing with GPU-only costs for DLA-compatible layers")

        # Build reverse mapping: ONNX layer name -> fused TRT layer name
        # GPU engines fuse as "LayerA + LayerB", DLA uses "{ForeignNode[A...Z]}"
        onnx_layer_names = [layer.name for layer in layers]
        gpu_fused_map = build_fused_layer_map(
            list(gpu_layer_data.keys()),
            onnx_layer_names,
        )
        dla_fused_map = build_fused_layer_map(
            list(dla_layer_data.keys()),
            onnx_layer_names,
        )

        # Pass 1: resolve GPU costs for every layer
        gpu_per_layer: dict[int, tuple[float, float]] = {}
        for layer in layers:
            resolved_gpu = _resolve_layer_cost(
                layer.name,
                gpu_layer_data,
                gpu_fused_map,
                verbose=verbose,
                label="GPU",
            )
            if resolved_gpu is not None:
                gpu_per_layer[layer.index] = resolved_gpu
            else:
                gpu_per_layer[layer.index] = (0.01, 0.001)

        # Pass 2: resolve DLA costs using proportional weighting for blobs
        # For DLA ForeignNode blobs (which cover many ONNX layers), evenly
        # splitting the blob cost gives every layer the same DLA cost. This
        # makes the solver believe DLA is uniformly bad (or good) for all
        # layers. Instead, distribute the blob cost proportionally to each
        # layer's GPU cost so the relative layer-importance is preserved.
        dla_per_layer: dict[int, tuple[float, float]] = {}

        # Group layers by their DLA fused blob
        blob_members: dict[str, list[int]] = {}
        for layer in layers:
            if not layer.can_run_on_dla:
                continue
            # Exact match first
            if layer.name in dla_layer_data:
                dla_per_layer[layer.index] = dla_layer_data[layer.name]
                continue
            # Fused match
            if layer.name in dla_fused_map:
                fused_name, _ = dla_fused_map[layer.name]
                blob_members.setdefault(fused_name, []).append(layer.index)

        # Distribute each DLA blob's cost proportionally to GPU costs
        for fused_name, member_indices in blob_members.items():
            blob_time, blob_energy = dla_layer_data[fused_name]
            total_gpu_t = sum(gpu_per_layer[i][0] for i in member_indices)
            total_gpu_e = sum(gpu_per_layer[i][1] for i in member_indices)
            for idx in member_indices:
                gpu_t, gpu_e = gpu_per_layer[idx]
                t_weight = gpu_t / total_gpu_t if total_gpu_t > 0 else 1.0 / len(member_indices)
                e_weight = gpu_e / total_gpu_e if total_gpu_e > 0 else 1.0 / len(member_indices)
                dla_per_layer[idx] = (blob_time * t_weight, blob_energy * e_weight)

        # Build LayerCost objects
        costs: dict[int, LayerCost] = {}

        for layer in layers:
            gpu_time, gpu_energy = gpu_per_layer[layer.index]

            dla_time: float | None = None
            dla_energy: float | None = None

            if layer.can_run_on_dla:
                if layer.index in dla_per_layer:
                    dla_time, dla_energy = dla_per_layer[layer.index]
                else:
                    # DLA-compatible but no profile data at all
                    if verbose:
                        LOG.warning(f"Layer {layer.name} DLA profile missing, using estimate")
                    dla_time = gpu_time * 1.5  # DLA often slower
                    dla_energy = gpu_energy * 0.5  # DLA more efficient

            cost = LayerCost(
                gpu_time_ms=gpu_time,
                gpu_energy_mj=gpu_energy,
                dla_time_ms=dla_time,
                dla_energy_mj=dla_energy,
            )
            costs[layer.index] = cost

            if verbose:
                LOG.info(f"Cost for {layer.name}: {cost}")

    if verbose:
        total_gpu_time = sum(c.gpu_time_ms for c in costs.values())
        total_gpu_energy = sum(c.gpu_energy_mj for c in costs.values())
        LOG.info(f"Total GPU time: {total_gpu_time:.2f}ms, energy: {total_gpu_energy:.2f}mJ")

        dla_costs = [c for c in costs.values() if c.dla_time_ms is not None]
        if dla_costs:
            total_dla_time = sum(c.dla_time_ms for c in dla_costs)  # type: ignore[misc]
            total_dla_energy = sum(c.dla_energy_mj for c in dla_costs if c.dla_energy_mj)
            LOG.info(
                f"Total DLA-compatible layer time: {total_dla_time:.2f}ms, "
                f"energy: {total_dla_energy:.2f}mJ"
            )

    return layers, costs
