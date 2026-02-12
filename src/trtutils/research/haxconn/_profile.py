# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Profiling utilities for HaX-CoNN optimization with EMC measurement."""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):
    import tensorrt as trt

from trtutils._log import LOG
from trtutils.builder._build import build_engine
from trtutils.builder._dla import can_run_on_dla
from trtutils.builder._onnx import read_onnx
from trtutils.jetson._profile import profile_engine as jetson_profile_engine

from ._grouping import assign_groups_to_layers, identify_layer_groups
from ._types import HaxconnConfig, Layer, LayerGroupCost

if TYPE_CHECKING:
    from trtutils.builder._batcher import AbstractBatcher

    from ._types import LayerGroup


def _get_tensor_size(tensor: trt.ITensor) -> int:
    """
    Calculate the size of a tensor in bytes.

    Parameters
    ----------
    tensor : trt.ITensor
        The TensorRT tensor.

    Returns
    -------
    int
        Size in bytes.

    """
    shape = tensor.shape
    num_elements = 1
    for dim in shape:
        num_elements *= max(1, dim)

    dtype = tensor.dtype
    dtype_sizes = {
        trt.DataType.FLOAT: 4,
        trt.DataType.HALF: 2,
        trt.DataType.INT8: 1,
        trt.DataType.INT32: 4,
        trt.DataType.BOOL: 1,
        trt.DataType.UINT8: 1,
    }
    dtype_size = dtype_sizes.get(dtype, 4)

    return num_elements * dtype_size


def extract_layer_info(
    onnx: Path | str | trt.INetworkDefinition,
    config: trt.IBuilderConfig | None = None,
    *,
    verbose: bool | None = None,
) -> list[Layer]:
    """
    Extract layer information from an ONNX model.

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
    if isinstance(onnx, trt.INetworkDefinition):
        if config is None:
            err_msg = "Config must be provided when onnx is a network"
            raise ValueError(err_msg)
        network = onnx
    else:
        network, _, config, _ = read_onnx(onnx)

    _, chunks = can_run_on_dla(network, config, verbose_layers=False, verbose_chunks=False)

    dla_layer_indices: set[int] = set()
    for _layer_list, start, end, on_dla in chunks:
        if on_dla:
            for idx in range(start, end + 1):
                dla_layer_indices.add(idx)

    layers: list[Layer] = []

    for idx in range(network.num_layers):
        trt_layer = network.get_layer(idx)
        layer_type = str(trt_layer.type).split(".")[-1]

        output_size = 0
        for out_idx in range(trt_layer.num_outputs):
            output = trt_layer.get_output(out_idx)
            if output is not None:
                output_size += _get_tensor_size(output)

        input_size = 0
        for in_idx in range(trt_layer.num_inputs):
            inp = trt_layer.get_input(in_idx)
            if inp is not None:
                input_size += _get_tensor_size(inp)

        dla_valid = idx in dla_layer_indices

        layer = Layer(
            index=idx,
            name=trt_layer.name,
            layer_type=layer_type,
            output_tensor_size=output_size,
            can_run_on_dla=dla_valid,
            input_tensor_size=input_size,
        )
        layers.append(layer)

        if verbose:
            LOG.info(f"Layer {idx}: {layer}")

    return layers


def _estimate_mem_throughput_from_tensors(
    layer_time_ms: float,
    input_tensor_size: int,
    output_tensor_size: int,
) -> float:
    """
    Estimate memory throughput from tensor sizes and execution time.

    Fallback when EMC_FREQ is not available from tegrastats.

    Parameters
    ----------
    layer_time_ms : float
        Execution time in milliseconds.
    input_tensor_size : int
        Input tensor size in bytes.
    output_tensor_size : int
        Output tensor size in bytes.

    Returns
    -------
    float
        Estimated memory throughput in MB/s.

    """
    if layer_time_ms <= 0:
        return 0.0

    total_bytes = input_tensor_size + output_tensor_size
    total_mb = total_bytes / (1024 * 1024)
    time_s = layer_time_ms / 1000.0
    return total_mb / time_s


def profile_single_dnn(
    onnx_path: Path,
    dnn_id: int,
    calibration_batcher: AbstractBatcher,
    config: HaxconnConfig,
    workspace: float = 4.0,
    timing_cache: Path | str | None = None,
    calibration_cache: Path | str | None = None,
    *,
    verbose: bool | None = None,
) -> tuple[list[Layer], list[LayerGroup], dict[int, LayerGroupCost]]:
    """
    Profile a single DNN standalone on GPU and DLA.

    Captures per-layer timing, energy, and memory throughput. Groups layers
    into LayerGroups and aggregates per-group costs.

    Parameters
    ----------
    onnx_path : Path
        Path to the ONNX model.
    dnn_id : int
        Identifier for this DNN.
    calibration_batcher : AbstractBatcher
        Data batcher for INT8 calibration.
    config : HaxconnConfig
        HaX-CoNN configuration.
    workspace : float, optional
        Workspace size in GB for engine building.
    timing_cache : Path | str | None, optional
        Path to timing cache.
    calibration_cache : Path | str | None, optional
        Path to calibration cache.
    verbose : bool | None, optional
        Whether to print verbose output.

    Returns
    -------
    tuple[list[Layer], list[LayerGroup], dict[int, LayerGroupCost]]
        Layers, groups, and per-group cost data.

    """
    if verbose:
        LOG.info(f"Extracting layer information from {onnx_path}")

    layers = extract_layer_info(onnx_path, verbose=verbose)

    if verbose:
        dla_layers = sum(1 for layer in layers if layer.can_run_on_dla)
        LOG.info(f"Found {len(layers)} layers, {dla_layers} are DLA-compatible")

    # Create layer groups
    groups = identify_layer_groups(layers, dnn_id)
    layers = assign_groups_to_layers(layers, groups)

    if verbose:
        LOG.info(f"Identified {len(groups)} layer groups for DNN {dnn_id}")

    has_dla_layers = any(layer.can_run_on_dla for layer in layers)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # ---- GPU profiling ----
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
            verbose=verbose,
        )

        gpu_layer_data: dict[str, tuple[float, float]] = {}
        for layer_info in gpu_profile.layers:
            gpu_layer_data[layer_info.name] = (layer_info.mean, layer_info.energy)

        # ---- DLA profiling ----
        dla_layer_data: dict[str, tuple[float, float]] = {}

        if has_dla_layers:
            dla_engine_path = temp_path / "dla_engine.engine"

            if verbose:
                LOG.info("Building DLA-enabled INT8 engine for profiling...")

            layer_precision: list[tuple[int, trt.DataType | None]] = []
            layer_device: list[tuple[int, trt.DeviceType | None]] = []

            for layer in layers:
                if layer.can_run_on_dla:
                    layer_precision.append((layer.index, trt.DataType.INT8))
                    layer_device.append((layer.index, trt.DeviceType.DLA))
                else:
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
                    verbose=verbose,
                )

                for layer_info in dla_profile.layers:
                    dla_layer_data[layer_info.name] = (layer_info.mean, layer_info.energy)

            except (RuntimeError, OSError) as e:
                if verbose:
                    LOG.warning(f"Failed to build DLA engine: {e}")
                    LOG.warning("Continuing with GPU-only costs for DLA-compatible layers")

        # ---- Aggregate per-group costs ----
        group_costs: dict[int, LayerGroupCost] = {}

        for group in groups:
            gpu_time = 0.0
            gpu_energy = 0.0
            gpu_input_size = 0
            gpu_output_size = 0

            dla_time: float | None = None
            dla_energy: float | None = None
            dla_input_size = 0
            dla_output_size = 0

            if group.can_run_on_dla:
                dla_time = 0.0
                dla_energy = 0.0

            for layer_idx in group.layer_indices:
                layer = layers[layer_idx]

                # GPU costs
                if layer.name in gpu_layer_data:
                    t, e = gpu_layer_data[layer.name]
                    gpu_time += t
                    gpu_energy += e
                else:
                    if verbose:
                        LOG.warning(f"Layer {layer.name} not found in GPU profile, using estimate")
                    gpu_time += 0.01
                    gpu_energy += 0.001

                gpu_input_size += layer.input_tensor_size
                gpu_output_size += layer.output_tensor_size

                # DLA costs
                if group.can_run_on_dla:
                    if layer.name in dla_layer_data:
                        t, e = dla_layer_data[layer.name]
                        dla_time += t  # type: ignore[operator]
                        dla_energy += e  # type: ignore[operator]
                    elif layer.can_run_on_dla:
                        if verbose:
                            LOG.warning(f"Layer {layer.name} DLA profile missing, using estimate")
                        dla_time += gpu_time * 1.5  # type: ignore[operator]
                        dla_energy += gpu_energy * 0.5  # type: ignore[operator]

                    dla_input_size += layer.input_tensor_size
                    dla_output_size += layer.output_tensor_size

            # Estimate memory throughput from tensor sizes and execution time
            gpu_mem_tput = _estimate_mem_throughput_from_tensors(
                gpu_time, gpu_input_size, gpu_output_size
            )

            dla_mem_tput: float | None = None
            if dla_time is not None and dla_time > 0:
                dla_mem_tput = _estimate_mem_throughput_from_tensors(
                    dla_time, dla_input_size, dla_output_size
                )

            cost = LayerGroupCost(
                gpu_time_ms=gpu_time,
                gpu_energy_mj=gpu_energy,
                gpu_mem_throughput_mbps=gpu_mem_tput,
                dla_time_ms=dla_time,
                dla_energy_mj=dla_energy,
                dla_mem_throughput_mbps=dla_mem_tput,
            )
            group_costs[group.group_id] = cost

            if verbose:
                LOG.info(f"Cost for group {group.group_id}: {cost}")

    if verbose:
        total_gpu_time = sum(c.gpu_time_ms for c in group_costs.values())
        total_gpu_energy = sum(c.gpu_energy_mj for c in group_costs.values())
        LOG.info(
            f"DNN {dnn_id} total GPU time: {total_gpu_time:.2f}ms, energy: {total_gpu_energy:.2f}mJ"
        )

    return layers, groups, group_costs


def profile_for_haxconn(
    onnx_paths: list[Path],
    calibration_batchers: list[AbstractBatcher],
    config: HaxconnConfig,
    workspace: float = 4.0,
    timing_cache: Path | str | None = None,
    calibration_cache: Path | str | None = None,
    *,
    verbose: bool | None = None,
) -> tuple[list[list[Layer]], list[list[LayerGroup]], list[dict[int, LayerGroupCost]]]:
    """
    Profile all DNNs for HaX-CoNN optimization.

    Parameters
    ----------
    onnx_paths : list[Path]
        Paths to each DNN's ONNX model.
    calibration_batchers : list[AbstractBatcher]
        One calibration batcher per DNN.
    config : HaxconnConfig
        HaX-CoNN configuration.
    workspace : float, optional
        Workspace size in GB for engine building.
    timing_cache : Path | str | None, optional
        Path to timing cache.
    calibration_cache : Path | str | None, optional
        Path to calibration cache.
    verbose : bool | None, optional
        Whether to print verbose output.

    Returns
    -------
    tuple[list[list[Layer]], list[list[LayerGroup]], list[dict[int, LayerGroupCost]]]
        Per-DNN layers, groups, and costs.

    """
    all_layers: list[list[Layer]] = []
    all_groups: list[list[LayerGroup]] = []
    all_costs: list[dict[int, LayerGroupCost]] = []

    for dnn_id, (onnx_path, batcher) in enumerate(zip(onnx_paths, calibration_batchers)):
        if verbose:
            LOG.info(f"Profiling DNN {dnn_id}: {onnx_path}")

        layers, groups, costs = profile_single_dnn(
            onnx_path=onnx_path,
            dnn_id=dnn_id,
            calibration_batcher=batcher,
            config=config,
            workspace=workspace,
            timing_cache=timing_cache,
            calibration_cache=calibration_cache,
            verbose=verbose,
        )

        all_layers.append(layers)
        all_groups.append(groups)
        all_costs.append(costs)

    if verbose:
        LOG.info(f"Profiling complete for {len(onnx_paths)} DNNs")

    return all_layers, all_groups, all_costs
