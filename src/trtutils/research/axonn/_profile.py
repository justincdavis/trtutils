# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Profiling utilities for AxoNN optimization."""

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
from trtutils.builder._utils import get_check_dla
from trtutils.jetson._profile import profile_engine as jetson_profile_engine

from ._types import AxoNNConfig, Layer, LayerCost

if TYPE_CHECKING:
    from trtutils.builder._batcher import AbstractBatcher


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
    # Handle dynamic dimensions by assuming 1
    num_elements = 1
    for dim in shape:
        num_elements *= max(1, dim)

    # Get dtype size
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
    onnx: Path | str,
    *,
    verbose: bool | None = None,
) -> list[Layer]:
    """
    Extract layer information from an ONNX model.

    Parameters
    ----------
    onnx : Path | str
        Path to the ONNX model.
    verbose : bool | None, optional
        Whether to print verbose output.

    Returns
    -------
    list[Layer]
        List of Layer objects with metadata.

    """
    network, _, config, _ = read_onnx(onnx)
    check_dla = get_check_dla(config)

    # Assign to DLA 0 for compatibility check
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = 0

    layers: list[Layer] = []

    for idx in range(network.num_layers):
        trt_layer = network.get_layer(idx)

        # Get layer type as string
        layer_type = str(trt_layer.type).split(".")[-1]

        # Calculate output tensor size
        output_size = 0
        for out_idx in range(trt_layer.num_outputs):
            output = trt_layer.get_output(out_idx)
            if output is not None:
                output_size += _get_tensor_size(output)

        # Calculate input tensor size
        input_size = 0
        for in_idx in range(trt_layer.num_inputs):
            inp = trt_layer.get_input(in_idx)
            if inp is not None:
                input_size += _get_tensor_size(inp)

        # Check DLA compatibility
        dla_valid = check_dla(trt_layer)

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


def profile_for_axonn(
    onnx: Path | str,
    calibration_batcher: AbstractBatcher,
    config: AxoNNConfig | None = None,
    workspace: float = 4.0,
    timing_cache: Path | str | None = None,
    calibration_cache: Path | str | None = None,
    *,
    verbose: bool | None = None,
) -> tuple[list[Layer], list[LayerCost]]:
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
    verbose : bool | None, optional
        Whether to print verbose output.

    Returns
    -------
    tuple[list[Layer], list[LayerCost]]
        Layer information and cost data for each layer.

    """
    if config is None:
        config = AxoNNConfig()

    onnx_path = Path(onnx)

    if verbose:
        LOG.info(f"Extracting layer information from {onnx_path}")

    # Extract layer information
    layers = extract_layer_info(onnx_path, verbose=verbose)

    if verbose:
        dla_layers = sum(1 for l in layers if l.can_run_on_dla)
        LOG.info(f"Found {len(layers)} layers, {dla_layers} are DLA-compatible")

    # Check if any layers can run on DLA
    has_dla_layers = any(l.can_run_on_dla for l in layers)

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
            verbose=verbose,
        )

        # Build layer name to GPU timing/energy mapping
        gpu_layer_data: dict[str, tuple[float, float]] = {}
        for layer_info in gpu_profile.layers:
            gpu_layer_data[layer_info.name] = (layer_info.mean, layer_info.energy)

        # Build and profile DLA engine if there are DLA-compatible layers
        dla_layer_data: dict[str, tuple[float, float]] = {}

        if has_dla_layers:
            dla_engine_path = temp_path / "dla_engine.engine"

            if verbose:
                LOG.info("Building DLA-enabled INT8 engine for profiling...")

            # Get DLA chunks info
            full_dla, chunks = can_run_on_dla(onnx_path, verbose_chunks=verbose)

            # Build with DLA using the existing build_dla infrastructure
            # We need to set up layer assignments for DLA-compatible layers
            network, _, trt_config, _ = read_onnx(onnx_path)
            check_dla = get_check_dla(trt_config)

            # Build layer assignments
            layer_precision: list[tuple[int, trt.DataType | None]] = []
            layer_device: list[tuple[int, trt.DeviceType | None]] = []

            for layer in layers:
                if layer.can_run_on_dla:
                    layer_precision.append((layer.index, trt.DataType.INT8))
                    layer_device.append((layer.index, trt.DeviceType.DLA))
                else:
                    layer_precision.append((layer.index, trt.DataType.HALF))
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

            except Exception as e:
                if verbose:
                    LOG.warning(f"Failed to build DLA engine: {e}")
                    LOG.warning("Continuing with GPU-only costs for DLA-compatible layers")

        # Build LayerCost objects
        costs: list[LayerCost] = []

        for layer in layers:
            # Get GPU costs
            gpu_time = 0.0
            gpu_energy = 0.0
            if layer.name in gpu_layer_data:
                gpu_time, gpu_energy = gpu_layer_data[layer.name]
            else:
                # Layer might have been fused, try to estimate
                if verbose:
                    LOG.warning(f"Layer {layer.name} not found in GPU profile, using estimate")
                gpu_time = 0.01  # Small default
                gpu_energy = 0.001

            # Get DLA costs if applicable
            dla_time: float | None = None
            dla_energy: float | None = None

            if layer.can_run_on_dla and layer.name in dla_layer_data:
                dla_time, dla_energy = dla_layer_data[layer.name]
            elif layer.can_run_on_dla:
                # Layer is DLA-compatible but no profile data
                # Estimate DLA costs based on GPU costs
                # DLA is typically slower but more energy efficient
                if verbose:
                    LOG.warning(f"Layer {layer.name} DLA profile missing, using estimate")
                dla_time = gpu_time * 1.5  # DLA often slower
                dla_energy = gpu_energy * 0.5  # DLA more efficient

            cost = LayerCost(
                layer_idx=layer.index,
                layer_name=layer.name,
                gpu_time_ms=gpu_time,
                gpu_energy_mj=gpu_energy,
                dla_time_ms=dla_time,
                dla_energy_mj=dla_energy,
            )
            costs.append(cost)

            if verbose:
                LOG.info(f"Cost for {layer.name}: {cost}")

    if verbose:
        total_gpu_time = sum(c.gpu_time_ms for c in costs)
        total_gpu_energy = sum(c.gpu_energy_mj for c in costs)
        LOG.info(f"Total GPU time: {total_gpu_time:.2f}ms, energy: {total_gpu_energy:.2f}mJ")

        dla_costs = [c for c in costs if c.dla_time_ms is not None]
        if dla_costs:
            total_dla_time = sum(c.dla_time_ms for c in dla_costs)  # type: ignore[misc]
            total_dla_energy = sum(c.dla_energy_mj for c in dla_costs if c.dla_energy_mj)
            LOG.info(
                f"Total DLA-compatible layer time: {total_dla_time:.2f}ms, "
                f"energy: {total_dla_energy:.2f}mJ"
            )

    return layers, costs
