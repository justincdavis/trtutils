# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):
    import tensorrt as trt  # type: ignore[import-untyped, import-not-found]

from trtutils._log import LOG

from ._build import build_engine
from ._onnx import read_onnx
from ._utils import get_check_dla

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ._batcher import AbstractBatcher


def can_run_on_dla(
    onnx: Path | str | trt.INetworkDefinition,
    config: trt.IBuilderConfig | None = None,
    *,
    verbose_layers: bool | None = None,
    verbose_chunks: bool | None = None,
) -> tuple[bool, list[tuple[list[trt.ILayer], int, int, bool]]]:
    """
    Whether or not the entire model can be run on a DLA.

    Parameters
    ----------
    onnx : Path, str, or trt.INetworkDefinition
        The path to the onnx file or a pre-made TensorRT network.
    config : trt.IBuilderConfig, optional
        The TensorRT builder config. Required if onnx is a network.
    verbose_layers : bool, optional
        Whether to print verbose output for individual layers, by default None
    verbose_chunks : bool, optional
        Whether to print verbose output for layer chunks, by default None

    Returns
    -------
    tuple[bool, list[tuple[list[trt.ILayer], int, int, bool]]]
        Whether or not the model will all run on DLA and each block of layers.
        Where each block can run on a single device, DLA or GPU.

    Raises
    ------
    ValueError
        If config is not provided when onnx is a network

    """
    # handle network input
    if isinstance(onnx, trt.INetworkDefinition):
        if config is None:
            err_msg = "Config must be provided when onnx is a network"
            raise ValueError(err_msg)
        network = onnx
    else:
        network, _, config, _ = read_onnx(onnx)

    check_dla: Callable[[trt.ILayer], bool] = get_check_dla(config)

    # assign to DLA 0, since core doesnt matter for this check
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = 0

    full_dla = True
    last_layer_dla = False
    chunks: list[tuple[list[trt.ILayer], int, int, bool]] = []
    curr_start: int = 0
    curr_layers: list[trt.ILayer] = []

    for idx in range(network.num_layers):
        layer = network.get_layer(idx)

        # check if the layer can run on DLA
        dla_valid = check_dla(layer)
        if not dla_valid:
            full_dla = False

        # handle chunk storage
        if dla_valid != last_layer_dla and len(curr_layers) > 0:
            chunks.append((curr_layers, curr_start, idx - 1, last_layer_dla))
            curr_layers = [layer]
            curr_start = idx
        else:
            curr_layers.append(layer)

        last_layer_dla = dla_valid

        if verbose_layers:
            LOG.info(
                f"Layer {idx}: {layer.name}, {layer.type}, {layer.precision}, {layer.metadata}",
            )
            LOG.info(f"\tDLA: {dla_valid}")

    # handle final chunk
    chunks.append((curr_layers, curr_start, network.num_layers - 1, last_layer_dla))

    if verbose_chunks:
        LOG.info(f"Found {len(chunks)} Chunks of Layers")
        for i, (layers, start, end, on_dla) in enumerate(chunks):
            LOG.info(
                f"\tChunk {i}: [{start} - {end}], {len(layers)} layers, {'DLA' if on_dla else 'GPU'}"
            )

    return full_dla, chunks


def build_dla_engine(
    onnx: Path | str,
    output_path: Path | str,
    data_batcher: AbstractBatcher,
    dla_core: int,
    timing_cache: Path | str | None = None,
    *,
    verbose: bool | None = None,
) -> None:
    """
    Automatically build a TensorRT engine for DLA with automatic layer assignments.

    This function will:
    1. Check which layers can run on DLA
    2. Find the largest chunk of DLA-compatible layers
    3. Assign those layers to DLA with INT8 precision
    4. Assign remaining layers to GPU with FP16 precision

    Parameters
    ----------
    onnx : Path, str
        The path to the ONNX model or a pre-made TensorRT network
    output_path : Path, str
        The path where the engine should be saved
    data_batcher : AbstractBatcher
        The data batcher instance for INT8 calibration
    dla_core : int
        The DLA core to use
    timing_cache : Path, str, optional
        The path to the timing cache file
    verbose : bool, optional
        Whether to print verbose output, by default False

    """
    # read the onnx path
    network, _, config, _ = read_onnx(onnx)

    # check layers for DLA compatibility and use int8 precision
    full_dla, chunks = can_run_on_dla(
        onnx=network,
        config=config,
        verbose_layers=verbose,
        verbose_chunks=verbose,
    )

    if verbose:
        LOG.info(f"Model can run fully on DLA: {full_dla}")
        LOG.info(f"Found {len(chunks)} chunks of layers")

    # case where the entire model can run on DLA
    if full_dla:
        build_engine(
            onnx,
            output_path,
            data_batcher=data_batcher,
            dla_core=dla_core,
            fp16=True,
            int8=True,
            verbose=verbose,
        )
        return

    # identify if any chunks contain DLA layers
    dla_chunks = [(i, chunk) for i, chunk in enumerate(chunks) if chunk[3]]

    # case where no DLA layers are found
    if not dla_chunks:
        LOG.warning("No DLA-compatible layers found. Building GPU-only engine.")
        build_engine(
            onnx,
            output_path,
            fp16=True,
            verbose=verbose,
        )
        return

    # Get the largest DLA chunk
    _, largest_dla_chunk = max(dla_chunks, key=lambda x: len(x[1][0]))
    dla_start, dla_end = largest_dla_chunk[1], largest_dla_chunk[2]

    if verbose:
        LOG.info(f"Largest DLA chunk: layers {dla_start} to {dla_end}")

    layer_precision: list[tuple[int, trt.DataType | None]] = []
    layer_device: list[tuple[int, trt.DeviceType | None]] = []

    # create specific DLA layer assignments
    exclude_layer_types = [trt.LayerType.CONSTANT, trt.LayerType.SHUFFLE]
    for idx in range(network.num_layers):
        layer = network.get_layer(idx)
        layer_name: str = layer.name
        layer_name = layer_name.lower()
        LOG.debug(f"Assigning layer: {idx} - ({layer_name})")
        if dla_start <= idx <= dla_end:
            layer_precision.append((idx, trt.DataType.INT8))
            layer_device.append((idx, trt.DeviceType.DLA))
            if verbose:
                LOG.info(f"Assigning layer {idx} ({layer.name}) to DLA (INT8)")
        else:
            # if the layer is not in the DLA chunk, simply assign to GPU
            layer_device.append((idx, trt.DeviceType.GPU))
            if verbose:
                LOG.info(f"Assigning layer {idx} ({layer.name}) to GPU (FP16)")
            # intelligently assign precision level to HALF unless layer
            # is Constant, Shuffle, or Tile
            if layer.type in exclude_layer_types or "tile" in layer_name:
                layer_precision.append((idx, None))
            else:
                layer_precision.append((idx, trt.DataType.HALF))

    # build engine with specific layer assignments
    build_engine(
        onnx,
        output_path,
        default_device=trt.DeviceType.GPU,  # default device GPU
        timing_cache=timing_cache,
        data_batcher=data_batcher,
        layer_precision=layer_precision,
        layer_device=layer_device,
        dla_core=dla_core,  # ensure DLA core is maintained
        gpu_fallback=True,  # enable GPU fallback to account for input/copy
        fp16=True,
        int8=True,
        verbose=verbose,
    )
