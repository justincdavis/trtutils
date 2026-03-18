# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils._log import LOG
from trtutils.builder._dla import can_run_on_dla
from trtutils.builder._onnx import read_onnx
from trtutils.compat._libs import trt

from ._tensor import get_tensor_size
from ._types import LayerInfo

if TYPE_CHECKING:
    from pathlib import Path


def inspect_onnx_layers(
    onnx: trt.INetworkDefinition | Path | str,
    config: trt.IBuilderConfig | None = None,
    *,
    verbose: bool | None = None,
) -> list[LayerInfo]:
    """
    Inspect the layers that TensorRT would create from an ONNX model.

    Returns detailed per-layer information including tensor sizes, precision,
    and DLA compatibility. On non-DLA systems, ``dla_compatible`` is always
    ``False``.

    Parameters
    ----------
    onnx : Path | str | trt.INetworkDefinition
        Path to the ONNX model or an already constructed TensorRT network.
    config : trt.IBuilderConfig | None, optional
        The TensorRT builder config. Required if ``onnx`` is a pre-built
        network and DLA compatibility checking is desired.
    verbose : bool | None, optional
        When True, logs detailed layer information.

    Returns
    -------
    list[LayerInfo]
        Per-layer information.

    Raises
    ------
    ValueError
        If ``onnx`` is a network and ``config`` is not provided.

    """
    if isinstance(onnx, trt.INetworkDefinition):
        if config is None:
            err_msg = "Config must be provided when onnx is a network"
            raise ValueError(err_msg)
        network = onnx
    else:
        network, _, config, _ = read_onnx(onnx)

    # Get DLA compatibility
    _, chunks = can_run_on_dla(network, config, verbose_layers=False, verbose_chunks=False)

    # Build a set of DLA-compatible layer indices
    dla_layer_indices: set[int] = set()
    for _layer_list, start, end, on_dla in chunks:
        if on_dla:
            for idx in range(start, end + 1):
                dla_layer_indices.add(idx)

    layers: list[LayerInfo] = []

    for idx in range(network.num_layers):
        trt_layer = network.get_layer(idx)

        # Get layer type as string
        layer_type = str(trt_layer.type).split(".")[-1]

        # Calculate output tensor size
        output_size = 0
        for out_idx in range(trt_layer.num_outputs):
            output = trt_layer.get_output(out_idx)
            if output is not None:
                output_size += get_tensor_size(output)

        # Calculate input tensor size
        input_size = 0
        for in_idx in range(trt_layer.num_inputs):
            inp = trt_layer.get_input(in_idx)
            if inp is not None:
                input_size += get_tensor_size(inp)

        layer = LayerInfo(
            index=idx,
            name=trt_layer.name,
            layer_type=layer_type,
            precision=trt_layer.precision,
            input_tensor_size=input_size,
            output_tensor_size=output_size,
            dla_compatible=idx in dla_layer_indices,
        )
        layers.append(layer)

        if verbose:
            LOG.info(f"Layer {idx}: {layer}")

    return layers
