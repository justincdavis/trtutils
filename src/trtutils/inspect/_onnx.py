# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils._log import LOG
from trtutils.builder._onnx import read_onnx
from trtutils.compat._libs import trt

if TYPE_CHECKING:
    from pathlib import Path


def inspect_onnx_layers(
    onnx: trt.INetworkDefinition | Path | str,
    *,
    verbose: bool | None = None,
) -> list[tuple[int, str, trt.ILayer, trt.DataType]]:
    """
    Inspect the layers that TensorRT would create from an ONNX model.

    Parameters
    ----------
    onnx : Path | str | trt.INetworkDefinition
        Path to the ONNX model or an already constructed TensorRT network
    verbose : bool | None, optional
        When True, logs detailed layer information at DEBUG level.

    Returns
    -------
    list[tuple[int, str, trt.ILayer, trt.DataType]]
        Per-layer information including index, name, type, and precision.

    """
    if not isinstance(onnx, trt.INetworkDefinition):
        network, _, _, _ = read_onnx(onnx)
    else:
        network = onnx

    layers_info: list[tuple[int, str, trt.ILayer, trt.DataType]] = []

    for idx in range(network.num_layers):
        layer = network.get_layer(idx)
        l_name = layer.name
        l_type = layer.type
        l_precision = layer.precision
        layers_info.append((idx, l_name, l_type, l_precision))

        if verbose:
            LOG.info(f"Layer {idx}: {l_name}, {l_type}, {l_precision}")

    return layers_info
