# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):
    import tensorrt as trt  # type: ignore[import-untyped, import-not-found]

from ._onnx import read_onnx

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_log = logging.getLogger(__name__)


def can_run_on_dla(
    onnx_path: Path | str,
    *,
    int8: bool | None = None,
    fp16: bool | None = None,
    verbose: bool | None = None,
) -> tuple[bool, list[tuple[list[trt.ILayer], int, int, bool]]]:
    """
    Whether or not the entire model can be run on a DLA.

    Parameters
    ----------
    onnx_path : Path, str
        The path to the onnx file.
    int8 : bool, optional
        Whether to use INT8 precision, by default None
        If neither int8 or fp16 are provided, fp16 will be used.
    fp16 : bool, optional
        Whether to use FP16 precision, by default None
        If neither int8 or fp16 are provided, fp16 will be used.
    verbose : bool, optional
        Whether to print verbose output, by default None

    Returns
    -------
    tuple[bool, list[tuple[list[trt.ILayer], int, int, bool]]]
        Whether or not the model will all run on DLA and each block of layers.
        Where each block can run on a single device, DLA or GPU.

    """
    network, _, config, _, _ = read_onnx(onnx_path)

    check_dla: Callable[[trt.ILayer], bool] = (
        config.can_run_on_DLA
        if hasattr(config, "can_run_on_DLA")
        else config.canRunOnDLA
    )

    # handle precision setup
    if int8 is None and fp16 is None:
        fp16 = True

    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
    elif fp16:
        config.set_flag(trt.BuilderFlag.FP16)

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
        if int8:
            layer.precision = trt.DataType.INT8
        else:
            layer.precision = trt.DataType.HALF

        # check if the layer can run on DLA
        dla_valid = check_dla(layer)
        if not dla_valid:
            full_dla = False

        # handle chunk storage
        if dla_valid != last_layer_dla:
            if len(curr_layers) > 0:
                chunks.append(
                    (curr_layers, curr_start, idx - 1, last_layer_dla)
                )
                curr_layers = [layer]
            curr_start = idx
        else:
            curr_layers.append(layer)

        last_layer_dla = dla_valid

        if verbose:
            _log.info(
                f"Layer {idx}: {layer.name}, {layer.type}, {layer.precision}, {layer.metadata}",
            )
            _log.info(f"\tDLA: {dla_valid}")

    # handle final chunk
    chunks.append(
        (curr_layers, curr_start, network.num_layers - 1, last_layer_dla)
    )

    if verbose:
        _log.info(f"Found {len(chunks)} Chunks of Layers")
        for i, (layers, start, end, on_dla) in enumerate(chunks):
            _log.info(f"\tChunk {i}: [{start} - {end}], {len(layers)} layers, {'DLA' if on_dla else 'GPU'}")

    return full_dla, chunks
