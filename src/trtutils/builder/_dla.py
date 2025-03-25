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
) -> bool:
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
    bool
        Whether or not the model will all run on DLA.

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

    for idx in range(network.num_layers):
        layer = network.get_layer(idx)
        if int8:
            layer.precision = trt.DataType.INT8
        else:
            layer.precision = trt.DataType.FP16

        # check if the layer can run on DLA
        dla_valid = check_dla(layer)
        if not dla_valid:
            full_dla = False

        if verbose:
            _log.info(
                f"Layer {idx}: {layer.name}, {layer.type}, {layer.precision}, {layer.metadata}",
            )
            _log.info(f"\tDLA: {dla_valid}")

    return full_dla
