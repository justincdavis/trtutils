# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
import logging
from pathlib import Path

with contextlib.suppress(ImportError):
    import tensorrt as trt  # type: ignore[import-untyped, import-not-found]

from ._onnx import read_onnx


_log = logging.getLogger(__name__)


def eval_dla(
    onnx_path: Path | str,
    *,
    verbose: bool | None = None,
) -> bool:
    """
    Whether or not the entire model can be run on a DLA.
    
    Parameters
    ----------
    onnx_path : Path, str
        The path to the onnx file.
    verbose : bool, optional
        Whether to print verbose output, by default None
    
    Returns
    -------
    bool
        Whether or not the model will all run on DLA.
    
    """
    network, builder, config, parser, trt_logger = read_onnx(onnx_path)

    check_dla = config.can_run_on_DLA if hasattr(config, "can_run_on_DLA") else config.canRunOnDLA
    config.set_flag(trt.BuilderFlag.INT8)
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = 0

    full_dla = True

    for idx in range(network.num_layers):
        layer = network.get_layer(idx)
        layer.precision = trt.DataType.INT8
        is_dla = check_dla(layer)
        if not is_dla:
            full_dla = False

        print(dir(layer))
        
        if verbose:
            print(f"Layer {idx}: {layer.name}, {layer.type}, {layer.precision}, {layer.metadata}")
            print(f"\tDLA: {is_dla}")
        
    return full_dla
