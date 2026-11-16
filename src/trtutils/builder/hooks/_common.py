# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import numpy as np

from trtutils._flags import FLAGS
from trtutils.compat._libs import trt


def make_plugin_field(name: str, value: float | list[int] | list[float]) -> trt.PluginField:
    """
    Create a plugin field for a TensorRT plugin.

    Parameters
    ----------
    name : str
        The name of the plugin field.
    value : float | list[int] | list[float]
        The value of the plugin field.

    Returns
    -------
    trt.PluginField
        The plugin field.

    """
    np_dtype = np.dtype(np.int32)
    if isinstance(value, float):
        np_dtype = np.dtype(np.float32)
    elif isinstance(value, int):
        np_dtype = np.dtype(np.int32)
    else:
        np_dtype = np.dtype(np.float32) if isinstance(value[0], float) else np.dtype(np.int32)
    dtype = (
        trt.PluginFieldType.INT32 if np_dtype == np.dtype(np.int32) else trt.PluginFieldType.FLOAT32
    )

    value_arr = [value] if not isinstance(value, (list, tuple)) else value

    return trt.PluginField(
        name,
        np.array(value_arr, dtype=np_dtype),
        dtype,
    )


def check_dynamic_range() -> None:
    """
    Raise if the installed TensorRT has no per-tensor INT8 dynamic ranges.

    Raises
    ------
    RuntimeError
        If ``ITensor.set_dynamic_range`` is missing (TensorRT 11).

    """
    if not FLAGS.DYNAMIC_RANGE:
        err_msg = (
            "Per-tensor INT8 dynamic ranges (ITensor.set_dynamic_range) were removed in "
            "TensorRT 11. Use explicit quantization instead: quantize the ONNX with "
            "trtutils.builder.quantize (modelopt) and build with strongly_typed=True."
        )
        raise RuntimeError(err_msg)
