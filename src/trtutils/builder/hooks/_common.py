# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib

import numpy as np

with contextlib.suppress(ImportError):
    import tensorrt as trt


def make_plugin_field(
    name: str, value: float | list[int] | list[float]
) -> trt.PluginField:
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
    np_dtype: np.dtype = np.int32  # type: ignore[assignment]
    if isinstance(value, float):
        np_dtype = np.float32  # type: ignore[assignment]
    elif isinstance(value, int):
        np_dtype = np.int32
    else:
        np_dtype = np.float32 if isinstance(value[0], float) else np.int32  # type: ignore[assignment]
    dtype = (
        trt.PluginFieldType.INT32
        if np_dtype == np.int32
        else trt.PluginFieldType.FLOAT32
    )

    value_arr = [value] if not isinstance(value, (list, tuple)) else value

    return trt.PluginField(
        name,
        np.array(value_arr, dtype=np_dtype),
        dtype,
    )
