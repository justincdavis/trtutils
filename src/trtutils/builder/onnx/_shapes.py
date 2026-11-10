# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Utilities for reading and modifying ONNX model input shapes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx

from trtutils._log import LOG

if TYPE_CHECKING:
    from pathlib import Path


_NCHW = ("N", 3, "H", "W")


def get_onnx_input(onnx_path: Path) -> tuple[str, tuple[int, ...]]:
    """
    Read the first input tensor name and shape from an ONNX model.

    Parameters
    ----------
    onnx_path : Path
        Path to the ONNX model file.

    Returns
    -------
    tuple[str, tuple[int, ...]]
        The input tensor name and its shape. Dynamic dimensions are replaced with 1.

    """
    model = onnx.load(str(onnx_path))
    inp = model.graph.input[0]
    name = inp.name
    dims = tuple(d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim)
    return name, dims


def get_static_imgsz(onnx_path: Path, input_name: str) -> int | None:
    """
    Read the static square image size of an NCHW ONNX input without loading weights.

    Parameters
    ----------
    onnx_path : Path
        Path to the ONNX model file.
    input_name : str
        Name of the image input tensor.

    Returns
    -------
    int | None
        The size when the input is ``(N, 3, S, S)`` with static equal H and W, else None.

    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    for inp in model.graph.input:
        if inp.name != input_name:
            continue
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        if len(dims) != len(_NCHW):
            continue
        _, channels, height, width = dims
        if channels == _NCHW[1] and height > 0 and height == width:
            return height
    return None


def get_onnx_output(onnx_path: Path) -> tuple[str, tuple[int, ...]]:
    """
    Read the first output tensor name and shape from an ONNX model.

    Parameters
    ----------
    onnx_path : Path
        Path to the ONNX model file.

    Returns
    -------
    tuple[str, tuple[int, ...]]
        The output tensor name and its shape. Dynamic dimensions are replaced with 1.

    """
    model = onnx.load(str(onnx_path))
    out = model.graph.output[0]
    name = out.name
    dims = tuple(d.dim_value if d.dim_value > 0 else 1 for d in out.type.tensor_type.shape.dim)
    return name, dims


def make_onnx_static(onnx_path: Path) -> None:
    """
    Set any dynamic dimensions in the ONNX model to 1 (batch size).

    Modifies the ONNX file in-place if any dynamic dimensions are found.

    Parameters
    ----------
    onnx_path : Path
        Path to the ONNX model file.

    """
    model = onnx.load(str(onnx_path))
    changed = False
    for inp in model.graph.input:
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value <= 0:
                dim.ClearField("dim_param")
                dim.dim_value = 1
                changed = True
    if changed:
        onnx.save(model, str(onnx_path))
        LOG.info(f"Fixed dynamic dimensions in {onnx_path}")
