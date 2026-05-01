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
