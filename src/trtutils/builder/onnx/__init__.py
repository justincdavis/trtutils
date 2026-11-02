# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Some basic utilties for using ONNX models.

Functions
---------
:func:`extract_subgraph`
    Extract a subgraph from an ONNX model using layer indices.
:func:`extract_subgraph_from_file`
    Extract a subgraph from an ONNX model file and save it to a new file.
:func:`split_model`
    Split an ONNX model into sequential pipeline subgraphs.
:func:`split_model_from_file`
    Split an ONNX model file into sequential pipeline subgraphs.
:func:`get_onnx_input`
    Read the first input tensor name and shape from an ONNX model.
:func:`make_onnx_static`
    Set any dynamic dimensions in the ONNX model to 1 (batch size).

"""

from __future__ import annotations

from ._shapes import get_onnx_input, make_onnx_static
from ._subgraph import (
    extract_subgraph,
    extract_subgraph_from_file,
    split_model,
    split_model_from_file,
)

__all__ = [
    "extract_subgraph",
    "extract_subgraph_from_file",
    "get_onnx_input",
    "make_onnx_static",
    "split_model",
    "split_model_from_file",
]
