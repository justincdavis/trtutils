# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
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

"""

from __future__ import annotations

from ._subgraph import (
    extract_subgraph,
    extract_subgraph_from_file,
    split_model,
    split_model_from_file,
)

__all__ = [
    "extract_subgraph",
    "extract_subgraph_from_file",
    "split_model",
    "split_model_from_file",
]
