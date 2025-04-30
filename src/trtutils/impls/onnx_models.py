# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
ONNX model implementations.

Attributes
----------
:attribute:`YOLO_PREPROC_BASE` : Path
    The path to the ONNX model for YOLO preprocessing.

"""

from __future__ import annotations

from pathlib import Path

YOLO_PREPROC_BASE: Path = (
    Path(__file__).parent / "_onnx" / "yolo_preproc_base.onnx"
)
