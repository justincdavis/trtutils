# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule for downloading and converting models to ONNX.

Functions
---------
:func:`download`
    Download a model by name and save to a location.
:func:`download_model`
    Lower-level function for downloading and converting a model to ONNX.

"""

from __future__ import annotations

from ._download import download, download_model

__all__ = [
    "download",
    "download_model",
]
