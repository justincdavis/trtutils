# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
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
:func:`get_supported_models`
    Return a list of supported model names.

Note:
----
All models downloaded through this module may have license restrictions.
Users must accept the license terms before downloading by using the accept parameter
or by responding to the interactive prompt.

"""

from __future__ import annotations

from ._download import download, download_model, get_supported_models, load_model_configs

__all__ = [
    "download",
    "download_model",
    "get_supported_models",
    "load_model_configs",
]
