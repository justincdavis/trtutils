# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule of implementations for TRTModels.

Submodules
----------
:mod:`common`
    Implementations which are generic to underlying models.
:mod:`yolo`
    Implementations of YOLO object detectors.
:mod:`kernels`
    CUDA kernels for various functions.

"""

from __future__ import annotations

import contextlib

from . import kernels

__all__ = ["kernels"]

# import common elements
with contextlib.suppress(ImportError):
    from . import common

    __all__ = ["common"]

# import yolo models
with contextlib.suppress(ImportError):
    from . import yolo

    __all__ += ["yolo"]
