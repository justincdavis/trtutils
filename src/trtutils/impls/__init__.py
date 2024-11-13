# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule of implementations for TRTModels.

Submodules
----------
common
    Implementations which are generic to underlying models.
yolo
    Implementations of YOLO object detectors.

"""

from __future__ import annotations

import contextlib

__all__ = []

# import common elements
with contextlib.suppress(ImportError):
    from . import common

    __all__ = ["common"]

# import yolo models
with contextlib.suppress(ImportError):
    from . import yolo

    __all__ += ["yolo"]
