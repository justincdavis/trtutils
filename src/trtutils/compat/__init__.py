# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Compatibility wrappers for other libraries.

Submodules
----------
:mod:`sahi`
    Compatibility wrappers for the SAHI library.

"""

from __future__ import annotations

import contextlib

__all__ = ["sahi"]

with contextlib.suppress(ImportError):
    from . import sahi

    __all__ += ["sahi"]
