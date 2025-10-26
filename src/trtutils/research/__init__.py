# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Research implementations.

Submodules
----------
axonn
    Implementation of the AxoNN paper.
flexpatch
    Implementation of the FlexPatch paper.
remix
    Implementation of the Remix paper.

"""

from __future__ import annotations

import contextlib

__all__ = []

with contextlib.suppress(ImportError):
    from . import flexpatch

    __all__ += ["flexpatch"]

with contextlib.suppress(ImportError):
    from . import remix

    __all__ += ["remix"]
