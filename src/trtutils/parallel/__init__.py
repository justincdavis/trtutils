# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Parallel implementations of TensorRT engines and models.

Submodules
----------
:mod:`image`
    Parallel implementations of image models.

Classes
-------
:class:`QueuedTRTEngine`
    A class for running a TRTEngine in a separate thread asynchronously.
:class:`ParallelTRTEngines`
    A class for running many TRTEngines in parallel.

"""

from __future__ import annotations

from . import image
from ._parallel_engines import ParallelTRTEngines
from ._queued_engine import QueuedTRTEngine

__all__ = [
    "ParallelTRTEngines",
    "QueuedTRTEngine",
    "image",
]
