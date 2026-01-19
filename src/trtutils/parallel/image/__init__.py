# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Parallel implementations of image models.

Classes
-------
:class:`ParallelDetector`
    Parallel implementation of Detector.
:class:`EngineInfo`
    Dataclass for specifying engine information for ParallelDetector.

"""

from __future__ import annotations

from ._detector import EngineInfo, ParallelDetector

__all__ = ["EngineInfo", "ParallelDetector"]
