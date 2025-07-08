# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Parallel implementations of image models.

Classes
-------
:class:`ParallelDetector`
    Parallel implementation of Detector.

"""

from __future__ import annotations

from ._detector import ParallelDetector

__all__ = ["ParallelDetector"]
