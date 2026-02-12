# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Implementations for research papers.

Submodules
----------
:mod:`axonn`
    Implementation of the AxoNN paper for energy-aware multi-accelerator
    neural network inference optimization.

:mod:`haxconn`
    Implementation of the HaX-CoNN paper for contention-aware concurrent
    DNN execution on shared-memory heterogeneous SoCs.

"""

from __future__ import annotations

from . import axonn, haxconn

__all__ = [
    "axonn",
    "haxconn",
]
