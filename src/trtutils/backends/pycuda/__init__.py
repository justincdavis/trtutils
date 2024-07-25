# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
PyCUDA backend for TRTEngine.

This module provides the PyCUDA backend for the TRTEngine class.

Classes
-------
TRTEngine
    A class for running inference on a TensorRT engine.

"""

from __future__ import annotations

from ._engine import TRTEngine

__all__ = [
    "TRTEngine",
]
