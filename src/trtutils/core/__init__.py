# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Core utilities for TensorRT.

Functions
---------
create_engine
    Create a TensorRT engine from a serialized engine file.

"""

from __future__ import annotations

from ._engine import create_engine

__all__ = [
    "create_engine",
]
