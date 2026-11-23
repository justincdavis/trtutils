# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Person re-identification model implementations."""

from __future__ import annotations

from ._fastreid import FastReID
from ._osnet import OSNet

__all__ = [
    "FastReID",
    "OSNet",
]
