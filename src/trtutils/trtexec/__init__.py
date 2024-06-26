# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule for utilities related to the trtexec tool.

Functions
---------
find_trtexec
    Find an instance of the trtexec binary on the system.
find_trtexec_binaries
    Find all instances of trtexec binaries on the system.

"""

from __future__ import annotations

from ._find import find_trtexec, find_trtexec_binaries

__all__ = ["find_trtexec", "find_trtexec_binaries"]
