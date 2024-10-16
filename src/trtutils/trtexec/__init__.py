# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule for utilities related to the trtexec tool.

Functions
---------
build_engine
    Build an engine from an ONNX file using trtexec.
find_trtexec
    Find an instance of the trtexec binary on the system.
find_trtexec_binaries
    Find all instances of trtexec binaries on the system.
run_trtexec
    Run trtexec command.

"""

from __future__ import annotations

from ._build import build_engine
from ._find import find_trtexec, find_trtexec_binaries
from ._run import run_trtexec

__all__ = ["build_engine", "find_trtexec", "find_trtexec_binaries", "run_trtexec"]
