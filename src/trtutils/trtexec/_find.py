# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def find_trtexec() -> Path:
    """
    Find an instance of the trtexec binary on the system.

    Requires the locate command to be installed on the system.
    As such, only works on Unix-like systems.

    Returns
    -------
    Path
        The path to the trtexec binary

    Raises
    ------
    FileNotFoundError
        If the trtexec binary is not found on the system

    """
    possible_dirs: list[Path] = [
        Path("/usr/src/tensorrt/bin"),
        Path("/usr/local/TensorRT/bin"),
        Path("/usr/local/tensorrt/bin"),
    ]
    for root in possible_dirs:
        if not root.exists():
            continue
        if not root.is_dir():
            continue
        trtexec_path = root / "trtexec"
        if trtexec_path.exists():
            return trtexec_path

    # could not find trtexec in the default locations
    err_msg = "trtexec binary not found on system"
    raise FileNotFoundError(err_msg)
