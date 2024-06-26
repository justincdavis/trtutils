# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S404
from __future__ import annotations

import subprocess
from pathlib import Path


def find_trtexec() -> Path:
    """
    Find an instance of the trtexec binary on the system.

    Returns
    -------
    Path
        The path to the trtexec binary

    """
    basic_path: Path = Path("/usr/src/tensorrt/bin/trtexec")
    if basic_path.exists():
        return basic_path

    # If the basic path is not present, use locate and parse
    output = subprocess.run(["locate", "trtexec"], stdout=subprocess.PIPE, check=False)
    text_output = output.stdout.decode("utf-8")

    print(text_output)
