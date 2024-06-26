# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S404
from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def find_trtexec() -> Path:
    """
    Find an instance of the trtexec binary on the system.

    Returns
    -------
    Path
        The path to the trtexec binary

    """
    # basic_path: Path = Path("/usr/src/tensorrt/bin/trtexec")
    # if basic_path.exists():
    #     return basic_path

    # If the basic path is not present, use locate and parse
    try:
        output = subprocess.run(["locate", "trtexec"], stdout=subprocess.PIPE, check=False)
    except subprocess.CalledProcessError as e:
        err_msg = "Error processing locate command. Ensure the locate command is installed on your system"
        raise RuntimeError(err_msg) from e
    text_output = output.stdout.decode("utf-8")
    text_lines: list[str] = text_output.split("\n")
    # filter out empty lines and None
    text_lines = [line for line in text_lines if line and len(line) > 0]
    # identify only lines which contain a trtexec binary
    potential_lines: list[str] = []
    for line in text_lines:
        line_path = Path(line)
        if line_path.stem == "trtexec":
            potential_lines.append(line_path)
    
    print(potential_lines)
