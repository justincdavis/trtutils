# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S404, S603, S607
from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def find_trtexec_binaries() -> list[Path]:
    """
    Find all instances of trtexec binaries on the system.

    Requires the locate command to be installed on the system.
    As such, only works on Unix-like systems.

    Returns
    -------
    list[Path]
        A list of paths to the trtexec binaries found on the system

    Raises
    ------
    RuntimeError
        If the locate command is not found on the system

    """
    try:
        output = subprocess.run(
            ["locate", "trtexec"],
            stdout=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        err_msg = "Error processing locate command. Ensure the locate command is installed on your system"
        raise RuntimeError(err_msg) from e
    text_output = output.stdout.decode("utf-8")
    text_lines: list[str] = text_output.split("\n")
    potential_paths: list[Path] = []
    for line in text_lines:
        # filter out empty lines and None
        if line is None:
            continue
        if len(line) < 1:
            continue
        # identify only lines which contain a trtexec binary
        line_path = Path(line)
        if line_path.stem == "trtexec" and not line_path.suffix and line_path.is_file():
            potential_paths.append(line_path)
    return potential_paths


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
    basic_path: Path = Path("/usr/src/tensorrt/bin/trtexec")
    if basic_path.exists():
        return basic_path

    # If the basic path is not present, use locate and parse
    potential_paths = find_trtexec_binaries()

    if len(potential_paths) == 0:
        err_msg = "trtexec binary not found on system"
        raise FileNotFoundError(err_msg)

    # identify any which are not in the /home directory
    # if no binaries in non-home directories, utilize rest of search
    non_home_paths = [line for line in potential_paths if "/home" not in str(line)]
    if len(non_home_paths) > 0:
        potential_paths = non_home_paths

    # return the shortest path (shortest is least complicated???)
    trtexec_binaries = sorted(potential_paths, key=lambda x: len(str(x.resolve())))
    return trtexec_binaries[0]
