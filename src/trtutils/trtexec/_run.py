# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S404, S603
from __future__ import annotations

import subprocess
from pathlib import Path

from ._find import find_trtexec


def run_trtexec(
    command: str,
    trtexec_path: Path | str | None = None,
) -> tuple[str, str]:
    """
    Run a command using trtexec.

    The goal of this function is make it easier to use trtexec
    within Python scripts. By returning the stdout/stderr streams
    via strings back to the Python program it can simplify
    logic or scripts which utilize trtexec.

    Parameters
    ----------
    command : str
        The command to run using trtexec
    trtexec_path : Path | str | None, optional
        The path to the trtexec binary to use.
        If None, find_trtexec will be used.

    Returns
    -------
    tuple[str, str]
        A tuple of stdout and stderr TextIOWrappers

    """
    if trtexec_path is None:
        trtexec_path = find_trtexec()
    if isinstance(trtexec_path, Path):
        trtexec_path = str(trtexec_path)
    command = f"{trtexec_path} {command}"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = ""
    if process.stdout is not None:
        stdout = process.stdout.read().decode()
    stderr = ""
    if process.stderr is not None:
        stderr = process.stderr.read().decode()
    return stdout, stderr
