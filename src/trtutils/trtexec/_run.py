# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S404, S603
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ._find import find_trtexec

_log = logging.getLogger(__name__)


def run_trtexec(
    command: str,
    trtexec_path: Path | str | None = None,
) -> tuple[bool, str, str]:
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
    tuple[bool, str, str]
        A tuple containing the following elements:  (success, stdout, stderr)

    """
    if trtexec_path is None:
        trtexec_path = find_trtexec()
    if isinstance(trtexec_path, Path):
        trtexec_path = str(trtexec_path)
    command = f"{trtexec_path} {command}"
    com_list = [p for p in command.split(" ") if len(p) > 0]
    try:
        process = subprocess.run(
            com_list,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        err_msg = f"Error running trtexec command: {command}"
        err_msg += f"\n\tReturn value: {e.returncode}"
        _log.error(err_msg)
        return False, e.stdout.decode(), e.stderr.decode()
    stdout = ""
    if process.stdout is not None:
        stdout = process.stdout.decode()
    stderr = ""
    if process.stderr is not None:
        stderr = process.stderr.decode()
    return True, stdout, stderr
