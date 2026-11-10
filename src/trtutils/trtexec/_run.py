# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S603
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from trtutils._log import LOG

from ._find import find_trtexec

if TYPE_CHECKING:
    from collections.abc import Sequence


def run_trtexec(
    command: str | Sequence[str],
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
    command : str | Sequence[str]
        The arguments to run using trtexec. A string is split with shlex;
        a sequence is used as-is (safe for paths containing spaces).
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
    args = shlex.split(command) if isinstance(command, str) else list(command)
    com_list = [str(trtexec_path), *args]
    try:
        process = subprocess.run(
            com_list,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        err_msg = f"Error running trtexec command: {shlex.join(com_list)}"
        err_msg += f"\n\tReturn value: {e.returncode}"
        LOG.error(err_msg)
        return False, e.stdout.decode(), e.stderr.decode()
    stdout = ""
    if process.stdout is not None:
        stdout = process.stdout.decode()
    stderr = ""
    if process.stderr is not None:
        stderr = process.stderr.decode()
    return True, stdout, stderr
