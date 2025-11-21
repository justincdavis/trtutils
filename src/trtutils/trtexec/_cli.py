# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S404, S603
from __future__ import annotations

import subprocess
import sys

from ._find import find_trtexec


def cli_trtexec(args: list[str] | None = None) -> None:
    """
    Run trtexec with given arguments as if using trtexec directly.

    Finds the trtexec binary on the system and executes it with the provided
    command-line arguments. This function acts as a pass-through wrapper to
    the native trtexec tool, preserving all its functionality.

    Parameters
    ----------
    args : list[str] | None, optional
        Command-line arguments to pass to trtexec. If None, uses arguments
        from sys.argv. Default is None.

    Examples
    --------
    Run trtexec to build an engine:

        >>> cli_trtexec(["--onnx=model.onnx", "--saveEngine=model.engine"])

    Convert FP16 with dynamic shapes:

        >>> cli_trtexec([
        ...     "--onnx=model.onnx",
        ...     "--saveEngine=model.engine",
        ...     "--fp16",
        ...     "--minShapes=input:1x3x224x224",
        ...     "--optShapes=input:4x3x224x224",
        ...     "--maxShapes=input:8x3x224x224"
        ... ])

    """
    trtexec_path = find_trtexec()
    if args is not None:
        sys.argv = ["trtexec", *args]
    try:
        subprocess.run(
            [str(trtexec_path), *sys.argv[1:]],
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
