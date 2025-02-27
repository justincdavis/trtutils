# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S404, S603, S607
from __future__ import annotations

import subprocess
import sys

from ._find import find_trtexec


def cli_trtexec(args: list[str] | None = None) -> None:
    """Run trtexec with given arguments as if using trtexec directly."""
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
