# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S404, S603, S607
from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

from ._find import find_trtexec

if TYPE_CHECKING:
    from types import SimpleNamespace


def cli_trtexec(_: SimpleNamespace | None = None) -> None:
    """Run trtexec with given arguments as if using trtexec directly."""
    trtexec_path = find_trtexec()
    try:
        subprocess.run(
            [str(trtexec_path), *sys.argv[1:]],
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
