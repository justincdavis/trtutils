# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S404, S603, S607
"""Run the trtexec tool from the command line."""

from __future__ import annotations

import subprocess
import sys

from ._find import find_trtexec


def main() -> None:
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


if __name__ == "__main__":
    main()
