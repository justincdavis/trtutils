# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Run the trtexec tool from the command line.
"""
from __future__ import annotations

import sys
import subprocess


def main() -> None:
    from ._find import find_trtexec

    trtexec_path = find_trtexec()
    try:
        output = subprocess.run([str(trtexec_path), *sys.argv[1:]], stdout=sys.stdout, stderr=sys.stderr, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
