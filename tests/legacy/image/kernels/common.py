# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

from trtutils.core import Kernel, create_stream, destroy_stream

IMG_PATH = str(Path(__file__).parent.parent.parent.parent / "data" / "horse.jpg")


def kernel_compile(kernel: tuple[Path, str]) -> None:
    """
    Test if a kernel will compile.

    Parameters
    ----------
    kernel : tuple[Path, str]
        The kernel info

    """
    stream = create_stream()
    compiled = Kernel(kernel[0], kernel[1])
    assert compiled is not None
    destroy_stream(stream)
