# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from trtutils.core import Kernel, create_stream, destroy_stream

if TYPE_CHECKING:
    from collections.abc import Generator

IMG_PATH = str(Path(__file__).parent.parent.parent.parent / "data" / "horse.jpg")


def kernel_compile(kernel: tuple[Path, str]) -> None:
    """Compile a kernel to validate NVRTC toolchain."""
    stream = create_stream()
    compiled = Kernel(*kernel)
    assert compiled is not None
    destroy_stream(stream)


@pytest.fixture
def cuda_stream() -> Generator[object, None, None]:
    """
    Provide a CUDA stream for kernel tests.

    Yields
    ------
    object
        A CUDA stream handle.

    """
    stream = create_stream()
    yield stream
    destroy_stream(stream)
