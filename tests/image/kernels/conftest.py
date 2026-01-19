# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

import pytest

from trtutils.core import Kernel, create_stream, destroy_stream

IMG_PATH = str(Path(__file__).parent.parent.parent.parent / "data" / "horse.jpg")


def kernel_compile(kernel: tuple[Path, str]) -> None:
    stream = create_stream()
    compiled = Kernel(*kernel)
    assert compiled is not None
    destroy_stream(stream)


@pytest.fixture
def cuda_stream():
    stream = create_stream()
    yield stream
    destroy_stream(stream)
