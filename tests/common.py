# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import trtutils

from .paths import ENGINE_PATH, ONNX_PATH

if TYPE_CHECKING:
    from pathlib import Path


def build_engine() -> Path:
    """
    Build a TensorRT engine from ONNX model.

    Returns
    -------
    Path
        The compiled engine.

    """
    if ENGINE_PATH.exists():
        return ENGINE_PATH

    ENGINE_PATH.parent.mkdir(parents=True, exist_ok=True)

    trtutils.builder.build_engine(
        ONNX_PATH,
        ENGINE_PATH,
        optimization_level=1,
    )

    return ENGINE_PATH
