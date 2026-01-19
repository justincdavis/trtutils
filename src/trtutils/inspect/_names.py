# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from pathlib import Path

from trtutils._engine import TRTEngine


def get_engine_names(
    engine: TRTEngine | Path | str,
) -> tuple[list[str], list[str]]:
    """
    Get the input/output names of a TensorRT engine in order.

    Parameters
    ----------
    engine : Path | str | trt.ICudaEngine
        Path to the TensorRT engine file or an already loaded engine

    Returns
    -------
    tuple[list[str], list[str]]
        The input and output tensors in order of enumeration.

    """
    loaded = False
    if isinstance(engine, (Path, str)):
        engine = TRTEngine(engine, warmup=False)
        loaded = True

    input_names, output_names = engine.input_names, engine.output_names

    if loaded:
        del engine

    return input_names, output_names
