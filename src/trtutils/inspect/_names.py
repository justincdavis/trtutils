# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
from pathlib import Path

with contextlib.suppress(ImportError):
    import tensorrt as trt

from trtutils._flags import FLAGS
from trtutils.core._engine import create_engine
from trtutils.core._stream import destroy_stream


def get_engine_names(
    engine: Path | str | trt.ICudaEngine,
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
        engine, context, logger, stream = create_engine(engine)
        loaded = True

    input_names: list[str] = []
    output_names: list[str] = []
    num_tensors = (
        range(engine.num_io_tensors) if FLAGS.TRT_10 else range(engine.num_bindings)
    )

    for i in num_tensors:
        # get the tensor name in-order
        if FLAGS.TRT_10:
            tensor_name = engine.get_tensor_name(i)
            is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
        else:
            tensor_name = engine.get_binding_name(i)
            is_input = engine.binding_is_input(i)

        # store
        if is_input:
            input_names.append(tensor_name)
        else:
            output_names.append(tensor_name)

    if loaded:
        del engine
        del context
        del logger
        destroy_stream(stream)

    return input_names, output_names
