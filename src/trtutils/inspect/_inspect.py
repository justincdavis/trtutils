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
from trtutils._log import LOG
from trtutils.core._engine import create_engine
from trtutils.core._stream import destroy_stream


def inspect_engine(
    engine: Path | str | trt.ICudaEngine,
    *,
    verbose: bool | None = None,
) -> tuple[
    int,
    int,
    list[tuple[str, tuple[int, ...], trt.DataType, trt.TensorFormat]],
    list[tuple[str, tuple[int, ...], trt.DataType, trt.TensorFormat]],
]:
    """
    Inspect a TensorRT engine.

    Parameters
    ----------
    engine : Path | str | trt.ICudaEngine
        Path to the TensorRT engine file or an already loaded engine
    verbose : bool | None, optional
        Whether to print verbose output, by default None

    Returns
    -------
    tuple[int, int, list[tuple[str, tuple[int, ...], trt.DataType, trt.TensorFormat]], list[tuple[str, tuple[int, ...], trt.DataType, trt.TensorFormat]]]
        The size in bytes of the engine, the max batch size, and two lists of input and output tensors

    """
    loaded = False
    if isinstance(engine, (Path, str)):
        engine, context, logger, stream = create_engine(engine)
        loaded = True

    engine_mem_size: int = 0
    if FLAGS.MEMSIZE_V2:
        engine_mem_size = engine.device_memory_size_v2
    else:
        engine_mem_size = engine.device_memory_size

    # Get all input and output tensors first
    input_tensors = []
    output_tensors = []
    num_tensors = (
        range(engine.num_io_tensors) if FLAGS.TRT_10 else range(engine.num_bindings)
    )

    for i in num_tensors:
        # check
        if FLAGS.TRT_10:
            tensor_name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(tensor_name)
            dtype = engine.get_tensor_dtype(tensor_name)
            fmt = engine.get_tensor_format(tensor_name)
            is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
        else:
            tensor_name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            fmt = engine.get_binding_format(i)
            is_input = engine.binding_is_input(i)

        # store
        if is_input:
            input_tensors.append((tensor_name, shape, dtype, fmt))
        else:
            output_tensors.append((tensor_name, shape, dtype, fmt))

    batch_size: int = 0
    try:
        batch_size = engine.max_batch_size
    except AttributeError:
        if input_tensors:
            _, shape, _, _ = input_tensors[0]
            if shape and len(shape) > 0:
                batch_size = shape[0]

    if verbose:
        LOG.info("Engine Info:")
        LOG.info(f"\tMax Batch Size: {batch_size}")
        LOG.info(f"\tNum IO Tensors: {num_tensors}")
        LOG.info(f"\tDevice Memory Size: {engine_mem_size / (1024 * 1024):.2f} MB")
        LOG.info("\tInput Tensors:")
        for name, shape, dtype, fmt in input_tensors:
            LOG.info(f"\t\t{name}: shape={shape}, dtype={dtype}, format={fmt}")
        LOG.info("\tOutput Tensors:")
        for name, shape, dtype, fmt in output_tensors:
            LOG.info(f"\t\t{name}: shape={shape}, dtype={dtype}, format={fmt}")
        LOG.info("")

    if loaded:
        del engine
        del context
        del logger
        destroy_stream(stream)

    return engine_mem_size, batch_size, input_tensors, output_tensors
