# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

with contextlib.suppress(Exception):
    import tensorrt as trt  # type: ignore[import-untyped, import-not-found]

from trtutils._log import LOG

from ._stream import create_stream

if TYPE_CHECKING:
    try:
        import cuda.bindings.cudart as cudart  # type: ignore[import-untyped, import-not-found]
    except (ImportError, ModuleNotFoundError):
        from cuda import cudart  # type: ignore[import-untyped, import-not-found]


def create_engine(
    engine_path: Path | str,
    *,
    no_warn: bool | None = None,
) -> tuple[trt.ICudaEngine, trt.IExecutionContext, trt.ILogger, cudart.cudaStream_t]:
    """
    Load a serialized engine from disk.

    Parameters
    ----------
    engine_path : Path | str
        The path to the serialized engine file.
    no_warn : bool | None, optional
        If True, suppresses warnings from TensorRT. Default is None.

    Returns
    -------
    tuple[trt.ICudaEngine, trt.IExecutionContext, trt.ILogger]
        The deserialized engine, execution context, and logger used.
        Logger returned is the same as the input logger if not None.

    Raises
    ------
    FileNotFoundError
        If the engine file is not found.
    RuntimeError
        If the TRT runtime could not be created.
        If the engine could not be deserialized.
        If the execution context could not be created.

    """
    engine_path = Path(engine_path) if isinstance(engine_path, str) else engine_path

    if not engine_path.exists():
        err_msg = f"Engine file not found: {engine_path}"
        raise FileNotFoundError(err_msg)

    # load the engine from file
    runtime = trt.Runtime(LOG)
    with Path.open(engine_path, "rb") as f:
        if runtime is None:
            err_msg = "Failed to create TRT runtime"
            raise RuntimeError(err_msg)
        if no_warn:
            with LOG.suppress():
                engine = runtime.deserialize_cuda_engine(f.read())
        else:
            engine = runtime.deserialize_cuda_engine(f.read())

    # final check on engine
    if engine is None:
        err_msg = f"Failed to deserialize engine from {engine_path}"
        raise RuntimeError(err_msg)

    # create the execution context
    context = engine.create_execution_context()
    if context is None:
        err_msg = "Failed to create execution context"
        raise RuntimeError(err_msg)

    # create a cudart stream
    stream = create_stream()

    return engine, context, LOG, stream
