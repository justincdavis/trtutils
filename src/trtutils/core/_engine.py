# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

# suppress pycuda import error for docs build
with contextlib.suppress(Exception):
    import tensorrt as trt  # type: ignore[import-untyped, import-not-found]

from ._stream import create_stream

if TYPE_CHECKING:
    from cuda import cudart  # type: ignore[import-untyped, import-not-found]

_CONTEXT_LOCK = Lock()
_STREAM_LOCK = Lock()


def create_engine(
    engine_path: Path | str,
    logger: trt.ILogger | None = None,
    log_level: trt.ILogger.Severity = trt.Logger.WARNING,
) -> tuple[trt.ICudaEngine, trt.IExecutionContext, trt.ILogger, cudart.cudaStream_t]:
    """
    Load a serialized engine from disk.

    Parameters
    ----------
    engine_path : Path | str
        The path to the serialized engine file.
    logger : trt.ILogger, optional
        The logger to use, by default None
    log_level : trt.ILogger.Severity, optional
        The log level to use if the logger is None, by default trt.Logger.WARNING

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

    # load the logger and libnvinfer plugins
    trt_logger = logger or trt.Logger(log_level)
    trt.init_libnvinfer_plugins(trt_logger, "")

    # load the engine from file
    with Path.open(engine_path, "rb") as f, trt.Runtime(
        trt_logger,
    ) as runtime:
        if runtime is None:
            err_msg = "Failed to create TRT runtime"
            raise RuntimeError(err_msg)
        engine = runtime.deserialize_cuda_engine(f.read())

    # final check on engine
    if engine is None:
        err_msg = f"Failed to deserialize engine from {engine_path}"
        raise RuntimeError(err_msg)

    # create the execution context
    with _CONTEXT_LOCK:
        context = engine.create_execution_context()
        if context is None:
            err_msg = "Failed to create execution context"
            raise RuntimeError(err_msg)

    # create a cudart stream
    with _STREAM_LOCK:
        stream = create_stream()

    return engine, context, trt_logger, stream
