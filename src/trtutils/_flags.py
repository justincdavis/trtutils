# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
from dataclasses import dataclass


@dataclass
class _FLAGS:
    """
    Class for storing flags for trtutils.

    Attributes
    ----------
    TRT_10 : bool
        Whether or not TensorRT is version 10 or greater.
    EXEC_ASYNC_V3 : bool
        Whether or not execute_async_v3 is available
    EXEC_ASYNC_V2 : bool
        Whether or not execute_async_v2 is available
    EXEC_ASYNC_V1 : bool
        Whether or not execute_async is available.
    EXEC_V2 : bool
        Whether or not execute_v2 is available.
    EXEC_V1 : bool
        Whether or not execute_v1 is available.

    """

    TRT_10: bool = False
    EXEC_ASYNC_V3: bool = False
    EXEC_ASYNC_V2: bool = False
    EXEC_ASYNC_V1: bool = False
    EXEC_V2: bool = False
    EXEC_V1: bool = False


FLAGS = _FLAGS()


with contextlib.suppress(ImportError):
    import tensorrt as trt  # type: ignore[import-untyped, import-not-found]

    FLAGS.TRT_10 = hasattr(trt.ICudaEngine, "num_io_tensors")
    FLAGS.EXEC_ASYNC_V3 = hasattr(trt.IExecutionContext, "execute_async_v3")
    FLAGS.EXEC_ASYNC_V2 = hasattr(trt.IExecutionContext, "execute_async_v2")
    FLAGS.EXEC_ASYNC_V1 = hasattr(trt.IExecutionContext, "execute_async")
    FLAGS.EXEC_V2 = hasattr(trt.IExecutionContext, "execute_v2")
    FLAGS.EXEC_V1 = hasattr(trt.IExecutionContext, "execute")
