# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class _FLAGS:
    """
    Class for storing flags for trtutils.

    Attributes
    ----------
    TRT_10 : bool
        Whether or not TensorRT is version 10 or greater.
    NEW_CAN_RUN_ON_DLA : bool
        Whether or not TensorRT supports the new can_run_on_dla method.
    BUILD_PROGRESS : bool
        Whether or not TensorRT supports the trt.IProgressMonitor interface.
    BUILD_SERIALIZED : bool
        Whether or not TensorRT supports IBuilder.build_serialized_engine or not.
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
    NEW_CAN_RUN_ON_DLA: bool = False
    MEMSIZE_V2: bool = False
    BUILD_PROGRESS: bool = False
    BUILD_SERIALIZED: bool = False
    EXEC_ASYNC_V3: bool = False
    EXEC_ASYNC_V2: bool = False
    EXEC_ASYNC_V1: bool = False
    EXEC_V2: bool = False
    EXEC_V1: bool = False
    IS_JETSON: bool = False


FLAGS = _FLAGS()


with contextlib.suppress(ImportError):
    import tensorrt as trt

    FLAGS.TRT_10 = hasattr(trt.ICudaEngine, "num_io_tensors")
    FLAGS.NEW_CAN_RUN_ON_DLA = hasattr(trt.IBuilderConfig, "can_run_on_DLA")
    FLAGS.MEMSIZE_V2 = hasattr(trt.ICudaEngine, "device_memory_size_v2")
    FLAGS.BUILD_PROGRESS = hasattr(trt, "IProgressMonitor")
    FLAGS.BUILD_SERIALIZED = hasattr(trt.Builder, "build_serialized_network")
    FLAGS.EXEC_ASYNC_V3 = hasattr(trt.IExecutionContext, "execute_async_v3")
    FLAGS.EXEC_ASYNC_V2 = hasattr(trt.IExecutionContext, "execute_async_v2")
    FLAGS.EXEC_ASYNC_V1 = hasattr(trt.IExecutionContext, "execute_async")
    FLAGS.EXEC_V2 = hasattr(trt.IExecutionContext, "execute_v2")
    FLAGS.EXEC_V1 = hasattr(trt.IExecutionContext, "execute")
    FLAGS.IS_JETSON = Path("/etc/nv_tegra_release").exists()
