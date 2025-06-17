# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Flags:
    """
    Class for storing flags for trtutils.

    Attributes
    ----------
    TRT_10 : bool
        Whether or not TensorRT is version 10 or greater.
    TRT_HAS_UINT8 : bool
        Whether or not TensorRT suports UINT8 datatype.
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
    IS_JETSON : bool
        Whether or not the system is a Jetson system
    JIT : bool
        Whether or not to use jit.
    FOUND_NUMBA : bool
        Whether or not a Numba installation was found.
    WARNED_NUMBA_NOT_FOUND : bool
        Whether or not the user has been warned that Numba was
        not found when calling enable_jit.

    """

    # TensorRT and CUDA flags
    TRT_10: bool = False
    TRT_HAS_UINT8: bool = False
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

    # Internal flags
    JIT: bool = False
    FOUND_NUMBA: bool = False
    WARNED_NUMBA_NOT_FOUND: bool = False


FLAGS = Flags()


with contextlib.suppress(ImportError):
    import tensorrt as trt

    FLAGS.TRT_10 = hasattr(trt.ICudaEngine, "num_io_tensors")
    FLAGS.TRT_HAS_UINT8 = hasattr(trt.DataType, "UINT8")
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
