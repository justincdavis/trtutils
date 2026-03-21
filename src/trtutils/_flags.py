# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from trtutils.compat._libs import trt


@dataclass
class Flags:
    """
    Class for storing flags for trtutils.

    Attributes
    ----------
    CUDA_VERSION : tuple[int, int]
        The CUDA major and minor version as a tuple.
    CUDA_11 : bool
        Whether cuda-python major version is 11.
    CUDA_12 : bool
        Whether cuda-python major version is 12.
    CUDA_13 : bool
        Whether cuda-python major version is 13.
    TRT_10 : bool
        Whether or not TensorRT is version 10 or greater.
    TRT_HAS_UINT8 : bool
        Whether or not TensorRT suports UINT8 datatype.
    TRT_HAS_INT64 : bool
        Whether or not TensorRT supports INT64 datatype.
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
    TRT_VERSION : tuple[int, int]
        The TensorRT major and minor version as a tuple.
    IS_JETSON : bool
        Whether or not the system is a Jetson system
    HAS_DLA : bool
        Whether or not DLA hardware is available on the system.
    NUM_DLA_CORES : int
        The number of DLA cores available on the system. 0 if none.
    JIT : bool
        Whether or not to use jit.
    FOUND_NUMBA : bool
        Whether or not a Numba installation was found.
    WARNED_NUMBA_NOT_FOUND : bool
        Whether or not the user has been warned that Numba was
        not found when calling enable_jit.
    NVTX_ENABLED : bool
        Whether or not NVTX profiling is enabled.

    """

    # CUDA flags
    CUDA_VERSION: tuple[int, int] = (0, 0)
    CUDA_11: bool = False
    CUDA_12: bool = False
    CUDA_13: bool = False

    # TensorRT flags
    TRT_VERSION: tuple[int, int] = (0, 0)
    TRT_10: bool = False
    TRT_HAS_UINT8: bool = False
    TRT_HAS_INT64: bool = False
    NEW_CAN_RUN_ON_DLA: bool = False
    MEMSIZE_V2: bool = False
    BUILD_PROGRESS: bool = False
    BUILD_SERIALIZED: bool = False
    EXEC_ASYNC_V3: bool = False
    EXEC_ASYNC_V2: bool = False
    EXEC_ASYNC_V1: bool = False
    EXEC_V2: bool = False
    EXEC_V1: bool = False

    # System flags
    IS_JETSON: bool = False
    HAS_DLA: bool = False
    NUM_DLA_CORES: int = 0

    # Internal flags
    JIT: bool = False
    FOUND_NUMBA: bool = False
    WARNED_NUMBA_NOT_FOUND: bool = False
    NVTX_ENABLED: bool = False


def _get_version(package: str) -> tuple[int, int]:
    try:
        major, minor = [int(x) for x in version(package).split(".")[:2]]
    except PackageNotFoundError:
        major, minor = 0, 0
    return (major, minor)


def _detect_dla_cores() -> int:
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
    except (AttributeError, RuntimeError):
        return 0
    else:
        return runtime.num_DLA_cores


FLAGS = Flags()

# Set CUDA flags
FLAGS.CUDA_VERSION = _get_version("cuda-python")
FLAGS.CUDA_11 = FLAGS.CUDA_VERSION[0] == 11  # noqa: PLR2004
FLAGS.CUDA_12 = FLAGS.CUDA_VERSION[0] == 12  # noqa: PLR2004
FLAGS.CUDA_13 = FLAGS.CUDA_VERSION[0] == 13  # noqa: PLR2004

# Set TensorRT flags
FLAGS.TRT_VERSION = _get_version(f"tensorrt_cu{FLAGS.CUDA_VERSION[0]}")
FLAGS.TRT_10 = hasattr(trt.ICudaEngine, "num_io_tensors")
FLAGS.TRT_HAS_UINT8 = hasattr(trt.DataType, "UINT8")
FLAGS.TRT_HAS_INT64 = hasattr(trt.DataType, "INT64")
FLAGS.NEW_CAN_RUN_ON_DLA = hasattr(trt.IBuilderConfig, "can_run_on_DLA")
FLAGS.MEMSIZE_V2 = hasattr(trt.ICudaEngine, "device_memory_size_v2")
FLAGS.BUILD_PROGRESS = hasattr(trt, "IProgressMonitor")
FLAGS.BUILD_SERIALIZED = hasattr(trt.Builder, "build_serialized_network")
FLAGS.EXEC_ASYNC_V3 = hasattr(trt.IExecutionContext, "execute_async_v3")
FLAGS.EXEC_ASYNC_V2 = hasattr(trt.IExecutionContext, "execute_async_v2")
FLAGS.EXEC_ASYNC_V1 = hasattr(trt.IExecutionContext, "execute_async")
FLAGS.EXEC_V2 = hasattr(trt.IExecutionContext, "execute_v2")
FLAGS.EXEC_V1 = hasattr(trt.IExecutionContext, "execute")

# Set system flags
FLAGS.IS_JETSON = Path("/etc/nv_tegra_release").exists()

# Set DLA flags
FLAGS.NUM_DLA_CORES = _detect_dla_cores()
FLAGS.HAS_DLA = FLAGS.NUM_DLA_CORES > 0
