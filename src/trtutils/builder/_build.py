# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
from pathlib import Path

with contextlib.suppress(ImportError):
    import tensorrt as trt  # type: ignore[import-untyped, import-not-found]


def build_engine(
    onnx: Path | str,
    output: Path | str,
    timing_cache: Path | str = "timing.cache",
    logger: trt.ILogger | None = None,
    log_level: trt.ILogger.Severity | None = None,
    workspace: float = 4.0,
    dla_core: int | None = None,
    *,
    gpu_fallback: bool = False,
    direct_io: bool = False,
    prefer_precision_constraints: bool = False,
    reject_empty_algorithms: bool = False,
    ignore_timing_mismatch: bool = False,
    fp16: bool | None = None,
) -> None:
    """
    Build a TensorRT engine from an ONNX model.

    Parameters
    ----------
    onnx : Path, str
        The path to the onnx model.
    output : Path, str
        The location to save the TensorRT engine.
    timing_cache : Path, str, optional
        Where to store the timing cache data.
        Default is timing.cache in current working directory.
    logger : trt.ILogger, optional
        The logger to use, by default None
    log_level : trt.ILogger.Severity, optional
        The log level to use if the logger is None, by default trt.Logger.WARNING
    workspace : float
        The size of the workspace in gigabytes.
        Default is 4.0 GiB.
    dla_core : int, optional
        The DLA core to build the engine for.
        By default, None or build the engine for GPU.
    gpu_fallback : bool
        Whether or not to allow GPU fallback for unsupported layers
        when building the engine for DLA.
        By default, False
    direct_io : bool
        Use direct IO for the engine.
        By default, False
    prefer_precision_constraints : bool
        Whether or not to prefer precision constraints.
        By default, False
    reject_empty_algorithms : bool
        Whether or not to reject empty algorithms.
        By default, False
    ignore_timing_mismatch : bool
        Whether or not to allow different CUDA device generated timing
        caches to be used in the building of engines.
        By default, False
    fp16 : bool, optional
        If True, quantize the engine to FP16 precision.

    Raises
    ------
    FileNotFoundError
        If the onnx model does not exist
    IsADirectoryError
        If the onnx model path is a directory
    ValueError
        If the onnx model path does not have .onnx extension
    RuntimeError
        If the ONNX model cannot be parsed
    RuntimeError
        If the TensorRT engines fails to build

    """
    onnx_path = Path(onnx).resolve()
    output_path = Path(output).resolve()
    if not onnx_path.exists():
        err_msg = f"Could not find ONNX model at: {onnx_path}"
        raise FileNotFoundError(err_msg)
    if onnx_path.is_dir():
        err_msg = f"Path given is a directory: {onnx_path}"
        raise IsADirectoryError(err_msg)
    if onnx_path.suffix != ".onnx":
        err_msg = "File does not have .onnx extension"
        raise ValueError(err_msg)

    if log_level is None:
        log_level = trt.Logger.WARNING
    trt_logger = logger or trt.Logger(log_level)

    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()

    # setup the workspace size
    workspace *= 1 << 30
    if hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace
    else:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)

    # make network
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH),
    )

    # setup parser
    parser = trt.OnnxParser(network, trt_logger)
    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            err_msg = "Cannot parse ONNX file"
            raise RuntimeError(err_msg)

    # create profile and config
    profile = builder.create_optimization_profile()
    config.add_optimization_profile(profile)

    # handle some flags
    if prefer_precision_constraints:
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    if direct_io:
        config.set_flag(trt.BuilderFlag.DIRECT_IO)
    if reject_empty_algorithms:
        config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)

    # handle DLA assignment
    if dla_core is not None:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla_core
    if gpu_fallback:
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    # load/setup the timing cache
    timing_cache_path = Path(timing_cache).resolve()
    buffer = b""
    if timing_cache_path.exists():
        with timing_cache_path.open("rb") as timing_cache_file:
            buffer = timing_cache_file.read()
    t_cache = config.create_timing_cache(buffer)
    config.set_timing_cache(t_cache, ignore_mismatch=ignore_timing_mismatch)

    # setup the precision sets
    if fp16:
        if not builder.platform_has_fast_fp16:
            trt_logger.warning("Platform does not have native fast FP16.")
        config.set_flag(trt.BuilderFlag.FP16)

    # build the engine
    if hasattr(builder, "build_serialized_network"):
        engine_bytes = builder.build_serialized_network(network, config)
    else:
        engine_bytes = builder.build_engine(network, config)

    # save the timing cache
    post_t_cache = config.get_timing_cache()
    with timing_cache_path.open("wb") as f:
        f.write(memoryview(post_t_cache.serialize()))

    if engine_bytes is None:
        err_msg = "Failed to build engine."
        raise RuntimeError(err_msg)

    with output_path.open("wb") as f:
        f.write(engine_bytes)
