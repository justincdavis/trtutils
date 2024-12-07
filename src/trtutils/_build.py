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
    logger: trt.ILogger | None = None,
    log_level: trt.ILogger.Severity | None = None,
    *,
    fp16: bool | None = None,
    int8: bool | None = None,
) -> None:
    """
    Build a TensorRT engine from an ONNX model.
    
    Parameters
    ----------
    onnx : Path, str
        The path to the onnx model.
    output : Path, str
        The location to save the TensorRT engine.
    logger : trt.ILogger, optional
        The logger to use, by default None
    log_level : trt.ILogger.Severity, optional
        The log level to use if the logger is None, by default trt.Logger.WARNING
    fp16 : bool, optional
        If True, quantize the engine to FP16 precision.
    int8 : bool, optional
        If True, quantize the engine to INT8 preicison.

    Raises
    ------
    FileNotFoundError
        If the onnx model does not exist

    """
    onnx_path = Path(onnx)
    output_path = Path(output)
    if not onnx_path.exists():
        err_msg = f"Could not find ONNX model at: {onnx_path}"
        raise FileNotFoundError(err_msg)
    if onnx_path.is_dir():
        err_msg = f"Path given is a directory: {onnx_path}"
        raise IsADirectoryError(err_msg)
    if onnx_path.suffix != ".onnx":
        err_msg = f"File does not have .onnx extension"
        raise ValueError(err_msg)
    
    if log_level is None:
        log_level = trt.Logger.WARNING
    trt_logger = logger or trt.Logger(log_level)
    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()
    workspace = 0
