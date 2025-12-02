# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
"""
ONNX model implementations.

Attributes
----------
:attribute:`IMAGE_PREPROC_BASE` : Path
    The path to the ONNX model for image preprocessing.
:attribute:`IMAGE_PREPROC_IMAGENET` : Path
    The path to the ONNX model for ImageNet preprocessing.

"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import numpy as np

with contextlib.suppress(ImportError):
    import tensorrt as trt

from trtutils.builder._build import build_engine
from trtutils.core import cache as caching_tools

IMAGE_PREPROC_BASE: Path = Path(__file__).parent / "_onnx" / "image_preproc_base.onnx"
IMAGE_PREPROC_IMAGENET: Path = Path(__file__).parent / "_onnx" / "image_preproc_imagenet.onnx"


def build_image_preproc(
    input_shape: tuple[int, int],
    output_dtype: np.dtype,
    batch_size: int = 1,
    trt_version: str | None = None,
) -> Path:
    """
    Build a image preproc TensorRT engine.

    Parameters
    ----------
    input_shape : tuple[int, int]
        The (width, height) of the image network.
    output_dtype : np.dtype
        The datatype to return, which the image network will take as input.
    batch_size : int
        The batch size for the engine. Default is 1.
    trt_version : str | None
        The version of TensorRT to use. If none, will not be used in the cache name.

    Returns
    -------
    Path
        The path to the compiled engine.

    """
    # resolve the trt datatype nad string version
    output_dtype_str = "float32"
    trt_output_dtype = trt.DataType.FLOAT
    if output_dtype == np.float16:
        output_dtype_str = "float16"
        trt_output_dtype = trt.DataType.HALF

    # resolve the file name - includes batch size
    name = f"image_preproc_{input_shape[0]}_{input_shape[1]}_{output_dtype_str}_b{batch_size}"
    if trt_version is not None:
        name += f"_{trt_version}"

    # compile the engine
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_output = Path(tmpdir).resolve() / f"{name}.engine"
        build_engine(
            IMAGE_PREPROC_BASE,
            temp_output,
            default_device=trt.DeviceType.GPU,
            workspace=1.0,
            direct_io=True,
            input_tensor_formats=[
                ("input", trt.DataType.UINT8, trt.TensorFormat.LINEAR),
                ("scale", trt.DataType.FLOAT, trt.TensorFormat.LINEAR),
                ("offset", trt.DataType.FLOAT, trt.TensorFormat.LINEAR),
            ],
            output_tensor_formats=[("output", trt_output_dtype, trt.TensorFormat.LINEAR)],
            shapes=[
                ("input", (batch_size, input_shape[1], input_shape[0], 3)),
            ],
            fp16=True,
            cache=True,
        )

    # engine will exist or this function did not succeed, dont need to check return code
    return caching_tools.query(name)[1]


def build_image_preproc_imagenet(
    input_shape: tuple[int, int],
    output_dtype: np.dtype,
    batch_size: int = 1,
    trt_version: str | None = None,
) -> Path:
    """
    Build an ImageNet preprocessing TensorRT engine.

    Parameters
    ----------
    input_shape : tuple[int, int]
        The (width, height) of the image network.
    output_dtype : np.dtype
        The datatype to return, which the image network will take as input.
    batch_size : int
        The batch size for the engine. Default is 1.
    trt_version : str | None
        The version of TensorRT to use. If none, will not be used in the cache name.

    Returns
    -------
    Path
        The path to the compiled engine.

    """
    # resolve the trt datatype and string version
    output_dtype_str = "float32"
    trt_output_dtype = trt.DataType.FLOAT
    if output_dtype == np.float16:
        output_dtype_str = "float16"
        trt_output_dtype = trt.DataType.HALF

    # resolve the file name - includes batch size
    name = f"image_preproc_imagenet_{input_shape[0]}_{input_shape[1]}_{output_dtype_str}_b{batch_size}"
    if trt_version is not None:
        name += f"_{trt_version}"

    # compile the engine
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_output = Path(tmpdir).resolve() / f"{name}.engine"
        build_engine(
            IMAGE_PREPROC_IMAGENET,
            temp_output,
            default_device=trt.DeviceType.GPU,
            workspace=1.0,
            direct_io=True,
            input_tensor_formats=[
                ("input", trt.DataType.UINT8, trt.TensorFormat.LINEAR),
                ("mean", trt.DataType.FLOAT, trt.TensorFormat.LINEAR),
                ("std", trt.DataType.FLOAT, trt.TensorFormat.LINEAR),
            ],
            output_tensor_formats=[("output", trt_output_dtype, trt.TensorFormat.LINEAR)],
            shapes=[
                ("input", (batch_size, input_shape[1], input_shape[0], 3)),
                ("mean", (1, 3, 1, 1)),
                ("std", (1, 3, 1, 1)),
            ],
            fp16=True,
            cache=True,
        )

    # engine will exist or this function did not succeed, dont need to check return code
    return caching_tools.query(name)[1]
