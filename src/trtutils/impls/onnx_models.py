# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
"""
ONNX model implementations.

Attributes
----------
:attribute:`YOLO_PREPROC_BASE` : Path
    The path to the ONNX model for YOLO preprocessing.

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

YOLO_PREPROC_BASE: Path = Path(__file__).parent / "_onnx" / "yolo_preproc_base.onnx"


def build_yolo_preproc(input_shape: tuple[int, int], output_dtype: np.dtype) -> Path:
    """
    Build a YOLO preproc TensorRT engine.

    Parameters
    ----------
    input_shape : tuple[int, int]
        The (width, height) of the YOLO network.
    output_dtype : np.dtype
        The datatype to return, which the YOLO network will take as input.

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

    # resolve the file name - only depends on the input size, scale/offset passed in
    name = f"yolo_preproc_{input_shape[0]}_{input_shape[1]}_{output_dtype_str}"

    # compile the engine
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_output = Path(tmpdir).resolve() / f"{name}.engine"
        build_engine(
            YOLO_PREPROC_BASE,
            temp_output,
            default_device=trt.DeviceType.GPU,
            workspace=1.0,
            direct_io=True,
            input_tensor_formats=[
                ("input", trt.DataType.UINT8, trt.TensorFormat.LINEAR),
                ("scale", trt.DataType.FLOAT, trt.TensorFormat.LINEAR),
                ("offset", trt.DataType.FLOAT, trt.TensorFormat.LINEAR),
            ],
            output_tensor_formats=[
                ("output", trt_output_dtype, trt.TensorFormat.LINEAR)
            ],
            shapes=[
                ("input", (input_shape[1], input_shape[0], 3)),
            ],
            fp16=True,
            cache=True,
        )

    # engine will exist or this function did not succeed, dont need to check return code
    return caching_tools.query_cache(name)[1]
