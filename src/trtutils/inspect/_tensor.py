# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from trtutils.compat._libs import trt


def get_tensor_size(tensor: trt.ITensor) -> int:
    """
    Calculate the size of a tensor in bytes.

    Computes the total memory footprint by multiplying the number of elements
    (derived from the tensor shape) by the per-element byte size of the dtype.
    Dynamic dimensions (``-1``) are treated as ``1``.

    Parameters
    ----------
    tensor : trt.ITensor
        The TensorRT tensor.

    Returns
    -------
    int
        Size in bytes.

    """
    shape = tensor.shape
    # Handle dynamic dimensions by assuming 1
    num_elements = 1
    for dim in shape:
        num_elements *= max(1, dim)

    # Get dtype size
    dtype = tensor.dtype
    dtype_sizes = {
        trt.DataType.FLOAT: 4,
        trt.DataType.HALF: 2,
        trt.DataType.INT8: 1,
        trt.DataType.INT32: 4,
        trt.DataType.BOOL: 1,
        trt.DataType.UINT8: 1,
    }
    dtype_size = dtype_sizes.get(dtype, 4)

    return num_elements * dtype_size
