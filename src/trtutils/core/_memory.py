# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from cuda import cudart  # type: ignore[import-untyped, import-not-found]

from ._cuda import cuda_call

if TYPE_CHECKING:
    import numpy as np


def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray) -> None:
    """
    Copy a numpy array to a device pointer with error checking.

    Parameters
    ----------
    device_ptr : int
        The device pointer to copy to.
    host_arr : np.ndarray
        The numpy array to copy.

    """
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            device_ptr,
            host_arr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        ),
    )


def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int) -> None:
    """
    Copy a device pointer to a numpy array with error checking.

    Parameters
    ----------
    host_arr : np.ndarray
        The numpy array to copy to.
    device_ptr : int
        The device pointer to copy.

    """
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            host_arr,
            device_ptr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        ),
    )
