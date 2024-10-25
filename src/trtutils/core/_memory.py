# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
import ctypes

import numpy as np

# suppress pycuda import error for docs build
with contextlib.suppress(Exception):
    from cuda import cudart  # type: ignore[import-untyped, import-not-found]

from ._cuda import cuda_call


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


def memcpy_host_to_device_async(
    device_ptr: int,
    host_arr: np.ndarray,
    stream: cudart.cudaStream_t,
) -> None:
    """
    Copy a numpy array to a device pointer with error checking.

    Parameters
    ----------
    device_ptr : int
        The device pointer to copy to.
    host_arr : np.ndarray
        The numpy array to copy.
    stream : cudart.cudaStream_t
        The stream to utilize.

    """
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpyAsync(
            device_ptr,
            host_arr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        ),
    )


def memcpy_device_to_host_async(
    host_arr: np.ndarray,
    device_ptr: int,
    stream: cudart.cudaStream_t,
) -> None:
    """
    Copy a device pointer to a numpy array with error checking.

    Parameters
    ----------
    host_arr : np.ndarray
        The numpy array to copy to.
    device_ptr : int
        The device pointer to copy.
    stream : cudart.cudaStream_t
        The stream to utilize.

    """
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpyAsync(
            host_arr,
            device_ptr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        ),
    )


def allocate_pinned_memory(nbytes: int, dtype: np.dtype) -> np.ndarray:
    """
    Allocate pinned (page-locked) memory on the host, required for asynchronous memory transfers.

    Parameters
    ----------
    nbytes : int
        The number of bytes to allocate.
    dtype : np.dtype
        The data type for the allocated memory.

    Returns
    -------
    np.ndarray
        A numpy array backed by pinned memory.

    """
    # Allocate pinned memory and get a pointer to it directly
    host_ptr = cuda_call(cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocDefault))

    # Create a NumPy array from the pointer
    array_type = ctypes.c_byte * nbytes
    array = np.ctypeslib.as_array(array_type.from_address(host_ptr))

    # Set the correct dtype and shape for the array
    return array.view(dtype).reshape((nbytes // dtype.itemsize,))
