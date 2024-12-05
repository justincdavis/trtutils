# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: PYI041
from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

import numpy as np

with contextlib.suppress(Exception):
    from cuda import cuda  # type: ignore[import-untyped, import-not-found]

from ._cuda import cuda_call
from ._nvrtc import compile_and_load_kernel

if TYPE_CHECKING:
    from typing_extensions import Self

    with contextlib.suppress(Exception):
        from cuda import cudart  # type: ignore[import-untyped, import-not-found]

_log = logging.getLogger(__name__)


class Kernel:
    """Holds kernel coda and PTX for execution."""

    def __init__(
        self: Self,
        kernel_code: str,
        name: str,
        *,
        verbose: bool | None = None,
    ) -> None:
        """
        Create the specified kernel from the CUDA code.

        Parameters
        ----------
        kernel_code : str
            The CUDA code containing the kernel definition.
        name: str
            The name of the kernel to compile.
        verbose : bool, optional
            Whether or not to output additional information
            to stdout. If not provided, will default to overall
            engines verbose setting.

        """
        self._name = name
        self._module, self._kernel = compile_and_load_kernel(
            kernel_code,
            name,
            verbose=verbose,
        )
        self._inter_args: list[np.ndarray] | None = None

    def free(self: Self) -> None:
        """Free the memory of the loaded kernel."""
        cuda_call(cuda.cuModuleUnload(self._module))

    def __del__(self: Self) -> None:
        with contextlib.suppress(AttributeError, RuntimeError):
            self.free()

    def create_args(
        self: Self,
        *args: int | float | np.ndarray,
        verbose: bool | None = False,
    ) -> np.ndarray:
        """
        Create the argument pointer array for a CUDA kernel call.

        Is a wrapper around :func:`trtutils.core.create_kernel_args`, which
        stores the intermediate pointer results in inside of the class.
        The intermediate arrays can be cleaned up by the garbage collector
        if the kernel does not access the memory fast enough.

        Parameters
        ----------
        *args : int | float | np.ndarray
            All args to pass to the kernel as integers, floats, or pre-formed args.
            If arrays are to be passed to the kernel, they should be
            given as an integer representing the pointer returned
            from CUDA malloc.
            A preformed arg is one which is already wrapped as an np.ndarray
            with specific type.
        verbose : bool, optional
            Whether or not to output additional information about the passed args.

        Returns
        -------
        np.ndarray
            The np.ndarray of argument pointers (one pointer per arg)

        """
        ptrs, intermediate = create_kernel_args(*args, verbose=verbose)
        self._inter_args = intermediate
        return ptrs

    def __call__(
        self: Self,
        num_blocks: tuple[int, int, int],
        num_threads: tuple[int, int, int],
        stream: cudart.cudaStream_t,
        args: np.ndarray,
        *,
        verbose: bool | None = None,
    ) -> None:
        self.call(num_blocks, num_threads, stream, args, verbose=verbose)

    def call(
        self: Self,
        num_blocks: tuple[int, int, int],
        num_threads: tuple[int, int, int],
        stream: cudart.cudaStream_t,
        args: np.ndarray,
        *,
        verbose: bool | None = None,
    ) -> None:
        """
        Launch the kernel with the specified blocks, threads, and args in a stream.

        Parameters
        ----------
        num_blocks: tuple[int, int, int]
            The number of blocks to use for the kernel calls.
        num_threads: tuple[int, int, int]
            The number of threads to use for the kernel calls.
        stream: cudart.cudaStream_t
            The CUDA stream to execute the kernel in.
        args: np.ndarray
            The NumPy array containing the pointers to the arguments.
            This array should be 1D containing int64 pointers to a NumPy
            array containing each individual argument.
        verbose : bool, optional
            Whether or not to output additional information
            to stdout. If not provided, will default to overall
            engines verbose setting.

        """
        if verbose:
            _log.debug(
                f"Calling kernel: {self._name}, blocks: {num_blocks}, threads: {num_threads}, args: {args}",
            )

        launch_kernel(
            self._kernel,
            num_blocks,
            num_threads,
            stream,
            args,
        )


def launch_kernel(
    kernel: cuda.CUkernel,
    num_blocks: tuple[int, int, int],
    num_threads: tuple[int, int, int],
    stream: cudart.cudaStream_t,
    args: np.ndarray,
) -> None:
    """
    Launch a CUDA kernel with specified blocks, threads, and args in a stream.

    Parameters
    ----------
    kernel: cuda.CUKernel
        The CUDA kernel as compiled by NVRTC using the compile_kernel function.
    num_blocks: tuple[int, int, int]
        The number of blocks to use for the kernel call.
    num_threads: tuple[int, int, int]
        The number of threads to use for the kernel call.
    stream: cudart.cudaStream_t
        The CUDA stream to execute the kernel in.
    args: np.ndarray
        The NumPy array containing the pointers to the arguments.
        This array should be 1D containing int64 pointers to a NumPy
        array containing each individual argument.

    """
    cuda_call(
        cuda.cuLaunchKernel(
            kernel,
            *num_blocks,
            *num_threads,
            0,
            stream,
            args.ctypes.data,
            0,
        ),
    )


def create_kernel_args(
    *args: int | float | np.ndarray,
    verbose: bool | None = False,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Create the argument pointer array for a CUDA kernel call.

    Adapted from the workflow present in:
    https://nvidia.github.io/cuda-python/overview.html#cuda-python-workflow
    This MUST be called for each kernel call. If the args are not
    regenerated the CUDA runtime will crash.

    The intermediate argument buffers MUST be saved as variable to ensure
    the garbage collector does not delete them before use. The Kernel wrapper
    class handles this and is the recomended way to interact with kernels
    inside of trtutils.

    Parameters
    ----------
    *args : int | float | np.ndarray
        All args to pass to the kernel as integers, floats, or pre-formed args.
        If arrays are to be passed to the kernel, they should be
        given as an integer representing the pointer returned
        from CUDA malloc.
        A preformed arg is one which is already wrapped as an np.ndarray
        with specific type.
    verbose : bool, optional
        Whether or not to output additional information about the passed args.

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray]]
        The np.ndarray of argument pointers (one pointer per arg), and the allocated arrays

    Raises
    ------
    TypeError
        If the type of an argument is not integer or float

    """
    # verbose output
    if verbose:
        _log.debug(f"Converting args: {args}")

    # convert all args to np.ndarrays
    converted_args: list[np.ndarray] = []
    for arg in args:
        if isinstance(arg, int):
            converted_args.append(np.array([arg], dtype=np.uint64))
        elif isinstance(arg, float):
            converted_args.append(np.array([arg], dtype=np.float32))
        elif isinstance(arg, np.ndarray):
            converted_args.append(arg)
        else:
            err_msg = f"Unrecognized arg type for CUDA kernel: {type(arg)}"
            raise TypeError(err_msg)

        if verbose:
            last_arg = converted_args[-1]
            _log.debug(f"Converted Arg: {arg} -> Array: {last_arg} {last_arg.dtype}")

    # get a pointer to each np.ndarray and pack into new array
    ptrs: np.ndarray = np.array(
        [arg.ctypes.data for arg in converted_args],
        dtype=np.uint64,
    )

    if verbose:
        _log.debug(f"Generated pointers: {ptrs}")

    return ptrs, converted_args
