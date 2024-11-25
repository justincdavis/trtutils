# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

with contextlib.suppress(Exception):
    from cuda import cuda  # type: ignore[import-untyped, import-not-found]

from ._cuda import cuda_call
from ._nvrtc import compile_and_load_kernel

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self

    with contextlib.suppress(Exception):
        from cuda import cudart  # type: ignore[import-untyped, import-not-found]


class Kernel:
    """Holds kernel coda and PTX for execution."""

    def __init__(
        self: Self,
        kernel_code: str,
        name: str,
        num_blocks: tuple[int, int, int],
        num_threads: tuple[int, int, int],
        stream: cudart.cudaStream_t,
    ) -> None:
        """
        Create the specified kernel from the CUDA code.

        Parameters
        ----------
        kernel_code : str
            The CUDA code containing the kernel definition.
        name: str
            The name of the kernel to compile.
        num_blocks: tuple[int, int, int]
            The number of blocks to use for the kernel calls.
        num_threads: tuple[int, int, int]
            The number of threads to use for the kernel calls.
        stream: cudart.cudaStream_t
            The CUDA stream to execute the kernel in.

        """
        self._name = name
        self._module, self._kernel = compile_and_load_kernel(kernel_code, name)
        self._blocks = num_blocks
        self._threads = num_threads
        self._stream = stream

    def __del__(self: Self) -> None:
        with contextlib.suppress(AttributeError, RuntimeError):
            cuda_call(cuda.cuModuleUnload(self._module))

    def __call__(
        self: Self,
        args: np.ndarray,
    ) -> None:
        self.call(args)

    def call(
        self: Self,
        args: np.ndarray,
    ) -> None:
        """
        Launch the kernel with the specified blocks, threads, and args in a stream.

        Parameters
        ----------
        args: np.ndarray
            The NumPy array containing the pointers to the arguments.
            This array should be 1D containing int64 pointers to a NumPy
            array containing each individual argument.

        """
        launch_kernel(
            self._kernel,
            self._blocks,
            self._threads,
            self._stream,
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
