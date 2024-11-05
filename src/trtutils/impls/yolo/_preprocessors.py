# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
import logging
import math
from typing import TYPE_CHECKING

import numpy as np
from cv2ext.image import letterbox

with contextlib.suppress(ImportError):
    from cuda import cuda, cudart  # type: ignore[import-untyped, import-not-found]

from trtutils.core._bindings import Binding, create_binding
from trtutils.core._cuda import cuda_call
from trtutils.core._memory import (
    memcpy_device_to_host_async,
    memcpy_host_to_device_async,
)
from trtutils.core._nvrtc import compile_and_load_kernel
from trtutils.core._stream import create_stream, destroy_stream, stream_synchronize

from ._kernels import SCALE_SWAP_TRANSPOSE_KERNEL_CODE
from ._process import preprocess

if TYPE_CHECKING:
    from typing_extensions import Self

_log = logging.getLogger(__name__)


class CPUPreprocessor:
    def __init__(
        self: Self,
        output_shape: tuple[int, int],
        output_range: tuple[float, float],
        dtype: np.dtype,
    ) -> None:
        _log.debug(
            f"Creating CPU preprocessor: {output_shape}, {output_range}, {dtype}",
        )
        self._o_shape = output_shape
        self._o_range = output_range
        self._o_dtype = dtype

    def __call__(
        self: Self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        return preprocess(image, self._o_shape, self._o_dtype, self._o_range)


class CUDAPreprocessor:
    def __init__(
        self: Self,
        output_shape: tuple[int, int],
        output_range: tuple[float, float],
        dtype: np.dtype,
        stream: cudart.cudaStream_t | None = None,
        *,
        no_copy: bool | None = None,
    ) -> None:
        _log.debug(
            f"Creating CUDA preprocessor: {output_shape}, {output_range}, {dtype}",
        )
        # allocate static output sizes
        self._o_shape = output_shape
        self._o_range = output_range
        self._o_dtype = dtype
        self._no_copy = no_copy

        # compute scale and offset
        self._scale = (self._o_range[1] - self._o_range[0]) / 255.0
        self._offset = self._o_range[0]

        # handle stream
        self._own_stream = False
        if stream:
            self._stream = stream
        else:
            self._stream = create_stream()
            self._own_stream = True

        # block and thread info
        self._num_threads = (16, 16, 3)
        self._num_blocks = (
            math.ceil(self._o_shape[1] / self._num_threads[0]),
            math.ceil(output_shape[0] / self._num_threads[1]),
            1,
        )

        # allocate output binding
        dummy = np.zeros(
            (1, 3, self._o_shape[1], self._o_shape[0]),
            dtype=self._o_dtype,
        )
        self._output_binding = create_binding(dummy, pagelocked_mem=True)

        # input shape/binding
        # input can be variable shape so only allocate when changed
        self._input_shape: tuple[int, ...] = (0, 0, 0)
        self._input_binding: Binding | None = None

        # store spot for the kernel args
        self._args: np.ndarray = np.ndarray([1], dtype=np.uint64)

        # load the kernel
        self._module, self._kernel = compile_and_load_kernel(
            SCALE_SWAP_TRANSPOSE_KERNEL_CODE,
            "scaleSwapTranspose",
        )

    def __del__(self: Self) -> None:
        with contextlib.suppress(AttributeError):
            cuda_call(cuda.cuModuleUnload(self._module))
        with contextlib.suppress(AttributeError):
            if self._own_stream:
                destroy_stream(self._stream)
        with contextlib.suppress(AttributeError):
            del self._output_binding
        with contextlib.suppress(AttributeError):
            del self._input_binding

    def _create_args(self: Self) -> None:
        if self._input_binding is None:
            err_msg = "CUDAPreprocessor arg creation occured before first input."
            raise RuntimeError(err_msg)

        input_arg: np.ndarray = np.array(
            [self._input_binding.allocation],
            dtype=np.uint64,
        )
        output_arg: np.ndarray = np.array(
            [self._output_binding.allocation],
            dtype=np.uint64,
        )
        height: np.ndarray = np.array([self._o_shape[1]], dtype=np.uint64)
        width: np.ndarray = np.array([self._o_shape[0]], dtype=np.uint64)
        scale: np.ndarray = np.array([self._scale], dtype=np.float32)
        offset: np.ndarray = np.array([self._offset], dtype=np.float32)

        args: list[np.ndarray] = [input_arg, output_arg, scale, offset, height, width]
        arg_ptrs: np.ndarray = np.array(
            [arg.ctypes.data for arg in args],
            dtype=np.uint64,
        )

        self._args = arg_ptrs

    def __call__(
        self: Self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        resized, ratios, padding = letterbox(image, self._o_shape)

        if self._input_binding is None or self._input_shape != resized.shape:
            self._input_shape = resized.shape
            self._input_binding = create_binding(
                resized,
                is_input=True,
                pagelocked_mem=True,
            )
            self._create_args()

        memcpy_host_to_device_async(
            self._input_binding.allocation,
            resized,
            self._stream,
        )

        cuda_call(
            cuda.cuLaunchKernel(
                self._kernel,
                *self._num_blocks,
                *self._num_threads,
                0,
                self._stream,
                self._args.ctypes.data,
                0,
            ),
        )

        memcpy_device_to_host_async(
            self._output_binding.host_allocation,
            self._output_binding.allocation,
            self._stream,
        )

        stream_synchronize(self._stream)

        if self._no_copy:
            return self._output_binding.host_allocation, ratios, padding
        return self._output_binding.host_allocation.copy(), ratios, padding
