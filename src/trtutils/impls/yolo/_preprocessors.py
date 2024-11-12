# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
import logging
import math
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from cv2ext.image import letterbox, resize_linear

with contextlib.suppress(ImportError):
    from cuda import cuda, cudart  # type: ignore[import-untyped, import-not-found]

from trtutils.core._bindings import create_binding
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

_CUDA_ALLOCATE_LOCK = Lock()

_log = logging.getLogger(__name__)


class CPUPreprocessor:
    """CPU-based preprocessor for YOLO."""

    def __init__(
        self: Self,
        output_shape: tuple[int, int],
        output_range: tuple[float, float],
        dtype: np.dtype,
    ) -> None:
        """
        Create a CPUPreprocessor for YOLO.

        Parameters
        ----------
        output_shape : tuple[int, int]
            The shape of the image YOLO expects.
            In the form [width, height]
        output_range : tuple[float, float]
            The range of the image values YOLO expects.
            Examples: (0.0, 1.0), (0.0, 255.0)
        dtype : np.dtype
            The datatype of the image.
            Examples: np.float32, np.float16, np.uint8

        """
        _log.debug(
            f"Creating CPU preprocessor: {output_shape}, {output_range}, {dtype}",
        )
        self._o_shape = output_shape
        self._o_range = output_range
        self._o_dtype = dtype

    def warmup(self: Self) -> None:
        """Compatibility function for CPU/CUDA parity."""
        rand_data: np.ndarray = np.random.default_rng().integers(
            0,
            255,
            (*self._o_shape, 3),
            dtype=np.uint8,
        )
        self.preprocess(rand_data)

    def __call__(
        self: Self,
        image: np.ndarray,
        resize: str = "letterbox",
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        """
        Preprocess an image for YOLO.

        Parameters
        ----------
        image : np.ndarray
            The image to preprocess.
        resize : str
            The method to resize the image with.
            By default letterbox, options are [letterbox, linear]

        Returns
        -------
        tuple[np.ndarray, tuple[float, float], tuple[float, float]]
            The preprocessed image, ratios, and padding used for resizing.

        """
        return self.preprocess(image, resize=resize)

    def preprocess(
        self: Self,
        image: np.ndarray,
        resize: str = "letterbox",
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        """
        Preprocess an image for YOLO.

        Parameters
        ----------
        image : np.ndarray
            The image to preprocess.
        resize : str
            The method to resize the image with.
            By default letterbox, options are [letterbox, linear]

        Returns
        -------
        tuple[np.ndarray, tuple[float, float], tuple[float, float]]
            The preprocessed image, ratios, and padding used for resizing.

        """
        return preprocess(image, self._o_shape, self._o_dtype, self._o_range, resize)


class CUDAPreprocessor:
    """CUDA-based preprocessor for YOLO."""

    def __init__(
        self: Self,
        output_shape: tuple[int, int],
        output_range: tuple[float, float],
        dtype: np.dtype,
        stream: cudart.cudaStream_t | None = None,
    ) -> None:
        """
        Create a CUDAPreprocessor for YOLO.

        Parameters
        ----------
        output_shape : tuple[int, int]
            The shape of the image YOLO expects.
            In the form [width, height]
        output_range : tuple[float, float]
            The range of the image values YOLO expects.
            Examples: (0.0, 1.0), (0.0, 255.0)
        dtype : np.dtype
            The datatype of the image.
            Examples: np.float32, np.float16, np.uint8
        stream : cudart.cudaStream_t, optional
            The CUDA stream to use for preprocessing execution.
            If not provided, the preprocessor will use its own stream.

        """
        _log.debug(
            f"Creating CUDA preprocessor: {output_shape}, {output_range}, {dtype}",
        )
        # allocate static output sizes
        self._o_shape = output_shape
        self._o_range = output_range
        self._o_dtype = dtype

        # compute scale and offset
        self._scale: float = (self._o_range[1] - self._o_range[0]) / 255.0
        self._offset: float = self._o_range[0]

        with _CUDA_ALLOCATE_LOCK:
            # handle stream
            self._stream: cudart.cudaStream_t
            self._own_stream = False
            if stream:
                self._stream = stream
            else:
                self._stream = create_stream()
                self._own_stream = True

            # block and thread info
            self._num_threads: tuple[int, int, int] = (16, 16, 3)
            self._num_blocks: tuple[int, int, int] = (
                math.ceil(self._o_shape[1] / self._num_threads[0]),
                math.ceil(output_shape[0] / self._num_threads[1]),
                1,
            )

            # allocate input/output binding
            dummy_input: np.ndarray = np.zeros(
                (self._o_shape[1], self._o_shape[0], 3),
                dtype=np.uint8,
            )
            # set is_input since we do not use the host_allocation here
            self._input_binding = create_binding(
                dummy_input,
                is_input=True,
            )
            dummy_output: np.ndarray = np.zeros(
                (1, 3, self._o_shape[1], self._o_shape[0]),
                dtype=self._o_dtype,
            )
            # set pagelocked memory since we read from the host allocation
            self._output_binding = create_binding(
                dummy_output,
                pagelocked_mem=True,
            )

            # load the kernel
            self._module, self._kernel = compile_and_load_kernel(
                SCALE_SWAP_TRANSPOSE_KERNEL_CODE,
                "scaleSwapTranspose",
            )

    def __del__(self: Self) -> None:
        with contextlib.suppress(AttributeError, RuntimeError):
            cuda_call(cuda.cuModuleUnload(self._module))
        with contextlib.suppress(AttributeError, RuntimeError):
            if self._own_stream:
                destroy_stream(self._stream)
        with contextlib.suppress(AttributeError):
            del self._output_binding
        with contextlib.suppress(AttributeError):
            del self._input_binding

    def _create_args(self: Self) -> np.ndarray:
        # create a np.ndarray of pointers to the numpy arrays (CPU side pointers)
        # From: https://nvidia.github.io/cuda-python/overview.html#cuda-python-workflow
        # Not-stated in the overview, BUT
        # args MUST BE REGENERATED (EVEN IF IDENTICAL) FOR EVERY KERNEL CALL
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

        return arg_ptrs

    def warmup(self: Self) -> None:
        """
        Warmup the CUDA preprocessor.

        Allocates all CUDA memory and enables future passes
        to be significantly faster.
        """
        rand_data: np.ndarray = np.random.default_rng().integers(
            0,
            255,
            (*self._o_shape, 3),
            dtype=np.uint8,
        )
        self.preprocess(rand_data, no_copy=True)

    def __call__(
        self: Self,
        image: np.ndarray,
        resize: str = "letterbox",
        *,
        no_copy: bool | None = None,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        """
        Preprocess an image for YOLO.

        Parameters
        ----------
        image : np.ndarray
            The image to preprocess.
        resize : str
            The method to resize the image with.
            By default letterbox, options are [letterbox, linear]
        no_copy : bool, optional
            If True, the outputs will not be copied out
            from the cuda allocated host memory. Instead,
            the host memory will be returned directly.
            This memory WILL BE OVERWRITTEN INPLACE
            by future preprocessing calls.

        Returns
        -------
        tuple[np.ndarray, tuple[float, float], tuple[float, float]]
            The preprocessed image, ratios, and padding used for resizing.

        """
        return self.preprocess(image, resize=resize, no_copy=no_copy)

    def preprocess(
        self: Self,
        image: np.ndarray,
        resize: str = "letterbox",
        *,
        no_copy: bool | None = None,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        """
        Preprocess an image for YOLO.

        Parameters
        ----------
        image : np.ndarray
            The image to preprocess.
        resize : str
            The method to resize the image with.
            By default letterbox, options are [letterbox, linear]
        no_copy : bool, optional
            If True, the outputs will not be copied out
            from the cuda allocated host memory. Instead,
            the host memory will be returned directly.
            This memory WILL BE OVERWRITTEN INPLACE
            by future preprocessing calls.

        Returns
        -------
        tuple[np.ndarray, tuple[float, float], tuple[float, float]]
            The preprocessed image, ratios, and padding used for resizing.

        Raises
        ------
        ValueError
            If the method for resizing is not 'letterbox' or 'linear'

        """
        if resize == "letterbox":
            resized, ratios, padding = letterbox(image, self._o_shape)
        elif resize == "linear":
            resized, ratios = resize_linear(image, self._o_shape)
            padding = (0.0, 0.0)
        else:
            err_msg = (
                "Unknown method for image resizing. Options are ['letterbox', 'linear']"
            )
            raise ValueError(err_msg)

        args = self._create_args()

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
                args.ctypes.data,
                0,
            ),
        )

        memcpy_device_to_host_async(
            self._output_binding.host_allocation,
            self._output_binding.allocation,
            self._stream,
        )

        stream_synchronize(self._stream)

        if no_copy:
            return self._output_binding.host_allocation, ratios, padding
        return self._output_binding.host_allocation.copy(), ratios, padding

    def direct_preproc(
        self: Self,
        image: np.ndarray,
        resize: str = "letterbox",
        *,
        no_warn: bool | None = None,
    ) -> tuple[int, tuple[float, float], tuple[float, float]]:
        """
        Preprocess an image for YOLO.

        Parameters
        ----------
        image : np.ndarray
            The image to preprocess.
        resize : str
            The method to resize the image with.
            By default letterbox, options are [letterbox, linear]
        no_warn : bool, optional
            If True, do not warn about usage.

        Returns
        -------
        tuple[int, tuple[float, float], tuple[float, float]]
            The GPU pointer to preprocessed data, ratios, and padding used for resizing.

        Raises
        ------
        ValueError
            If the method for resizing is not 'letterbox' or 'linear'

        """
        if not no_warn:
            _log.warning(
                "Calling direct_preproc is potentially dangerous. Outputs can be overwritten inplace!",
            )

        if resize == "letterbox":
            resized, ratios, padding = letterbox(image, self._o_shape)
        elif resize == "linear":
            resized, ratios = resize_linear(image, self._o_shape)
            padding = (0.0, 0.0)
        else:
            err_msg = (
                "Unknown method for image resizing. Options are ['letterbox', 'linear']"
            )
            raise ValueError(err_msg)

        args = self._create_args()

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
                args.ctypes.data,
                0,
            ),
        )

        return self._output_binding.allocation, ratios, padding
