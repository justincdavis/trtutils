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

from trtutils.core._bindings import create_binding
from trtutils.core._kernels import Kernel
from trtutils.core._memory import (
    memcpy_device_to_host_async,
    memcpy_host_to_device_async,
)
from trtutils.core._stream import create_stream, destroy_stream, stream_synchronize
from trtutils.impls.kernels import LETTERBOX_RESIZE, LINEAR_RESIZE, SCALE_SWAP_TRANSPOSE

from ._process import preprocess

if TYPE_CHECKING:
    from typing_extensions import Self

    with contextlib.suppress(ImportError):
        from cuda import cudart  # type: ignore[import-untyped, import-not-found]

_COLOR_CHANNELS = 3
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
        resize: str = "letterbox",
        stream: cudart.cudaStream_t | None = None,
        threads: tuple[int, int, int] | None = None,
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
        resize : str, optional
            The default resize method to use.
            By default, letterbox resizing will be used.
            Options are: ['letterbox', 'linear']
        stream : cudart.cudaStream_t, optional
            The CUDA stream to use for preprocessing execution.
            If not provided, the preprocessor will use its own stream.
        threads : tuple[int, int, int], optional
            The number of threads to use per-block of computation.
            Can be changed depending on GPU size.

        Raises
        ------
        ValueError
            If the resize method is not valid

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

        # resize methods
        self._valid_methods = ["letterbox", "linear"]
        if resize not in self._valid_methods:
            err_msg = (
                f"Unknown method for image resizing. Options are {self._valid_methods}"
            )
            raise ValueError(err_msg)
        self._resize = resize

        with _CUDA_ALLOCATE_LOCK:
            # handle stream
            self._stream: cudart.cudaStream_t
            self._own_stream = False
            if stream:
                self._stream = stream
            else:
                self._stream = create_stream()
                self._own_stream = True

            # allocate input, sst_input, output binding
            # call resize kernel then sst kernel
            # need input -> intermediate -> output
            # for now just allocate 1080p image, reallocate when needed
            # resize kernel input binding
            self._allocated_input_shape: tuple[int, int, int] = (1080, 1920, 3)
            dummy_input: np.ndarray = np.zeros(
                self._allocated_input_shape,
                dtype=np.uint8,
            )
            self._input_binding = create_binding(
                dummy_input,
                is_input=True,
            )
            # these two CUDA allocations are static size
            # sst kernel input binding
            dummy_sstinput: np.ndarray = np.zeros(
                (self._o_shape[1], self._o_shape[0], 3),
                dtype=np.uint8,
            )
            self._sst_input_binding = create_binding(
                dummy_sstinput,
                is_input=True,
            )
            # sst kernel output binding
            dummy_output: np.ndarray = np.zeros(
                (1, 3, self._o_shape[1], self._o_shape[0]),
                dtype=self._o_dtype,
            )
            self._output_binding = create_binding(
                dummy_output,
                pagelocked_mem=True,
            )

            # block and thread info
            self._num_threads: tuple[int, int, int] = threads or (32, 32, 1)
            self._sst_num_blocks: tuple[int, int, int] = (
                math.ceil(self._o_shape[1] / self._num_threads[0]),
                math.ceil(self._o_shape[0] / self._num_threads[1]),
                1,
            )
            self._resize_num_blocks: tuple[int, int, int] = (
                math.ceil(self._allocated_input_shape[0] / self._num_threads[0]),
                math.ceil(self._allocated_input_shape[1] / self._num_threads[1]),
                1,
            )

            # load the kernels
            # sst kernel always used
            self._sst_kernel = Kernel(*SCALE_SWAP_TRANSPOSE)
            # either letterbox or linear is used
            self._linear_kernel = Kernel(*LINEAR_RESIZE)
            self._letterbox_kernel = Kernel(*LETTERBOX_RESIZE)

    def __del__(self: Self) -> None:
        with contextlib.suppress(AttributeError, RuntimeError):
            if self._own_stream:
                destroy_stream(self._stream)
        with contextlib.suppress(AttributeError):
            del self._input_binding
        with contextlib.suppress(AttributeError):
            del self._output_binding
        with contextlib.suppress(AttributeError):
            del self._sst_input_binding

    def _create_args(
        self: Self,
        height: int,
        width: int,
        method: str,
        *,
        verbose: bool | None = None,
    ) -> tuple[
        Kernel,
        np.ndarray,
        tuple[float, float],
        tuple[float, float],
        np.ndarray,
    ]:
        # pre-compute the common potions
        o_width, o_height = self._o_shape
        scale_x = o_width / width
        scale_y = o_height / height
        if method == "letterbox":
            scale = min(scale_x, scale_y)
            new_width = width * scale
            new_height = height * scale
            padding_x = (o_width - new_width) / 2
            padding_y = (o_height - new_height) / 2
            ratios = (scale, scale)
            padding = (padding_x, padding_y)

            # create args and assign kernel
            resize_kernel = self._letterbox_kernel
            resize_args = resize_kernel.create_args(
                self._input_binding.allocation,
                self._sst_input_binding.allocation,
                width,
                height,
                o_width,
                o_height,
                int(padding_x),
                int(padding_y),
                int(new_width),
                int(new_height),
                verbose=verbose,
            )
        else:
            o_width, o_height = self._o_shape
            scale_x = o_width / width
            scale_y = o_height / height
            ratios = (scale_x, scale_y)
            padding = (0.0, 0.0)

            # create args and assign kernel
            resize_kernel = self._linear_kernel
            resize_args = resize_kernel.create_args(
                self._input_binding.allocation,
                self._sst_input_binding.allocation,
                width,
                height,
                o_width,
                o_height,
                verbose=verbose,
            )
        sst_args = self._sst_kernel.create_args(
            self._sst_input_binding.allocation,
            self._output_binding.allocation,
            self._scale,
            self._offset,
            self._o_shape[0],
            verbose=verbose,
        )

        return resize_kernel, resize_args, ratios, padding, sst_args

    def _reallocate_input(self: Self, image: np.ndarray) -> None:
        self._input_binding = create_binding(
            image,
            is_input=True,
        )
        self._allocated_input_shape = image.shape  # type: ignore[assignment]
        self._resize_num_blocks = (
            math.ceil(self._allocated_input_shape[0] / self._num_threads[0]),
            math.ceil(self._allocated_input_shape[1] / self._num_threads[1]),
            1,
        )

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
        self.preprocess(rand_data, resize=self._resize, no_copy=True)

    def __call__(
        self: Self,
        image: np.ndarray,
        resize: str | None = None,
        *,
        no_copy: bool | None = None,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        """
        Preprocess an image for YOLO.

        Parameters
        ----------
        image : np.ndarray
            The image to preprocess.
        resize : str, optional
            The method to resize the image with.
            Options are [letterbox, linear], will use method
            provided in constructor by default.
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
        resize: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        """
        Preprocess an image for YOLO.

        Parameters
        ----------
        image : np.ndarray
            The image to preprocess.
        resize : str, optional
            The method to resize the image with.
            Options are [letterbox, linear], will use method
            provided in constructor by default.
        no_copy : bool, optional
            If True, the outputs will not be copied out
            from the cuda allocated host memory. Instead,
            the host memory will be returned directly.
            This memory WILL BE OVERWRITTEN INPLACE
            by future preprocessing calls.
        verbose : bool, optional
            Whether or not to output additional information
            to stdout. If not provided, will default to overall
            engines verbose setting.

        Returns
        -------
        tuple[np.ndarray, tuple[float, float], tuple[float, float]]
            The preprocessed image, ratios, and padding used for resizing.

        Raises
        ------
        ValueError
            If the method for resizing is not 'letterbox' or 'linear'
        ValueError
            If the image given is not color.

        """
        # valid the method
        resize = resize if resize is not None else self._resize
        if resize not in self._valid_methods:
            err_msg = (
                f"Unknown method for image resizing. Options are {self._valid_methods}"
            )
            raise ValueError(err_msg)

        # check if the image shape is the same as re have allocated with, if not update
        img_shape: tuple[int, int, int] = image.shape  # type: ignore[assignment]
        if img_shape != self._allocated_input_shape:
            if img_shape[2] != _COLOR_CHANNELS:
                err_msg = "Can only preprocess color images."
                raise ValueError(err_msg)

            self._reallocate_input(image)

        # create the arguments
        height, width = image.shape[:2]
        resize_kernel, resize_args, ratios, padding, sst_args = self._create_args(
            height,
            width,
            resize,
            verbose=verbose,
        )

        memcpy_host_to_device_async(
            self._input_binding.allocation,
            image,
            self._stream,
        )

        resize_kernel.call(
            self._resize_num_blocks,
            self._num_threads,
            self._stream,
            resize_args,
        )

        self._sst_kernel.call(
            self._sst_num_blocks,
            self._num_threads,
            self._stream,
            sst_args,
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
        resize: str | None = None,
        *,
        no_warn: bool | None = None,
        verbose: bool | None = None,
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
        verbose : bool, optional
            Whether or not to output additional information
            to stdout. If not provided, will default to overall
            engines verbose setting.

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

        # valid the method
        resize = resize if resize is not None else self._resize
        if resize not in self._valid_methods:
            err_msg = (
                f"Unknown method for image resizing. Options are {self._valid_methods}"
            )
            raise ValueError(err_msg)

        # check if the image shape is the same as re have allocated with, if not update
        img_shape: tuple[int, int, int] = image.shape  # type: ignore[assignment]
        if img_shape != self._allocated_input_shape:
            self._input_binding.free()  # free the memory explicitly
            if img_shape[2] != _COLOR_CHANNELS:
                err_msg = "Can only preprocess color images."
                raise ValueError(err_msg)

            self._reallocate_input(image)

        # create the arguments
        height, width = image.shape[:2]
        resize_kernel, resize_args, ratios, padding, sst_args = self._create_args(
            height,
            width,
            resize,
            verbose=verbose,
        )

        memcpy_host_to_device_async(
            self._input_binding.allocation,
            image,
            self._stream,
        )

        resize_kernel.call(
            self._resize_num_blocks,
            self._num_threads,
            self._stream,
            resize_args,
        )

        self._sst_kernel.call(
            self._sst_num_blocks,
            self._num_threads,
            self._stream,
            sst_args,
        )

        return self._output_binding.allocation, ratios, padding
