# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING

import numpy as np

from trtutils._engine import TRTEngine
from trtutils._log import LOG
from trtutils.core._bindings import create_binding
from trtutils.core._kernels import Kernel
from trtutils.core._memory import (
    memcpy_device_to_host_async,
    memcpy_host_to_device_async,
)
from trtutils.core._stream import create_stream, destroy_stream, stream_synchronize
from trtutils.impls.kernels import LETTERBOX_RESIZE, LINEAR_RESIZE
from trtutils.impls.onnx_models import build_yolo_preproc

if TYPE_CHECKING:
    from typing_extensions import Self

    with contextlib.suppress(ImportError):
        try:
            import cuda.bindings.runtime as cudart
        except (ImportError, ModuleNotFoundError):
            from cuda import cudart

_COLOR_CHANNELS = 3


class TRTPreprocessor:
    """TRT-based preprocessor for YOLO."""

    def __init__(
        self: Self,
        output_shape: tuple[int, int],
        output_range: tuple[float, float],
        dtype: np.dtype,
        resize: str = "letterbox",
        stream: cudart.cudaStream_t | None = None,
        threads: tuple[int, int, int] | None = None,
        tag: str | None = None,
        *,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
    ) -> None:
        """
        Create a TRTPreprocessor for YOLO.

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
        tag : str
            The tag to prefix to all logging statements made.
            By default, 'TRTPreprocessor'
            If used within a YOLO class, will be the YOLO tag.
        pagelocked_mem : bool, optional
            Whether or not to allocate output memory as pagelocked.
            By default, pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which means the default host allocation will be used.

        Raises
        ------
        ValueError
            If the resize method is not valid

        """
        self._tag = "TRTPreprocessor" if tag is None else f"{tag}.TRTPreprocessor"

        LOG.debug(
            f"{self._tag}: Creating preprocessor: {output_shape}, {output_range}, {dtype}",
        )
        # allocate static output sizes
        self._o_shape = output_shape
        self._o_range = output_range
        self._o_dtype = dtype
        self._pagelocked_mem = pagelocked_mem if pagelocked_mem is not None else True
        self._unified_mem = unified_mem

        # compute scale and offset
        self._scale: float = (self._o_range[1] - self._o_range[0]) / 255.0
        self._offset: float = self._o_range[0]

        # resize methods
        self._valid_methods = ["letterbox", "linear"]
        if resize not in self._valid_methods:
            err_msg = f"{self._tag}: Unknown method for image resizing. Options are {self._valid_methods}"
            raise ValueError(err_msg)
        self._resize = resize

        # handle stream
        self._stream: cudart.cudaStream_t
        self._own_stream = False
        if stream is not None:
            self._stream = stream
        else:
            self._stream = create_stream()
            self._own_stream = True

        # allocate input, output binding
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
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )
        dummy_intermediate: np.ndarray = np.zeros(
            (self._o_shape[1], self._o_shape[0], 3),
            dtype=np.uint8,
        )
        self._intermediate_binding = create_binding(
            dummy_intermediate,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )

        # block and thread info
        self._num_threads: tuple[int, int, int] = threads or (32, 32, 1)
        self._num_blocks: tuple[int, int, int] = (
            math.ceil(self._o_shape[0] / self._num_threads[0]),
            math.ceil(self._o_shape[1] / self._num_threads[1]),
            1,
        )

        # either letterbox or linear is used
        self._linear_kernel = Kernel(*LINEAR_RESIZE)
        self._letterbox_kernel = Kernel(*LETTERBOX_RESIZE)

        # allocate the scale/offset CUDA locations
        scale_arr: np.ndarray = np.array((self._scale,), dtype=np.float32)
        self._scale_binding = create_binding(scale_arr)
        memcpy_host_to_device_async(
            self._scale_binding.allocation,
            scale_arr,
            self._stream,
        )
        offset_arr: np.ndarray = np.array((self._offset,), dtype=np.float32)
        self._offset_binding = create_binding(offset_arr)
        memcpy_host_to_device_async(
            self._offset_binding.allocation,
            offset_arr,
            self._stream,
        )
        stream_synchronize(self._stream)

        # allocate the trtengine
        self._engine_path = build_yolo_preproc(self._o_shape, self._o_dtype)
        self._engine = TRTEngine(
            self._engine_path,
            stream=self._stream,
            warmup_iterations=1,
            warmup=True,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )
        self._engine_output_binding = self._engine.output_bindings[0]

        # pre-allocate the input pointer list for the engine
        self._gpu_pointers = [
            self._intermediate_binding.allocation,
            self._scale_binding.allocation,
            self._offset_binding.allocation,
        ]

    def __del__(self: Self) -> None:
        with contextlib.suppress(AttributeError, RuntimeError):
            if self._own_stream:
                destroy_stream(self._stream)
        with contextlib.suppress(AttributeError):
            del self._input_binding
        with contextlib.suppress(AttributeError):
            del self._intermediate_binding
        with contextlib.suppress(AttributeError):
            del self._scale_binding
        with contextlib.suppress(AttributeError):
            del self._offset_binding
        with contextlib.suppress(AttributeError):
            del self._engine

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
    ]:
        if verbose:
            LOG.debug(f"{self._tag}: create_args")

        # pre-compute the common potions
        o_width, o_height = self._o_shape
        scale_x = o_width / width
        scale_y = o_height / height
        if method == "letterbox":
            if verbose:
                LOG.debug(f"{self._tag}: Making letterbox args")

            scale = min(scale_x, scale_y)
            new_width = int(width * scale)
            new_height = int(height * scale)
            padding_x = int((o_width - new_width) / 2)
            padding_y = int((o_height - new_height) / 2)
            ratios = (scale, scale)
            padding = (padding_x, padding_y)

            # create args and assign kernel
            resize_kernel = self._letterbox_kernel
            resize_args = resize_kernel.create_args(
                self._input_binding.allocation,
                self._intermediate_binding.allocation,
                width,
                height,
                o_width,
                o_height,
                padding_x,
                padding_y,
                new_width,
                new_height,
                verbose=verbose,
            )
        else:
            if verbose:
                LOG.debug(f"{self._tag}: Making linear args")

            o_width, o_height = self._o_shape
            scale_x = o_width / width
            scale_y = o_height / height
            ratios = (scale_x, scale_y)
            padding = (0, 0)

            # create args and assign kernel
            resize_kernel = self._linear_kernel
            resize_args = resize_kernel.create_args(
                self._input_binding.allocation,
                self._intermediate_binding.allocation,
                width,
                height,
                o_width,
                o_height,
                verbose=verbose,
            )

        return resize_kernel, resize_args, ratios, padding

    def _reallocate_input(
        self: Self,
        image: np.ndarray,
        *,
        verbose: bool | None = None,
    ) -> None:
        if verbose:
            LOG.debug(f"{self._tag}: Reallocating input bindings")
            LOG.debug(
                f"{self._tag}: Reallocation -> new shape: {image.shape}, old shape: {self._input_binding.shape}",
            )

        self._allocated_input_shape = image.shape  # type: ignore[assignment]
        self._input_binding = create_binding(
            image,
            is_input=True,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )

    def _validate_input(
        self: Self,
        image: np.ndarray,
        resize: str | None = None,
        *,
        verbose: bool | None = None,
    ) -> str:
        if verbose:
            LOG.debug(f"{self._tag}: validate_input")

        # valid the method
        resize = resize if resize is not None else self._resize
        if resize not in self._valid_methods:
            err_msg = f"{self._tag}: Unknown method for image resizing. Options are {self._valid_methods}"
            raise ValueError(err_msg)

        img_shape: tuple[int, int, int] = image.shape  # type: ignore[assignment]

        if verbose:
            LOG.debug(
                f"{self._tag}: Image shape: {img_shape}, Allocated shape: {self._allocated_input_shape}",
            )

        # check if the image shape is the same as re have allocated with, if not update
        if img_shape != self._allocated_input_shape:
            if img_shape[2] != _COLOR_CHANNELS:
                err_msg = f"{self._tag}: Can only preprocess color images."
                raise ValueError(err_msg)

            self._reallocate_input(image, verbose=verbose)

        return resize

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

        """
        return self.preprocess(image, resize=resize, no_copy=no_copy, verbose=verbose)

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

        """
        _, ratios, padding = self.direct_preproc(
            image,
            resize=resize,
            no_warn=True,
            verbose=verbose,
        )

        if not self._unified_mem:
            memcpy_device_to_host_async(
                self._engine_output_binding.host_allocation,
                self._engine_output_binding.allocation,
                self._stream,
            )

        stream_synchronize(self._stream)

        if no_copy:
            return self._engine_output_binding.host_allocation, ratios, padding
        return self._engine_output_binding.host_allocation.copy(), ratios, padding

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

        """
        if verbose:
            LOG.debug(f"{self._tag}: direct_preproc")

        if not no_warn:
            LOG.warning(
                "Calling direct_preproc is potentially dangerous. Outputs can be overwritten inplace!",
            )

        # valid the method
        resize = self._validate_input(image, resize, verbose=verbose)

        # create the arguments
        height, width = image.shape[:2]
        resize_kernel, resize_args, ratios, padding = self._create_args(
            height,
            width,
            resize,
            verbose=verbose,
        )

        if verbose:
            LOG.debug(f"Ratios: {ratios}")
            LOG.debug(f"Padding: {padding}")

        if self._pagelocked_mem and self._unified_mem:
            np.copyto(self._input_binding.host_allocation, image)
        else:
            memcpy_host_to_device_async(
                self._input_binding.allocation,
                image,
                self._stream,
            )

        resize_kernel.call(
            self._num_blocks,
            self._num_threads,
            self._stream,
            resize_args,
        )

        output_ptrs = self._engine.raw_exec(self._gpu_pointers, no_warn=True)

        return output_ptrs[0], ratios, padding
