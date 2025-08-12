# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np

from trtutils._log import LOG
from trtutils.core._bindings import create_binding
from trtutils.core._kernels import Kernel
from trtutils.core._memory import (
    memcpy_device_to_host_async,
    memcpy_host_to_device_async,
)
from trtutils.core._stream import destroy_stream, stream_synchronize
from trtutils.image.kernels import SST_FAST

from ._abc import GPUImagePreprocessor

if TYPE_CHECKING:
    from typing_extensions import Self

    with contextlib.suppress(ImportError):
        try:
            import cuda.bindings.runtime as cudart
        except (ImportError, ModuleNotFoundError):
            from cuda import cudart


class CUDAPreprocessor(GPUImagePreprocessor):
    """CUDA-based preprocessor for image processing models."""

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
        Create a CUDAPreprocessor for image processing models.

        Parameters
        ----------
        output_shape : tuple[int, int]
            The shape of the image the model expects.
            In the form [width, height]
        output_range : tuple[float, float]
            The range of the image values the model expects.
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
            By default, 'CUDAPreprocessor'
            If used within a model class, will be the model tag.
        pagelocked_mem : bool, optional
            Whether or not to allocate output memory as pagelocked.
            By default, pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which means the default host allocation will be used.

        """
        tag = "CUDAPreprocessor" if tag is None else f"{tag}.CUDAPreprocessor"
        super().__init__(
            output_shape,
            output_range,
            dtype,
            resize,
            stream,
            threads,
            tag,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
        )

        # sst kernel input binding
        dummy_sstinput: np.ndarray = np.zeros(
            (self._o_shape[1], self._o_shape[0], 3),
            dtype=np.uint8,
        )
        self._sst_input_binding = create_binding(
            dummy_sstinput,
            is_input=True,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )
        # sst kernel output binding
        dummy_output: np.ndarray = np.zeros(
            (1, 3, self._o_shape[1], self._o_shape[0]),
            dtype=self._o_dtype,
        )
        self._output_binding = create_binding(
            dummy_output,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )

        # sst kernel always used
        self._sst_kernel = Kernel(*SST_FAST)

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
                self._sst_input_binding.allocation,
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
                self._sst_input_binding.allocation,
                width,
                height,
                o_width,
                o_height,
                verbose=verbose,
            )

        if verbose:
            LOG.debug(f"{self._tag}: Making sst args")

        sst_args = self._sst_kernel.create_args(
            self._sst_input_binding.allocation,
            self._output_binding.allocation,
            self._scale,
            self._offset,
            self._o_shape[0],
            verbose=verbose,
        )

        return resize_kernel, resize_args, ratios, padding, sst_args

    def preprocess(
        self: Self,
        image: np.ndarray,
        resize: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        """
        Preprocess an image for the model.

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
        Preprocess an image for the model.

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
        resize_kernel, resize_args, ratios, padding, sst_args = self._create_args(
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

        self._sst_kernel.call(
            self._num_blocks,
            self._num_threads,
            self._stream,
            sst_args,
        )

        return self._output_binding.allocation, ratios, padding
