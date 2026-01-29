# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np

from trtutils._log import LOG
from trtutils.core._bindings import create_binding
from trtutils.core._kernels import Kernel
from trtutils.core._stream import destroy_stream
from trtutils.image.kernels import IMAGENET_SST, SST_FAST

from ._image_preproc import GPUImagePreprocessor

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.compat._libs import cudart
    from trtutils.core._bindings import Binding


class CUDAPreprocessor(GPUImagePreprocessor):
    """CUDA-based preprocessor for image processing models."""

    def __init__(
        self: Self,
        output_shape: tuple[int, int],
        output_range: tuple[float, float],
        dtype: np.dtype[Any],
        resize: str = "letterbox",
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
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
        mean : tuple[float, float, float], optional
            The mean to subtract from the image.
            By default, None, which will not subtract any mean.
        std : tuple[float, float, float], optional
            The standard deviation to divide the image by.
            By default, None, which will not divide by any standard deviation.
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
            mean,
            std,
            stream,
            threads,
            tag,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
        )

        # SST input binding: (N, H', W', 3) uint8 - starts at batch_size=1
        dummy_sstinput: np.ndarray = np.zeros(
            (1, self._o_shape[1], self._o_shape[0], 3),
            dtype=np.uint8,
        )
        self._sst_input_binding = create_binding(
            dummy_sstinput,
            is_input=True,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )

        # SST output binding: (N, 3, H', W') float - starts at batch_size=1
        dummy_output: np.ndarray = np.zeros(
            (1, 3, self._o_shape[1], self._o_shape[0]),
            dtype=self._o_dtype,
        )
        self._output_binding = create_binding(
            dummy_output,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )

        # choose sst kernel based on whether imagenet mean/std are provided
        self._use_imagenet: bool = self._mean is not None and self._std is not None
        if self._use_imagenet:
            self._sst_kernel = Kernel(IMAGENET_SST[0], IMAGENET_SST[1])
        else:
            self._sst_kernel = Kernel(SST_FAST[0], SST_FAST[1])

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

    @property
    def output_binding(self: Self) -> Binding:
        """Get the output binding for the CUDA preprocessor."""
        return self._output_binding

    def _reallocate_batch_buffers(self: Self, batch_size: int) -> None:
        """Reallocate SST buffers if batch size changed."""
        if batch_size == self._current_batch_size:
            return

        # Reallocate SST input buffer: (N, H', W', 3) uint8
        dummy_sst_input = np.zeros(
            (batch_size, self._o_shape[1], self._o_shape[0], 3),
            dtype=np.uint8,
        )
        self._sst_input_binding = create_binding(
            dummy_sst_input,
            is_input=True,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )

        # Reallocate output buffer: (N, 3, H', W') float
        dummy_output = np.zeros(
            (batch_size, 3, self._o_shape[1], self._o_shape[0]),
            dtype=self._o_dtype,
        )
        self._output_binding = create_binding(
            dummy_output,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )

        self._current_batch_size = batch_size

    def _create_sst_args(
        self: Self,
        batch_size: int,
        *,
        verbose: bool | None = None,
    ) -> np.ndarray:
        """
        Create arguments for SST kernel with batch support.

        Returns
        -------
        np.ndarray
            Packed kernel argument array.

        Raises
        ------
        RuntimeError
            If the imagenet buffers are not allocated.

        """
        if verbose:
            LOG.debug(f"{self._tag}: Making sst args (batch_size={batch_size})")

        o_width, o_height = self._o_shape

        if self._use_imagenet:
            if self._mean_buffer is None or self._std_buffer is None:
                err_msg = "Imagenet buffers not allocated for SST kernel."
                raise RuntimeError(err_msg)
            # Signature: input, output, mean, std, height, width, batch_size
            sst_args = self._sst_kernel.create_args(
                self._sst_input_binding.allocation,
                self._output_binding.allocation,
                self._mean_buffer.allocation,
                self._std_buffer.allocation,
                o_height,
                o_width,
                batch_size,
                verbose=verbose,
            )
        else:
            # Signature: input, output, scale, offset, height, width, batch_size
            sst_args = self._sst_kernel.create_args(
                self._sst_input_binding.allocation,
                self._output_binding.allocation,
                self._scale,
                self._offset,
                o_height,
                o_width,
                batch_size,
                verbose=verbose,
            )

        return sst_args

    def direct_preproc(
        self: Self,
        images: list[np.ndarray],
        resize: str | None = None,
        *,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[int, list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Preprocess images for the model.

        Parameters
        ----------
        images : list[np.ndarray]
            The images to preprocess.
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
        tuple[int, list[tuple[float, float]], list[tuple[float, float]]]
            The GPU pointer to preprocessed data, list of ratios, and list of padding per image.

        """
        if verbose:
            LOG.debug(f"{self._tag}: direct_preproc")

        if not no_warn:
            LOG.warning(
                "Calling direct_preproc is potentially dangerous. Outputs can be overwritten inplace!",
            )

        batch_size = len(images)

        # Reallocate buffers if batch size changed
        self._reallocate_batch_buffers(batch_size)

        # Resize images and copy to batch buffer
        ratios_list, padding_list = self._resize_images_to_batch(
            images,
            self._sst_input_binding.allocation,
            resize=resize,
            verbose=verbose,
        )

        # Create SST args for batched processing
        sst_args = self._create_sst_args(batch_size, verbose=verbose)

        # Run batched SST kernel with batch_size in Z dimension
        batch_num_blocks = (
            self._num_blocks[0],
            self._num_blocks[1],
            batch_size,
        )

        self._sst_kernel.call(
            batch_num_blocks,
            self._num_threads,
            self._stream,
            sst_args,
        )

        return self._output_binding.allocation, ratios_list, padding_list
