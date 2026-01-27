# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from trtutils._engine import TRTEngine
from trtutils._log import LOG
from trtutils.compat._libs import trt
from trtutils.core._bindings import create_binding
from trtutils.core._memory import (
    memcpy_device_to_host_async,
    memcpy_host_to_device_async,
)
from trtutils.core._stream import destroy_stream, stream_synchronize
from trtutils.image.onnx_models import build_image_preproc, build_image_preproc_imagenet

from ._image_preproc import GPUImagePreprocessor, _is_single_image

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.compat._libs import cudart


class TRTPreprocessor(GPUImagePreprocessor):
    """TRT-based preprocessor for image processing models."""

    def __init__(
        self: Self,
        output_shape: tuple[int, int],
        output_range: tuple[float, float],
        dtype: np.dtype[Any],
        batch_size: int = 1,
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
        Create a TRTPreprocessor for image processing models.

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
        batch_size : int
            The batch size for the preprocessing engine.
            Default is 1.
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
            By default, 'TRTPreprocessor'
            If used within a model class, will be the model tag.
        pagelocked_mem : bool, optional
            Whether or not to allocate output memory as pagelocked.
            By default, pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which means the default host allocation will be used.

        Raises
        ------
        RuntimeError
            If imagenet buffers are required but not allocated.

        """
        tag = "TRTPreprocessor" if tag is None else f"{tag}.TRTPreprocessor"
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

        self._batch_size = batch_size

        # Batch intermediate buffer for TRT engine input
        dummy_intermediate: np.ndarray = np.zeros(
            (self._batch_size, self._o_shape[1], self._o_shape[0], 3),
            dtype=np.uint8,
        )
        self._intermediate_binding = create_binding(
            dummy_intermediate,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )

        # determine preprocessing mode, imagenet or yolo
        self._use_imagenet: bool = self._mean_buffer is not None and self._std_buffer is not None
        # if not imagenet, allocate the scale/offset CUDA locations
        if not self._use_imagenet:
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

        # assign the built engine path based on preprocessing mode
        if self._use_imagenet:
            self._engine_path = build_image_preproc_imagenet(
                self._o_shape, self._o_dtype, self._batch_size, trt.__version__
            )
        else:
            self._engine_path = build_image_preproc(
                self._o_shape, self._o_dtype, self._batch_size, trt.__version__
            )

        # create the engine
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
        self._gpu_pointers = [self._intermediate_binding.allocation]
        if not self._use_imagenet:
            self._gpu_pointers.extend(
                [
                    self._scale_binding.allocation,
                    self._offset_binding.allocation,
                ]
            )
        else:
            if self._mean_buffer is None or self._std_buffer is None:
                err_msg = "Imagenet buffers not allocated for TRT preprocessor."
                raise RuntimeError(err_msg)
            self._gpu_pointers.extend(
                [
                    self._mean_buffer.allocation,
                    self._std_buffer.allocation,
                ]
            )

    def __del__(self: Self) -> None:
        with contextlib.suppress(AttributeError, RuntimeError):
            if self._own_stream:
                destroy_stream(self._stream)
        with contextlib.suppress(AttributeError):
            del self._intermediate_binding
        with contextlib.suppress(AttributeError):
            del self._engine

    # preprocess overloads
    @overload
    def preprocess(
        self: Self,
        images: np.ndarray,
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @overload
    def preprocess(
        self: Self,
        images: list[np.ndarray],
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    def preprocess(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        resize: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Preprocess images for the model.

        Parameters
        ----------
        images : np.ndarray | list[np.ndarray]
            A single image (HWC format) or list of images to preprocess.
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
        tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]
            The preprocessed batch tensor, list of ratios, and list of padding per image.

        """
        # Handle single-image input
        is_single = _is_single_image(images)
        if is_single:
            images = [images]  # type: ignore[list-item]

        _, ratios_list, padding_list = self.direct_preproc(
            images,  # type: ignore[arg-type]
            resize=resize,
            no_warn=True,
            verbose=verbose,
        )

        batch_size = len(images)

        if not self._unified_mem:
            memcpy_device_to_host_async(
                self._engine_output_binding.host_allocation,
                self._engine_output_binding.allocation,
                self._stream,
            )

        stream_synchronize(self._stream)

        if no_copy:
            return (
                self._engine_output_binding.host_allocation[:batch_size],
                ratios_list,
                padding_list,
            )
        return (
            self._engine_output_binding.host_allocation[:batch_size].copy(),
            ratios_list,
            padding_list,
        )

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

        Raises
        ------
        ValueError
            If batch size exceeds the configured batch size for the TRT engine.

        """
        if verbose:
            LOG.debug(f"{self._tag}: direct_preproc")

        if not no_warn:
            LOG.warning(
                "Calling direct_preproc is potentially dangerous. Outputs can be overwritten inplace!",
            )

        batch_size = len(images)

        # Check batch size doesn't exceed configured engine batch size
        if batch_size > self._batch_size:
            err_msg = f"{self._tag}: Batch size {batch_size} exceeds configured batch size {self._batch_size}"
            raise ValueError(err_msg)

        # Resize images and copy to batch buffer
        ratios_list, padding_list = self._resize_images_to_batch(
            images,
            self._intermediate_binding.allocation,
            resize=resize,
            verbose=verbose,
        )

        # Run TRT engine on batched intermediate buffer
        output_ptrs = self._engine.raw_exec(self._gpu_pointers, no_warn=True)

        return output_ptrs[0], ratios_list, padding_list
