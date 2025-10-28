# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np

with contextlib.suppress(ImportError):
    import tensorrt as trt

from trtutils._engine import TRTEngine
from trtutils._log import LOG
from trtutils.core._bindings import create_binding
from trtutils.core._memory import (
    memcpy_device_to_host_async,
    memcpy_host_to_device_async,
)
from trtutils.core._stream import destroy_stream, stream_synchronize
from trtutils.image.onnx_models import build_image_preproc, build_image_preproc_imagenet

from ._image_preproc import GPUImagePreprocessor

if TYPE_CHECKING:
    from typing_extensions import Self

    with contextlib.suppress(ImportError):
        try:
            import cuda.bindings.runtime as cudart
        except (ImportError, ModuleNotFoundError):
            from cuda import cudart

    from trtutils.core._kernels import Kernel


class TRTPreprocessor(GPUImagePreprocessor):
    """TRT-based preprocessor for image processing models."""

    def __init__(
        self: Self,
        output_shape: tuple[int, int],
        output_range: tuple[float, float],
        dtype: np.dtype,
        resize: str = "letterbox",
        stream: cudart.cudaStream_t | None = None,
        threads: tuple[int, int, int] | None = None,
        tag: str | None = None,
        imagenet_mean: np.ndarray | None = None,
        imagenet_std: np.ndarray | None = None,
        *,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        use_imagenet_norm: bool | None = None,
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
            If used within a model class, will be the model tag.
        pagelocked_mem : bool, optional
            Whether or not to allocate output memory as pagelocked.
            By default, pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which means the default host allocation will be used.
        use_imagenet_norm : bool, optional
            Whether to use ImageNet normalization instead of standard scale/offset.
            By default None, which will enable ImageNet normalization if mean/std are provided.
        imagenet_mean : np.ndarray, optional
            Mean values for ImageNet normalization, shape (1, 3, 1, 1) or (3,).
            By default None.
        imagenet_std : np.ndarray, optional
            Standard deviation values for ImageNet normalization, shape (1, 3, 1, 1) or (3,).
            By default None.

        """
        tag = "TRTPreprocessor" if tag is None else f"{tag}.TRTPreprocessor"
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

        dummy_intermediate: np.ndarray = np.zeros(
            (self._o_shape[1], self._o_shape[0], 3),
            dtype=np.uint8,
        )
        self._intermediate_binding = create_binding(
            dummy_intermediate,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )

        # Determine if we should use ImageNet normalization
        self._use_imagenet_norm = use_imagenet_norm
        if self._use_imagenet_norm is None:
            self._use_imagenet_norm = (imagenet_mean is not None and imagenet_std is not None)

        if self._use_imagenet_norm:
            # Validate and prepare mean/std
            print("IMAGENET")
            if imagenet_mean is None or imagenet_std is None:
                err_msg = "imagenet_mean and imagenet_std must be provided when use_imagenet_norm is True"
                raise ValueError(err_msg)
            
            # Reshape if needed
            mean_arr = self._prepare_norm_array(imagenet_mean, "mean")
            std_arr = self._prepare_norm_array(imagenet_std, "std")
            
            # Allocate mean/std CUDA locations
            self._mean_binding = create_binding(mean_arr)
            self._std_binding = create_binding(std_arr)
            memcpy_host_to_device_async(
                self._mean_binding.allocation,
                mean_arr,
                self._stream,
            )
            memcpy_host_to_device_async(
                self._std_binding.allocation,
                std_arr,
                self._stream,
            )
            stream_synchronize(self._stream)

            # allocate the ImageNet preprocessing engine
            self._engine_path = build_image_preproc_imagenet(
                self._o_shape, self._o_dtype, trt.__version__
            )
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
                self._mean_binding.allocation,
                self._std_binding.allocation,
            ]
        else:
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
            self._engine_path = build_image_preproc(
                self._o_shape, self._o_dtype, trt.__version__
            )
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
            del self._mean_binding
        with contextlib.suppress(AttributeError):
            del self._std_binding
        with contextlib.suppress(AttributeError):
            del self._engine

    def _prepare_norm_array(
        self: Self,
        arr: np.ndarray,
        name: str,
    ) -> np.ndarray:
        """
        Prepare normalization array (mean or std) to correct shape.

        Parameters
        ----------
        arr : np.ndarray
            Input array, can be shape (3,) or (1, 3, 1, 1)
        name : str
            Name of the array for error messages

        Returns
        -------
        np.ndarray
            Array reshaped to (1, 3, 1, 1) with dtype float32

        Raises
        ------
        ValueError
            If array has incorrect shape

        """
        arr = np.asarray(arr, dtype=np.float32)
        
        if arr.shape == (3,):
            # Reshape from (3,) to (1, 3, 1, 1)
            arr = arr.reshape(1, 3, 1, 1)
        elif arr.shape == (1, 3, 1, 1):
            # Already correct shape
            pass
        else:
            err_msg = f"{name} must have shape (3,) or (1, 3, 1, 1), got {arr.shape}"
            raise ValueError(err_msg)
        
        return arr

    def update_mean_std(
        self: Self,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        """
        Update the mean and std values for ImageNet normalization.

        Parameters
        ----------
        mean : np.ndarray
            Mean values, shape (3,) or (1, 3, 1, 1)
        std : np.ndarray
            Standard deviation values, shape (3,) or (1, 3, 1, 1)

        Raises
        ------
        RuntimeError
            If ImageNet normalization is not enabled

        """
        if not self._use_imagenet_norm:
            err_msg = "Cannot update mean/std when ImageNet normalization is not enabled"
            raise RuntimeError(err_msg)
        
        # Prepare arrays
        mean_arr = self._prepare_norm_array(mean, "mean")
        std_arr = self._prepare_norm_array(std, "std")
        
        # Update GPU memory
        memcpy_host_to_device_async(
            self._mean_binding.allocation,
            mean_arr,
            self._stream,
        )
        memcpy_host_to_device_async(
            self._std_binding.allocation,
            std_arr,
            self._stream,
        )
        stream_synchronize(self._stream)

    @property
    def use_imagenet_norm(self: Self) -> bool:
        """Get whether ImageNet normalization is enabled."""
        return self._use_imagenet_norm

    @use_imagenet_norm.setter
    def use_imagenet_norm(self: Self, value: bool) -> None:
        """
        Set whether ImageNet normalization is enabled.

        Parameters
        ----------
        value : bool
            Whether to enable ImageNet normalization

        Raises
        ------
        RuntimeError
            If trying to enable ImageNet normalization but mean/std not allocated

        """
        if value and not hasattr(self, "_mean_binding"):
            err_msg = "Cannot enable ImageNet normalization: mean/std not allocated. Create a new preprocessor with imagenet_mean and imagenet_std parameters."
            raise RuntimeError(err_msg)
        if not value and not hasattr(self, "_scale_binding"):
            err_msg = "Cannot disable ImageNet normalization: scale/offset not allocated. Create a new preprocessor without imagenet normalization."
            raise RuntimeError(err_msg)
        
        err_msg = "Cannot dynamically switch between ImageNet and standard normalization modes. Create a new preprocessor instead."
        raise RuntimeError(err_msg)

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
        resize_kernel, resize_args, ratios, padding = self._create_args(
            height,
            width,
            resize,
            verbose=verbose,
        )

        if verbose:
            LOG.debug(f"Ratios: {ratios}")
            LOG.debug(f"Padding: {padding}")

        # Update extra GPU buffers for input schemas
        self._update_extra_buffers(height, width, ratios)

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
