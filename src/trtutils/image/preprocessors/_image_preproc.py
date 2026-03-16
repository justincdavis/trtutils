# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import nvtx

from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.core._bindings import create_binding
from trtutils.core._kernels import Kernel
from trtutils.core._memory import (
    memcpy_device_to_host_async,
    memcpy_host_to_device_async,
)
from trtutils.core._stream import create_stream, stream_synchronize
from trtutils.image.kernels import LETTERBOX_RESIZE, LINEAR_RESIZE

if TYPE_CHECKING:
    from typing import ClassVar

    from typing_extensions import Self

    from trtutils.compat._libs import cudart
    from trtutils.core._bindings import Binding

_COLOR_CHANNELS = 3
_IMAGE_DIMENSIONS = 3


def _is_single_image(images: np.ndarray | list[np.ndarray]) -> bool:
    """
    Check if input is a single HWC image vs a batch.

    Returns
    -------
    bool
        True if images is a single HWC ndarray (ndim == 3), False otherwise.

    """
    return isinstance(images, np.ndarray) and images.ndim == _IMAGE_DIMENSIONS


class ImagePreprocessor(ABC):
    """Abstract base class for image preprocessors."""

    _valid_methods: ClassVar[list[str]] = ["letterbox", "linear"]

    def __init__(
        self: Self,
        output_shape: tuple[int, int],
        output_range: tuple[float, float],
        dtype: np.dtype[Any],
        resize: str = "letterbox",
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        tag: str | None = None,
    ) -> None:
        """
        Create an ImagePreprocessor.

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
        tag : str
            The tag to prefix to all logging statements made.
            By default, 'TRTPreprocessor'
            If used within a model class, will be the model tag.

        Raises
        ------
        ValueError
            If the resize method is not valid

        """
        # output info
        self._o_shape: tuple[int, int] = output_shape
        self._o_range: tuple[float, float] = output_range
        self._o_dtype: np.dtype = dtype

        # preprocessing info
        self._scale: float = 255.0
        self._offset: float = 0.0
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

        # class info
        self._tag: str | None = tag

        # setup nvtx tags
        self._nvtx_tags: dict[str, str] = {
            "init": f"preproc::init [{self._tag}]",
            "warmup": f"preproc::warmup [{self._tag}]",
            "preprocess": f"preproc::preprocess [{self._tag}]",
        }

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["init"])

        # compute scale, offset
        self.update_output_range(output_range)
        # setup the mean and std arrays
        if mean is not None and std is not None:
            self.update_mean_std(mean, std)

        # check resize method
        if resize not in ImagePreprocessor._valid_methods:
            err_msg = f"{self._tag}: Unknown method for image resizing. Options are {ImagePreprocessor._valid_methods}"
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # init
            raise ValueError(err_msg)
        self._resize: str = resize

        # mark setup in logs
        LOG.debug(
            f"{self._tag}: Creating preprocessor: {output_shape}, {output_range}, {dtype}",
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # init

    def update_output_range(self: Self, output_range: tuple[float, float]) -> None:
        """
        Update the output range of the preprocessor.

        Parameters
        ----------
        output_range : tuple[float, float]
            The new output range.

        """
        self._o_range = output_range
        self._scale = (self._o_range[1] - self._o_range[0]) / 255.0
        self._offset = self._o_range[0]

    def update_mean_std(
        self: Self, mean: tuple[float, float, float], std: tuple[float, float, float]
    ) -> None:
        """
        Update the mean and standard deviation of the preprocessor.

        Parameters
        ----------
        mean : tuple[float, float, float]
            The new mean.
        std : tuple[float, float, float]
            The new standard deviation.

        Raises
        ------
        ValueError
            If the mean or std is not a tuple of 3 floats

        """
        if len(mean) != _COLOR_CHANNELS:
            err_msg = f"{self._tag}: Mean must be a tuple of 3 floats"
            raise ValueError(err_msg)
        if len(std) != _COLOR_CHANNELS:
            err_msg = f"{self._tag}: Std must be a tuple of 3 floats"
            raise ValueError(err_msg)
        self._mean = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)

    @abstractmethod
    def warmup(self: Self) -> None: ...

    # __call__ overloads
    @overload
    def __call__(
        self: Self,
        images: np.ndarray,
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @overload
    def __call__(
        self: Self,
        images: list[np.ndarray],
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @abstractmethod
    def __call__(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        resize: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

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

    @abstractmethod
    def preprocess(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        resize: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...


class GPUImagePreprocessor(ImagePreprocessor):
    """GPU-based image preprocessor."""

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
        orig_size_dtype: np.dtype[Any] | None = None,
    ) -> None:
        """
        Create a GPUImagePreprocessor for image processing models.

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
            By default, 'TRTPreprocessor'
            If used within a model class, will be the model tag.
        pagelocked_mem : bool, optional
            Whether or not to allocate output memory as pagelocked.
            By default, pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which means the default host allocation will be used.
        orig_size_dtype : np.dtype, optional
            The dtype to use for the orig_size buffer. Default is np.int32.

        """
        super().__init__(
            output_shape=output_shape,
            output_range=output_range,
            dtype=dtype,
            resize=resize,
            mean=mean,
            std=std,
            tag=tag,
        )

        self._nvtx_tags.update(
            {
                "gpu_init": f"preproc::gpu_init [{self._tag}]",
                "gpu_warmup": f"preproc::gpu_warmup [{self._tag}]",
                "gpu_preprocess": f"preproc::gpu_preprocess [{self._tag}]",
                "direct_preproc": f"preproc::direct_preproc [{self._tag}]",
                "resize_images_to_batch": f"preproc::resize_images_to_batch [{self._tag}]",
                "create_resize_args": f"preproc::create_resize_args [{self._tag}]",
                "update_extra_buffers": f"preproc::update_extra_buffers [{self._tag}]",
                "validate_input": f"preproc::validate_input [{self._tag}]",
                "reallocate_input": f"preproc::reallocate_input [{self._tag}]",
                "allocate_imagenet_buffers": f"preproc::allocate_imagenet_buffers [{self._tag}]",
                "resize_single": f"preproc::resize_single_image [{self._tag}]",
                "resize_homogeneous": f"preproc::resize_homogeneous_batch [{self._tag}]",
                "resize_heterogeneous": f"preproc::resize_heterogeneous_batch [{self._tag}]",
                "reallocate_batch_input": f"preproc::reallocate_batch_input [{self._tag}]",
                "pack_batch_buffer": f"preproc::pack_batch_buffer [{self._tag}]",
                "stream_sync": f"preproc::stream_sync [{self._tag}]",
                "copy_output": f"preproc::copy_output [{self._tag}]",
            }
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["gpu_init"])

        # handle memory flags
        self._pagelocked_mem = pagelocked_mem if pagelocked_mem is not None else True
        self._unified_mem = unified_mem

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

        # block and thread info
        self._num_threads: tuple[int, int, int] = threads or (32, 32, 1)
        self._num_blocks: tuple[int, int, int] = (
            math.ceil(self._o_shape[0] / self._num_threads[0]),
            math.ceil(self._o_shape[1] / self._num_threads[1]),
            1,
        )

        # either letterbox or linear is used
        self._linear_kernel = Kernel(LINEAR_RESIZE[0], LINEAR_RESIZE[1])
        self._letterbox_kernel = Kernel(LETTERBOX_RESIZE[0], LETTERBOX_RESIZE[1])

        # if the imagenet mean/std are supplied, allocate the cuda buffers
        self._mean_buffer: Binding | None = None
        self._std_buffer: Binding | None = None
        if mean is not None and std is not None:
            self._allocate_imagenet_buffers()

        # Allocate GPU buffers for alternative input schemas
        self._orig_size_dtype: np.dtype[Any] = (
            orig_size_dtype if orig_size_dtype is not None else np.dtype(np.int32)
        )
        orig_size_arr: np.ndarray = np.array([1080, 1920], dtype=self._orig_size_dtype)
        self._orig_size_host = orig_size_arr
        self._orig_size_buffer = create_binding(orig_size_arr)

        scale_factor_arr: np.ndarray = np.array([1.0, 1.0], dtype=np.float32)
        self._scale_factor_host = scale_factor_arr
        self._scale_factor_buffer = create_binding(scale_factor_arr)

        self._buffers_valid = False
        self._last_transferred_shape: tuple[int, int] | None = None

        # Track current batch size for buffer reallocation
        self._current_batch_size: int = 1

        # Cache for _create_resize_args: avoid repacking kernel args for same resolution
        self._cached_resize_key: tuple[int, int, str] | None = None
        self._cached_resize_result: (
            tuple[Kernel, tuple[float, float], tuple[float, float], tuple[int, ...]] | None
        ) = None

        # Homogeneous-batch staging buffer: one contiguous H2D for same-sized images.
        self._allocated_batch_input_shape: tuple[int, int, int, int] = (1, 1080, 1920, 3)
        dummy_batch_input: np.ndarray = np.zeros(
            self._allocated_batch_input_shape,
            dtype=np.uint8,
        )
        self._batch_input_binding = create_binding(
            dummy_batch_input,
            is_input=True,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # gpu_init

    def __del__(self: Self) -> None:
        with contextlib.suppress(AttributeError):
            del self._orig_size_buffer
        with contextlib.suppress(AttributeError):
            del self._scale_factor_buffer
        with contextlib.suppress(AttributeError):
            del self._mean_buffer
        with contextlib.suppress(AttributeError):
            del self._std_buffer
        with contextlib.suppress(AttributeError):
            del self._input_binding
        with contextlib.suppress(AttributeError):
            del self._batch_input_binding

    def warmup(self: Self) -> None:
        """
        Warmup the CUDA preprocessor.

        Allocates all CUDA memory and enables future passes
        to be significantly faster.
        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["gpu_warmup"])

        rand_data: np.ndarray = np.random.default_rng().integers(
            0,
            255,
            (*self._o_shape, 3),
            dtype=np.uint8,
        )
        self.preprocess([rand_data], resize=self._resize, no_copy=True)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # gpu_warmup

    # __call__ overloads
    @overload
    def __call__(
        self: Self,
        images: np.ndarray,
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @overload
    def __call__(
        self: Self,
        images: list[np.ndarray],
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    def __call__(
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
        return self.preprocess(images, resize=resize, no_copy=no_copy, verbose=verbose)

    def _allocate_imagenet_buffers(self: Self) -> None:
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["allocate_imagenet_buffers"])

        # self._mean and self._std are set during the initialiation of ImagePreprocessor
        if self._mean is None or self._std is None:
            err_msg = "Imagenet buffers require mean/std to be set."
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # allocate_imagenet_buffers
            raise ValueError(err_msg)
        self._mean_buffer = create_binding(self._mean)
        memcpy_host_to_device_async(
            self._mean_buffer.allocation,
            self._mean,
            self._stream,
        )
        self._std_buffer = create_binding(self._std)
        memcpy_host_to_device_async(
            self._std_buffer.allocation,
            self._std,
            self._stream,
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # allocate_imagenet_buffers

    def _reallocate_input(
        self: Self,
        image: np.ndarray,
        img_shape: tuple[int, int, int],
        *,
        verbose: bool | None = None,
    ) -> None:
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["reallocate_input"])

        if verbose:
            LOG.debug(f"{self._tag}: Reallocating input bindings")
            LOG.debug(
                f"{self._tag}: Reallocation -> new shape: {img_shape}, old shape: {self._allocated_input_shape}",
            )

        self._allocated_input_shape = img_shape
        self._input_binding = create_binding(
            image,
            is_input=True,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )

        # Invalidate resize args cache: input binding allocation changed
        self._cached_resize_key = None
        self._cached_resize_result = None

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # reallocate_input

    def _validate_input(
        self: Self,
        image: np.ndarray,
        resize: str | None = None,
        *,
        verbose: bool | None = None,
    ) -> str:
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["validate_input"])

        if verbose:
            LOG.debug(f"{self._tag}: validate_input")

        # valid the method
        resize = resize if resize is not None else self._resize
        if resize not in self._valid_methods:
            err_msg = (
                f"{self._tag}: Unknown method for image resizing. Options are {self._valid_methods}"
            )
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # validate_input
            raise ValueError(err_msg)

        if image.ndim != _IMAGE_DIMENSIONS:
            err_msg = f"{self._tag}: Image must be (height, width, channels)"
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # validate_input
            raise ValueError(err_msg)

        # verified ndim is 3 above, so we can ignore the type checker
        img_shape: tuple[int, int, int] = image.shape  # type: ignore[assignment]

        if verbose:
            LOG.debug(
                f"{self._tag}: Image shape: {img_shape}, Allocated shape: {self._allocated_input_shape}",
            )

        # check if the image shape is the same as re have allocated with, if not update
        if img_shape != self._allocated_input_shape:
            # put ignore here, since we verified dmin is 3 above
            if img_shape[2] != _COLOR_CHANNELS:
                err_msg = f"{self._tag}: Can only preprocess color images."
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # validate_input
                raise ValueError(err_msg)

            self._reallocate_input(image, img_shape, verbose=verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # validate_input

        return resize

    def _reallocate_batch_input(
        self: Self,
        batch_size: int,
        height: int,
        width: int,
    ) -> None:
        """Reallocate homogeneous batch staging buffer when shape changes."""
        requested_shape = (batch_size, height, width, _COLOR_CHANNELS)
        if requested_shape == self._allocated_batch_input_shape:
            return
        self._allocated_batch_input_shape = requested_shape
        self._batch_input_binding = create_binding(
            np.zeros(requested_shape, dtype=np.uint8),
            is_input=True,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )

    @abstractmethod
    def direct_preproc(
        self: Self,
        images: list[np.ndarray],
        resize: str | None = None,
        *,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[int, list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Preprocess images for the model with H2D copies and GPU kernels.

        This method performs the complete preprocessing pipeline:
        1. Host-to-device copy of input images
        2. Resize kernels (letterbox or linear)
        3. Normalization (SST) kernel

        Parameters
        ----------
        images : list[np.ndarray]
            The images to preprocess (HWC format, uint8).
        resize : str, optional
            The resize method. Options are ['letterbox', 'linear'].
            If None, uses the configured default.
        no_warn : bool, optional
            If True, suppress warnings about usage.
        verbose : bool, optional
            Enable verbose logging.

        Returns
        -------
        tuple[int, list[tuple[float, float]], list[tuple[float, float]]]
            GPU pointer to preprocessed output, list of ratios (scale_x, scale_y),
            and list of padding (pad_x, pad_y) per image.

        """
        ...

    @property
    @abstractmethod
    def output_binding(self: Self) -> Binding:
        """
        Get the output binding for the preprocessor.

        Subclasses must implement this to return their specific output binding.

        Returns
        -------
        Binding
            The output binding containing the preprocessed data.

        """
        ...

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
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["gpu_preprocess"])

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
        output_binding = self.output_binding

        if not self._unified_mem:
            memcpy_device_to_host_async(
                output_binding.host_allocation,
                output_binding.allocation,
                self._stream,
            )

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["stream_sync"])
        stream_synchronize(self._stream)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # stream_sync

        if no_copy:
            result = (output_binding.host_allocation[:batch_size], ratios_list, padding_list)
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # gpu_preprocess
            return result

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["copy_output"])
        result = (output_binding.host_allocation[:batch_size].copy(), ratios_list, padding_list)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # copy_output
            nvtx.pop_range()  # gpu_preprocess
        return result

    @property
    def orig_size_allocation(self: Self) -> tuple[int, bool]:
        """
        Get GPU pointer and validity for orig_image_size buffer.

        Returns
        -------
        tuple[int, bool]
            The GPU pointer and validity flag.

        """
        return (self._orig_size_buffer.allocation, self._buffers_valid)

    @property
    def scale_factor_allocation(self: Self) -> tuple[int, bool]:
        """
        Get GPU pointer and validity for scale_factor buffer.

        Returns
        -------
        tuple[int, bool]
            The GPU pointer and validity flag.

        """
        return (self._scale_factor_buffer.allocation, self._buffers_valid)

    def _update_extra_buffers(
        self: Self,
        height: int,
        width: int,
        ratios: tuple[float, float],
    ) -> None:
        """
        Update GPU buffers for orig_image_size and scale_factor.

        Parameters
        ----------
        height : int
            The original image height.
        width : int
            The original image width.
        ratios : tuple[float, float]
            The scale ratios (scale_x, scale_y).

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["update_extra_buffers"])

        current_shape = (height, width)
        if self._last_transferred_shape == current_shape:
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # update_extra_buffers
            return

        # Update host arrays
        self._orig_size_host[0] = height
        self._orig_size_host[1] = width
        self._scale_factor_host[0] = ratios[0]
        self._scale_factor_host[1] = ratios[1]

        memcpy_host_to_device_async(
            self._orig_size_buffer.allocation,
            self._orig_size_host,
            self._stream,
        )
        memcpy_host_to_device_async(
            self._scale_factor_buffer.allocation,
            self._scale_factor_host,
            self._stream,
        )

        self._last_transferred_shape = current_shape
        self._buffers_valid = True

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # update_extra_buffers

    def _create_resize_args(
        self: Self,
        height: int,
        width: int,
        method: str,
        input_ptr: int,
        output_ptr: int,
        *,
        input_stride: int = 0,
        output_stride: int = 0,
        verbose: bool | None = None,
    ) -> tuple[Kernel, np.ndarray, tuple[float, float], tuple[float, float]]:
        """
        Create arguments for resize kernel.

        Parameters
        ----------
        height : int
            Input image height.
        width : int
            Input image width.
        method : str
            Resize method ('letterbox' or 'linear').
        input_ptr : int
            Device pointer to source image data.
        output_ptr : int
            Device pointer to destination image data.
        input_stride : int, optional
            Bytes between batch elements in source buffer. 0 means single image.
        output_stride : int, optional
            Bytes between batch elements in destination buffer. 0 means single image.
        verbose : bool, optional
            Enable verbose logging.

        Returns
        -------
        tuple[Kernel, np.ndarray, tuple[float, float], tuple[float, float]]
            Resize kernel, kernel args, ratios, and padding.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["create_resize_args"])

        # Check cache: for constant-resolution input, geometry is identical every frame.
        cache_key = (height, width, method)
        if cache_key == self._cached_resize_key and self._cached_resize_result is not None:
            resize_kernel, ratios, padding, geometry_args = self._cached_resize_result
        else:
            if verbose:
                LOG.debug(f"{self._tag}: create_resize_args")

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
                resize_kernel = self._letterbox_kernel
                geometry_args = (
                    width,
                    height,
                    o_width,
                    o_height,
                    padding_x,
                    padding_y,
                    new_width,
                    new_height,
                )
            else:
                if verbose:
                    LOG.debug(f"{self._tag}: Making linear args")

                ratios = (scale_x, scale_y)
                padding = (0, 0)
                resize_kernel = self._linear_kernel
                geometry_args = (
                    width,
                    height,
                    o_width,
                    o_height,
                )

            self._cached_resize_key = cache_key
            self._cached_resize_result = (resize_kernel, ratios, padding, geometry_args)

        resize_args = resize_kernel.create_args(
            input_ptr,
            output_ptr,
            *geometry_args,
            input_stride,
            output_stride,
            verbose=verbose,
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # create_resize_args

        return resize_kernel, resize_args, ratios, padding

    def _resize_single_image_to_batch(
        self: Self,
        image: np.ndarray,
        batch_buffer_ptr: int,
        resize_method: str,
        *,
        verbose: bool | None = None,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Resize a single image directly into the batch destination."""
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["resize_single"])

        height, width = image.shape[:2]
        resize_kernel, resize_args, ratios, padding = self._create_resize_args(
            height,
            width,
            resize_method,
            self._input_binding.allocation,
            batch_buffer_ptr,
            verbose=verbose,
        )
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

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # resize_single

        return ratios, padding

    def _resize_homogeneous_batch(
        self: Self,
        images: list[np.ndarray],
        batch_buffer_ptr: int,
        resize_method: str,
        *,
        verbose: bool | None = None,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Resize same-sized images with one H2D copy and one kernel launch."""
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["resize_homogeneous"])

        batch_size = len(images)
        height, width = images[0].shape[:2]

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["reallocate_batch_input"])
        self._reallocate_batch_input(batch_size, height, width)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # reallocate_batch_input
            nvtx.push_range(self._nvtx_tags["pack_batch_buffer"])

        # Pack host batch buffer once; avoid np.stack allocation in hot path.
        for i, image in enumerate(images):
            np.copyto(self._batch_input_binding.host_allocation[i], image)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # pack_batch_buffer

        if not (self._pagelocked_mem and self._unified_mem):
            memcpy_host_to_device_async(
                self._batch_input_binding.allocation,
                self._batch_input_binding.host_allocation,
                self._stream,
            )

        o_width, o_height = self._o_shape
        in_stride = height * width * _COLOR_CHANNELS
        out_stride = o_height * o_width * _COLOR_CHANNELS
        resize_kernel, resize_args, ratios, padding = self._create_resize_args(
            height,
            width,
            resize_method,
            self._batch_input_binding.allocation,
            batch_buffer_ptr,
            input_stride=in_stride,
            output_stride=out_stride,
            verbose=verbose,
        )
        self._update_extra_buffers(height, width, ratios)
        resize_kernel.call(
            (self._num_blocks[0], self._num_blocks[1], batch_size),
            self._num_threads,
            self._stream,
            resize_args,
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # resize_homogeneous

        return [ratios] * batch_size, [padding] * batch_size

    def _resize_heterogeneous_batch(
        self: Self,
        images: list[np.ndarray],
        batch_buffer_ptr: int,
        resize_method: str,
        *,
        verbose: bool | None = None,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Resize mixed-size images with direct writes to batch output."""
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["resize_heterogeneous"])

        ratios_list: list[tuple[float, float]] = []
        padding_list: list[tuple[float, float]] = []
        o_width, o_height = self._o_shape
        single_image_bytes = o_height * o_width * _COLOR_CHANNELS

        for i, image in enumerate(images):
            self._validate_input(image, resize_method, verbose=verbose)
            height, width = image.shape[:2]
            resize_kernel, resize_args, ratios, padding = self._create_resize_args(
                height,
                width,
                resize_method,
                self._input_binding.allocation,
                batch_buffer_ptr + (i * single_image_bytes),
                verbose=verbose,
            )
            if i == 0:
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
            ratios_list.append(ratios)
            padding_list.append(padding)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # resize_heterogeneous

        return ratios_list, padding_list

    def _resize_images_to_batch(
        self: Self,
        images: list[np.ndarray],
        batch_buffer_ptr: int,
        resize: str | None = None,
        *,
        verbose: bool | None = None,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Resize images and copy to batch buffer.

        Parameters
        ----------
        images : list[np.ndarray]
            Images to resize.
        batch_buffer_ptr : int
            GPU pointer to batch buffer to copy resized images to.
        resize : str, optional
            Resize method.
        verbose : bool, optional
            Enable verbose logging.

        Returns
        -------
        tuple[list[tuple[float, float]], list[tuple[float, float]]]
            Lists of ratios and padding per image.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["resize_images_to_batch"])

        if len(images) == 0:
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # resize_images_to_batch
            return [], []

        resize_method = self._validate_input(images[0], resize, verbose=verbose)
        if len(images) == 1:
            ratios, padding = self._resize_single_image_to_batch(
                images[0],
                batch_buffer_ptr,
                resize_method,
                verbose=verbose,
            )
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # resize_images_to_batch
            return [ratios], [padding]

        first_height, first_width = images[0].shape[:2]
        homogeneous = True
        for image in images[1:]:
            if image.ndim != _IMAGE_DIMENSIONS or image.shape[2] != _COLOR_CHANNELS:
                err_msg = f"{self._tag}: Image must be (height, width, channels=3)"
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # resize_images_to_batch
                raise ValueError(err_msg)
            if image.shape[:2] != (first_height, first_width):
                homogeneous = False

        if homogeneous:
            ratios_list, padding_list = self._resize_homogeneous_batch(
                images,
                batch_buffer_ptr,
                resize_method,
                verbose=verbose,
            )
        else:
            ratios_list, padding_list = self._resize_heterogeneous_batch(
                images,
                batch_buffer_ptr,
                resize_method,
                verbose=verbose,
            )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # resize_images_to_batch

        return ratios_list, padding_list
