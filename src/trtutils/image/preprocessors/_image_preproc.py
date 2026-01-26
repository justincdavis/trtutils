# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from trtutils._log import LOG
from trtutils.core._bindings import create_binding
from trtutils.core._kernels import Kernel
from trtutils.core._memory import (
    memcpy_device_to_device_async,
    memcpy_host_to_device_async,
)
from trtutils.core._stream import create_stream
from trtutils.image.kernels import LETTERBOX_RESIZE, LINEAR_RESIZE

if TYPE_CHECKING:
    from typing import ClassVar

    from typing_extensions import Self

    from trtutils.compat._libs import cudart
    from trtutils.core._bindings import Binding

_COLOR_CHANNELS = 3
_IMAGE_DIMENSIONS = 3


class ImagePreprocessor(ABC):
    """Abstract base class for image preprocessors."""

    _valid_methods: ClassVar[list[str]] = ["letterbox", "linear"]

    def __init__(
        self: Self,
        output_shape: tuple[int, int],
        output_range: tuple[float, float],
        dtype: np.dtype,
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

        # compute scale, offset
        self.update_output_range(output_range)
        # setup the mean and std arrays
        if mean is not None and std is not None:
            self.update_mean_std(mean, std)

        # check resize method
        if resize not in ImagePreprocessor._valid_methods:
            err_msg = f"{self._tag}: Unknown method for image resizing. Options are {ImagePreprocessor._valid_methods}"
            raise ValueError(err_msg)
        self._resize: str = resize

        # mark setup in logs
        LOG.debug(
            f"{self._tag}: Creating preprocessor: {output_shape}, {output_range}, {dtype}",
        )

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

    @abstractmethod
    def __call__(
        self: Self,
        images: list[np.ndarray],
        resize: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @abstractmethod
    def preprocess(
        self: Self,
        images: list[np.ndarray],
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
        dtype: np.dtype,
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
        orig_size_arr: np.ndarray = np.array([1080, 1920], dtype=np.int32)
        self._orig_size_host = orig_size_arr
        self._orig_size_buffer = create_binding(orig_size_arr)

        scale_factor_arr: np.ndarray = np.array([1.0, 1.0], dtype=np.float32)
        self._scale_factor_host = scale_factor_arr
        self._scale_factor_buffer = create_binding(scale_factor_arr)

        self._buffers_valid = False
        self._last_transferred_shape: tuple[int, int] | None = None

        # Track current batch size for buffer reallocation
        self._current_batch_size: int = 1

        # Single-image resize output buffer (used before D2D copy to batch buffer)
        dummy_resize_output: np.ndarray = np.zeros(
            (self._o_shape[1], self._o_shape[0], 3),
            dtype=np.uint8,
        )
        self._resize_output_binding = create_binding(
            dummy_resize_output,
            is_input=True,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
        )

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
            del self._resize_output_binding

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
        self.preprocess([rand_data], resize=self._resize, no_copy=True)

    def __call__(
        self: Self,
        images: list[np.ndarray],
        resize: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Preprocess images for the model.

        Parameters
        ----------
        images : list[np.ndarray]
            The images to preprocess.
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
        # self._mean and self._std are set during the initialiation of ImagePreprocessor
        if self._mean is None or self._std is None:
            err_msg = "Imagenet buffers require mean/std to be set."
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

    def _reallocate_input(
        self: Self,
        image: np.ndarray,
        img_shape: tuple[int, int, int],
        *,
        verbose: bool | None = None,
    ) -> None:
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
            err_msg = (
                f"{self._tag}: Unknown method for image resizing. Options are {self._valid_methods}"
            )
            raise ValueError(err_msg)

        if image.ndim != _IMAGE_DIMENSIONS:
            err_msg = f"{self._tag}: Image must be (height, width, channels)"
            raise ValueError(err_msg)

        # verified ndim is 3 above, so we can ignore the type checker
        img_shape: tuple[int, int, int] = image.shape

        if verbose:
            LOG.debug(
                f"{self._tag}: Image shape: {img_shape}, Allocated shape: {self._allocated_input_shape}",
            )

        # check if the image shape is the same as re have allocated with, if not update
        if img_shape != self._allocated_input_shape:
            # put ignore here, since we verified dmin is 3 above
            if img_shape[2] != _COLOR_CHANNELS:
                err_msg = f"{self._tag}: Can only preprocess color images."
                raise ValueError(err_msg)

            self._reallocate_input(image, img_shape, verbose=verbose)

        return resize

    @abstractmethod
    def direct_preproc(
        self: Self,
        images: list[np.ndarray],
        resize: str | None = None,
        *,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[int, list[tuple[float, float]], list[tuple[float, float]]]: ...

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
        current_shape = (height, width)
        if self._last_transferred_shape == current_shape:
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

    def _create_resize_args(
        self: Self,
        height: int,
        width: int,
        method: str,
        *,
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
        verbose : bool, optional
            Enable verbose logging.

        Returns
        -------
        tuple[Kernel, np.ndarray, tuple[float, float], tuple[float, float]]
            Resize kernel, kernel args, ratios, and padding.

        """
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
            resize_args = resize_kernel.create_args(
                self._input_binding.allocation,
                self._resize_output_binding.allocation,
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

            ratios = (scale_x, scale_y)
            padding = (0, 0)

            resize_kernel = self._linear_kernel
            resize_args = resize_kernel.create_args(
                self._input_binding.allocation,
                self._resize_output_binding.allocation,
                width,
                height,
                o_width,
                o_height,
                verbose=verbose,
            )

        return resize_kernel, resize_args, ratios, padding

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
        ratios_list: list[tuple[float, float]] = []
        padding_list: list[tuple[float, float]] = []

        o_width, o_height = self._o_shape
        single_image_bytes = o_height * o_width * 3  # uint8

        for i, image in enumerate(images):
            # Validate input and get resize method
            resize_method = self._validate_input(image, resize, verbose=verbose)

            # Create resize arguments
            height, width = image.shape[:2]
            resize_kernel, resize_args, ratios, padding = self._create_resize_args(
                height,
                width,
                resize_method,
                verbose=verbose,
            )

            if verbose:
                LOG.debug(f"Image {i}: Ratios: {ratios}, Padding: {padding}")

            ratios_list.append(ratios)
            padding_list.append(padding)

            # Update extra GPU buffers for input schemas (use first image's values)
            if i == 0:
                self._update_extra_buffers(height, width, ratios)

            # Copy image to input binding
            if self._pagelocked_mem and self._unified_mem:
                np.copyto(self._input_binding.host_allocation, image)
            else:
                memcpy_host_to_device_async(
                    self._input_binding.allocation,
                    image,
                    self._stream,
                )

            # Run resize kernel (outputs to single-image resize_output_binding)
            resize_kernel.call(
                self._num_blocks,
                self._num_threads,
                self._stream,
                resize_args,
            )

            # D2D copy resized image to batch buffer at correct offset
            offset_bytes = i * single_image_bytes
            memcpy_device_to_device_async(
                batch_buffer_ptr + offset_bytes,
                self._resize_output_binding.allocation,
                single_image_bytes,
                self._stream,
            )

        return ratios_list, padding_list
