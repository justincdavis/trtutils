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
from trtutils.core._memory import memcpy_host_to_device_async
from trtutils.core._stream import create_stream
from trtutils.image.kernels import LETTERBOX_RESIZE, LINEAR_RESIZE

if TYPE_CHECKING:
    import contextlib
    from typing import ClassVar

    from typing_extensions import Self

    with contextlib.suppress(Exception):
        try:
            import cuda.bindings.runtime as cudart
        except (ImportError, ModuleNotFoundError):
            from cuda import cudart

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

    def update_mean_std(self: Self, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
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
        if len(mean) != 3:
            err_msg = f"{self._tag}: Mean must be a tuple of 3 floats"
            raise ValueError(err_msg)
        if len(std) != 3:
            err_msg = f"{self._tag}: Std must be a tuple of 3 floats"
            raise ValueError(err_msg)
        self._mean = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)

    @abstractmethod
    def warmup(self: Self) -> None: ...

    @abstractmethod
    def __call__(
        self: Self,
        image: np.ndarray,
        resize: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]: ...

    @abstractmethod
    def preprocess(
        self: Self,
        image: np.ndarray,
        resize: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]: ...


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
        self._linear_kernel = Kernel(*LINEAR_RESIZE)
        self._letterbox_kernel = Kernel(*LETTERBOX_RESIZE)

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
        return self.preprocess(image, resize=resize, no_copy=no_copy, verbose=verbose)

    def _allocate_imagenet_buffers(self: Self) -> None:
        # self._mean and self._std are set during the initialiation of ImagePreprocessor
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
            err_msg = f"{self._tag}: Unknown method for image resizing. Options are {self._valid_methods}"
            raise ValueError(err_msg)

        if image.ndim != _IMAGE_DIMENSIONS:
            err_msg = f"{self._tag}: Image must be (height, width, channels)"
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
                raise ValueError(err_msg)

            self._reallocate_input(image, img_shape, verbose=verbose)

        return resize

    @abstractmethod
    def direct_preproc(
        self: Self,
        image: np.ndarray,
        resize: str | None = None,
        *,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[int, tuple[float, float], tuple[float, float]]: ...

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
