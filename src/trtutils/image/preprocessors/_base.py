# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import nvtx

from trtutils._flags import FLAGS
from trtutils._log import LOG

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import ClassVar

    from typing_extensions import Self

    from trtutils.core._buffer import Buffer


_COLOR_CHANNELS = 3
_IMAGE_DIMENSIONS = 3


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
        images: np.ndarray | Buffer,
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @overload
    def __call__(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @abstractmethod
    def __call__(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        resize: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    # preprocess overloads
    @overload
    def preprocess(
        self: Self,
        images: np.ndarray | Buffer,
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @overload
    def preprocess(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @abstractmethod
    def preprocess(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        resize: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...
