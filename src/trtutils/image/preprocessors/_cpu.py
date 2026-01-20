# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._image_preproc import ImagePreprocessor
from ._process import preprocess

if TYPE_CHECKING:
    from typing_extensions import Self

_COLOR_CHANNELS = 3


class CPUPreprocessor(ImagePreprocessor):
    """CPU-based preprocessor for image processing models."""

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
        Create a CPUPreprocessor for image processing models.

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
            The method to resize the image with.
            Options are [letterbox, linear], will use method
            provided in constructor by default.
        mean : tuple[float, float, float], optional
            The mean to subtract from the image.
            By default, None, which will not subtract any mean.
        std : tuple[float, float, float], optional
            The standard deviation to divide the image by.
            By default, None, which will not divide by any standard deviation.
        tag : str
            The tag to prefix to all logging statements made.
            By default, 'CPUPreprocessor'
            If used within a model class, will be the model tag.

        """
        tag = "CPUPreprocessor" if tag is None else f"{tag}.CPUPreprocessor"
        super().__init__(
            output_shape=output_shape,
            output_range=output_range,
            dtype=dtype,
            resize=resize,
            mean=mean,
            std=std,
            tag=tag,
        )

    def warmup(self: Self) -> None:
        """Compatibility function for CPU/CUDA parity."""
        rand_data: np.ndarray = np.random.default_rng().integers(
            0,
            255,
            (*self._o_shape, 3),
            dtype=np.uint8,
        )
        self.preprocess([rand_data])

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
        resize : str
            The method to resize the image with.
            By default letterbox, options are [letterbox, linear]
        no_copy : bool, optional
            Compatibility parameter for CUDA parity.
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

    def preprocess(
        self: Self,
        images: list[np.ndarray],
        resize: str | None = None,
        *,
        no_copy: bool | None = None,  # noqa: ARG002
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Preprocess images for the model.

        Parameters
        ----------
        images : list[np.ndarray]
            The images to preprocess.
        resize : str
            The method to resize the image with.
            By default letterbox, options are [letterbox, linear]
        no_copy : bool, optional
            Compatibility parameter for CUDA parity.
        verbose : bool, optional
            Whether or not to output additional information
            to stdout. If not provided, will default to overall
            engines verbose setting.

        Returns
        -------
        tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]
            The preprocessed batch tensor, list of ratios, and list of padding per image.

        """
        resize = resize if resize is not None else self._resize
        mean = self._mean
        std = self._std
        mean_tuple: tuple[float, float, float] | None = None
        std_tuple: tuple[float, float, float] | None = None
        if mean is not None:
            mean_tuple = tuple(
                mean.reshape(-1)
                if mean.size == _COLOR_CHANNELS
                else mean.flatten()[:_COLOR_CHANNELS]
            )
        if std is not None:
            std_tuple = tuple(
                std.reshape(-1) if std.size == _COLOR_CHANNELS else std.flatten()[:_COLOR_CHANNELS]
            )
        return preprocess(
            images,
            self._o_shape,
            self._o_dtype,
            self._o_range,
            resize,
            mean_tuple,
            std_tuple,
            verbose=verbose,
        )
