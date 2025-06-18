# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trtutils._log import LOG
from trtutils.impls.yolo._process import preprocess

from ._image_preproc import ImagePreprocessor

if TYPE_CHECKING:
    from typing_extensions import Self


class CPUPreprocessor(ImagePreprocessor):
    """CPU-based preprocessor for image processing models."""

    def __init__(
        self: Self,
        output_shape: tuple[int, int],
        output_range: tuple[float, float],
        dtype: np.dtype,
        resize: str = "letterbox",
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
        tag : str
            The tag to prefix to all logging statements made.
            By default, 'CPUPreprocessor'
            If used within a model class, will be the model tag.

        """
        self._tag = "CPUPreprocessor" if tag is None else f"{tag}.CPUPreprocessor"
        self._resize = resize

        LOG.debug(
            f"{self._tag}: Creating preprocessor: {output_shape}, {output_range}, {dtype}",
        )
        self._o_shape = output_shape
        self._o_range = output_range
        self._o_dtype = dtype

    def warmup(self: Self) -> None:
        """Compatibility function for CPU/CUDA parity."""
        rand_data: np.ndarray = np.random.default_rng().integers(
            0,
            255,
            (*self._o_shape, 3),
            dtype=np.uint8,
        )
        self.preprocess(rand_data)

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
        tuple[np.ndarray, tuple[float, float], tuple[float, float]]
            The preprocessed image, ratios, and padding used for resizing.

        """
        resize = resize if resize is not None else self._resize
        return self.preprocess(image, resize=resize, no_copy=no_copy, verbose=verbose)

    def preprocess(
        self: Self,
        image: np.ndarray,
        resize: str | None = None,
        *,
        no_copy: bool | None = None,  # noqa: ARG002
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        """
        Preprocess an image for the model.

        Parameters
        ----------
        image : np.ndarray
            The image to preprocess.
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
        tuple[np.ndarray, tuple[float, float], tuple[float, float]]
            The preprocessed image, ratios, and padding used for resizing.

        """
        resize = resize if resize is not None else self._resize
        return preprocess(
            image,
            self._o_shape,
            self._o_dtype,
            self._o_range,
            resize,
            verbose=verbose,
        )
