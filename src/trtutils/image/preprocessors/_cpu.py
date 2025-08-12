# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from typing import TYPE_CHECKING

from ._abc import ImagePreprocessor
from ._process import preprocess

if TYPE_CHECKING:
    import numpy as np
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
        tag = "CPUPreprocessor" if tag is None else f"{tag}.CPUPreprocessor"
        super().__init__(
            output_shape,
            output_range,
            dtype,
            resize,
            tag,
        )

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
