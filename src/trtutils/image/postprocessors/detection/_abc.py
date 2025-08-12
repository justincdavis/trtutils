# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from trtutils._log import LOG

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


class DetectionPostprocessor(ABC):
    """Abstract base class for detection postprocessors."""

    def __init__(
        self: Self,
        tag: str | None = None,
    ) -> None:
        """
        Create a DetectionPostprocessor.

        Parameters
        ----------
        tag : str
            The tag to prefix to all logging statements made.
            By default, 'DetectionPostprocessor'
            If used within a model class, will be the model tag.

        """
        # tag
        self._tag = tag

        # mark setup in logs
        LOG.debug(
            f"{self._tag}: Creating postprocessor",
        )

    def __call__(
        self: Self,
        boxes: np.ndarray,
        ratios: tuple[float, float],
        padding: tuple[float, float],
        conf_thres: float | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> np.ndarray:
        """
        Postprocess the detections.

        Parameters
        ----------
        boxes : np.ndarray
            The boxes to postprocess.
        ratios : tuple[float, float]
            The ratios used to scale the boxes.
        padding : tuple[float, float]
            The padding used to scale the boxes.
        conf_thres : float, optional
            The confidence threshold to use.
        no_copy : bool, optional
            If True, the outputs will not be copied out
            from the CUDA allocated host memory (if CUDA).
            Instead, the host memory will be returned directly.
            This memory WILL BE OVERWRITTEN INPLACE
            by future postprocessing calls.
        verbose : bool, optional
            Whether to log additional information.

        Returns
        -------
        np.ndarray
            The postprocessed detections.

        """
        return self.postprocess(
            boxes, ratios, padding, conf_thres, no_copy=no_copy, verbose=verbose
        )

    @abstractmethod
    def postprocess(
        self: Self,
        boxes: np.ndarray,
        ratios: tuple[float, float],
        padding: tuple[float, float],
        conf_thres: float | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> np.ndarray: ...
