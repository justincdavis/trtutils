# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from ._abc import DetectionPostprocessor
from ._process import postprocess_detections

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


class CPUDetectionPostprocessor(DetectionPostprocessor):
    """CPU-based detection postprocessor."""

    def __init__(
        self: Self,
        tag: str | None = None,
    ) -> None:
        """
        Create a CPUDetectionPostprocessor.

        Parameters
        ----------
        tag : str, optional
            The tag to prefix to all logging statements made.
            By default, 'CPUDetectionPostprocessor'

        """
        super().__init__(tag)

    def postprocess(
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
            Whether to copy the output to the host.
            For CUDA parity, this is ignored.
        verbose : bool, optional
            Whether to log additional information.

        Returns
        -------
        np.ndarray
            The postprocessed detections.

        """
        return postprocess_detections(
            boxes, ratios, padding, conf_thres, no_copy=no_copy, verbose=verbose
        )
