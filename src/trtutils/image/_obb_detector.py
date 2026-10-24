# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, overload

from ._yolo_task_model import _YoloTaskModel
from .postprocessors import get_obb_detections, postprocess_yolo_obb

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from typing_extensions import Self

    from trtutils.core._buffer import Buffer

    from .postprocessors import OBBDetection


class OBBDetector(_YoloTaskModel):
    """Implementation of oriented bounding box detectors (YOLO-style OBB heads)."""

    _output_names = ("output0",)
    _postprocess_fn = staticmethod(postprocess_yolo_obb)
    _get_fn = staticmethod(get_obb_detections)

    @overload
    def get_obbs(
        self: Self,
        outputs: list[np.ndarray],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[OBBDetection]: ...
    @overload
    def get_obbs(
        self: Self,
        outputs: list[list[np.ndarray]],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[OBBDetection]]: ...
    def get_obbs(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[OBBDetection] | list[list[OBBDetection]]:
        """
        Get the oriented boxes from postprocessed outputs.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[list[np.ndarray]]
            For single image: the postprocessed outputs of that image.
            For batch: the postprocessed outputs per image.
        conf_thres : float, optional
            The confidence threshold to filter by.
            By default None, which will use value passed during initialization.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[OBBDetection] | list[list[OBBDetection]]
            The oriented boxes of the image, or per image for a batch.
            Each is ((cx, cy, w, h, angle), score, class_id).

        """
        return self._get(outputs, conf_thres, verbose=verbose)

    @overload
    def end2end(
        self: Self,
        images: np.ndarray | Buffer,
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[OBBDetection]: ...
    @overload
    def end2end(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[OBBDetection]]: ...
    def end2end(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[OBBDetection] | list[list[OBBDetection]]:
        """
        Perform end to end inference for an image or a batch of images.

        Equivalent to running preprocess, run, postprocess, and
        get_obbs in that order.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to perform inference with.
        conf_thres : float, optional
            The confidence threshold to filter by.
            By default None, which will use value passed during initialization.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[OBBDetection] | list[list[OBBDetection]]
            The oriented boxes of the image, or per image for a batch.

        """
        return self._end2end(images, conf_thres, verbose=verbose)
