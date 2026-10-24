# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, overload

from ._yolo_task_model import _YoloTaskModel
from .postprocessors import get_segmentations, postprocess_yolo_seg

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from typing_extensions import Self

    from trtutils.core._buffer import Buffer

    from .postprocessors import Segmentation


class Segmenter(_YoloTaskModel):
    """Implementation of instance segmentation models (YOLO-style ``output0`` + ``output1`` prototypes)."""

    _output_names = ("output0", "output1")
    _postprocess_fn = staticmethod(postprocess_yolo_seg)
    _get_fn = staticmethod(get_segmentations)
    _needs_input_size = True

    @overload
    def get_segmentations(
        self: Self,
        outputs: list[np.ndarray],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[Segmentation]: ...
    @overload
    def get_segmentations(
        self: Self,
        outputs: list[list[np.ndarray]],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[Segmentation]]: ...
    def get_segmentations(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[Segmentation] | list[list[Segmentation]]:
        """
        Get the segmentations from postprocessed outputs.

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
        list[Segmentation] | list[list[Segmentation]]
            The segmentations of the image, or per image for a batch.
            Each is ((x1, y1, x2, y2), score, class_id, mask).

        """
        return self._get(outputs, conf_thres, verbose=verbose)

    @overload
    def end2end(
        self: Self,
        images: np.ndarray | Buffer,
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[Segmentation]: ...
    @overload
    def end2end(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[Segmentation]]: ...
    def end2end(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[Segmentation] | list[list[Segmentation]]:
        """
        Perform end to end inference for an image or a batch of images.

        Equivalent to running preprocess, run, postprocess, and
        get_segmentations in that order.

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
        list[Segmentation] | list[list[Segmentation]]
            The segmentations of the image, or per image for a batch.

        """
        return self._end2end(images, conf_thres, verbose=verbose)
