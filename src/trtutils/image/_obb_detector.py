# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from typing import TYPE_CHECKING, overload

from trtutils._log import LOG

from ._detector import Detector
from .postprocessors import OBBDetection, get_obb_detections, postprocess_yolo_obb

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from typing_extensions import Self

    from trtutils.core._buffer import Buffer


class OBBDetector(Detector):
    """Implementation of oriented bounding box detectors (YOLO-style OBB heads)."""

    def postprocess(
        self: Self,
        outputs: list[np.ndarray],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        Postprocess the raw engine outputs.

        Parameters
        ----------
        outputs : list[np.ndarray]
            The raw outputs from the engine to postprocess.
        ratios : list[tuple[float, float]]
            The rescale ratios used during preprocessing for each image.
        padding : list[tuple[float, float]]
            The padding values used during preprocessing for each image.
        conf_thres : float, optional
            The confidence threshold to filter by.
            If not passed, will use value from constructor.
        no_copy : bool, optional
            Kept for interface symmetry, results are always new arrays.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            The postprocessed outputs per image, each containing
            [bboxes, scores, class_ids, rboxes (N, 5)].

        """
        if verbose:
            LOG.debug(f"{self._tag}: postprocess")
        t0 = time.perf_counter()
        data = postprocess_yolo_obb(
            outputs,
            ratios,
            padding,
            conf_thres or self._conf_thres,
            self._input_size,
            self._nms_iou,
            no_copy=no_copy,
            verbose=verbose,
        )
        self._post_profile = (t0, time.perf_counter())
        return data

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
            For single image: the oriented boxes of that image.
            For batch: the oriented boxes per image.
            Each is ((cx, cy, w, h, angle), score, class_id).

        """
        batch, single = self._as_batch(outputs)
        result = get_obb_detections(batch, conf_thres or self._conf_thres, verbose=verbose)
        return result[0] if single else result

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
    def end2end(  # ty: ignore[invalid-method-override]
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
            For single image: the oriented boxes of that image.
            For batch: the oriented boxes per image.

        """
        return self._end2end_core(
            images,
            verbose=verbose,
            post=self._post_fn(conf_thres, verbose=verbose),
            get=lambda pp: self.get_obbs(pp, conf_thres, verbose=verbose),
        )
