# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from cv2ext.bboxes import nms
from cv2ext.image import patch as patch_image

from trtutils._log import LOG

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.image.interfaces import DetectorInterface


class SAHI:
    """Simple implementation of SAHI."""

    def __init__(
        self: Self,
        detector: DetectorInterface,
        slice_size: tuple[int, int] | None = None,
        slice_overlap: tuple[float, float] = (0.2, 0.2),
        iou_threshold: float = 0.5,
        *,
        agnostic_nms: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the SAHI wrapper.

        Parameters
        ----------
        detector: DetectorInterface
            The detector to wrap.
        slice_size: tuple[int, int] | None, optional
            The size of the slices to take.
            If None, the slice size will be the detectors input size.
            By default, None.
        slice_overlap: tuple[float, float], optional
            The overlap between slices.
            By default, (0.2, 0.2).
        iou_threshold: float, optional
            The IoU threshold for non-maximum suppression.
            By default, 0.5.
        agnostic_nms: bool, optional
            Whether to use agnostic NMS.
            By default, False.
        verbose: bool, optional
            Whether to print verbose output.
            By default, False.

        """
        self._detector = detector
        slice_size = slice_size if slice_size is not None else detector.input_shape
        self._slice_width = slice_size[0]
        self._slice_height = slice_size[1]
        self._slice_overlap = slice_overlap
        self._iou_threshold = iou_threshold
        self._agnostic_nms = agnostic_nms
        self._verbose = verbose

        # allocate the executor to process contiguous patchs in parallel
        cpu_cores = os.cpu_count()
        if cpu_cores is None:
            cpu_cores = 1
        else:
            cpu_cores -= 1
        self._executor = ThreadPoolExecutor(max_workers=cpu_cores)
        self._lock = Lock()

    def _execute(
        self: Self,
        patch: np.ndarray,
        offset: tuple[int, int],
        scale: tuple[float, float],
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        if not patch.flags.c_contiguous:
            patch = np.ascontiguousarray(patch)
        with self._lock:
            dets = self._detector.end2end(
                patch,
                conf_thres=conf_thres,
                nms_iou_thres=nms_iou_thres,
                extra_nms=extra_nms,
                agnostic_nms=agnostic_nms,
            )
        x, y = offset
        sx, sy = scale
        corrected_dets = []
        for det in dets:
            bbox, conf, class_id = det
            x1, y1, x2, y2 = bbox
            # offset based on patch
            x1 += x
            y1 += y
            x2 += x
            y2 += y
            # scale based on what patches are generated on
            x1 = int(x1 * sx)
            y1 = int(y1 * sy)
            x2 = int(x2 * sx)
            y2 = int(y2 * sy)
            # write back
            corrected_dets.append(((x1, y1, x2, y2), conf, class_id))
        return corrected_dets

    def end2end(
        self: Self,
        image: np.ndarray,
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Perform end to end inference using detection model and SAHI.

        Parameters
        ----------
        image : np.ndarray
            The image to perform inference with.
        conf_thres : float, optional
            The confidence threshold with which to retrieve bounding boxes.
            By default None
        nms_iou_thres : float
            The IOU threshold to use during the optional/additional
            NMS operation. By default, None which will use value
            provided during initialization.
        extra_nms : bool, optional
            Whether or not to perform an additional NMS operation.
            By default None, which will use value provided during
            initialization.
        agnostic_nms: bool, optional
            Whether or not to perform class-agnostic NMS for the
            optional/additional operation. By default None, which
            will use value provided during initialization.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[tuple[tuple[int, int, int, int], float, int]]
            The detections where each entry is bbox, conf, class_id

        """
        if verbose is None:
            verbose = self._verbose

        patches, offsets, (nw, nh) = patch_image(
            image, (self._slice_width, self._slice_height), overlap=self._slice_overlap
        )
        height, width = image.shape[:2]
        sx = width / nw
        sy = height / nh

        futures = []
        for patch, offset in zip(patches, offsets):
            futures.append(
                self._executor.submit(
                    self._execute,
                    patch,
                    offset,
                    (sx, sy),
                    conf_thres,
                    nms_iou_thres,
                    extra_nms=extra_nms,
                    agnostic_nms=agnostic_nms,
                )
            )
        detections = []
        for future in futures:
            sub_dets = future.result()
            if verbose:
                LOG.info(f"SAHI: {len(sub_dets)} detections in slice {offset}")
            detections.extend(sub_dets)

        dets = nms(detections, self._iou_threshold, agnostic=self._agnostic_nms)

        if verbose:
            LOG.info(f"SAHI: {len(dets)} detections overall")

        return dets
