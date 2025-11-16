# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from cv2ext.tracking.trackers import KLTMultiTracker

if TYPE_CHECKING:
    from typing_extensions import Self


class FlexPatch:

    def __init__(self: Self,
                detectors: list[tuple[int, Callable[[np.ndarray], list[tuple[tuple[int, int, int, int], float, int]]]]],
            ) -> None:
        self._dets = detectors
        self._tracker = KLTMultiTracker()
        self._boxes: list[tuple[int, int, int, int]] = []

    def _get_closest_detector(self: Self, fsize: int) -> Callable[[np.ndarray], list[tuple[tuple[int, int, int, int], float, int]]]:
        idx = len(self._dets) - 1
        for i in range(len(self._dets) - 2, 0, -1):
            dsize, _ = self._dets[i]
            if dsize > fsize:
                idx = i
            else:
                break
        return self._dets[idx][1]

    @staticmethod
    def _pack_boxes(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:


    def run(
        self,
        frame: np.ndarray,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Run FlexPatch on the next frame in a sequence.

        Parameters
        ----------
        frame : np.ndarray
            The next frame in a sequence.

        Returns
        -------
        list[tuple[tuple[int, int, int, int], float, int]]
            The detections

        """
        # two cases, we have some boxes or we do not
        # have no existing boxes so must run detector
        if len(self._boxes) == 0:
            # use largest detector, get best accuracy to start
            det = self._dets[-1][1]
            dets = det(frame)
            bboxes = [b for b, _, _ in dets]
            self._tracker.init(frame, bboxes)
            self._bboxes = bboxes
            return dets

        # otherwise we have some detections so need to be more intelligent
        # 1: track the existing bounding boxes on the new frame
        # 2: compute the metrics from the paper across each bounding box
        #   a: minimum eigenvalue of spatial gradient matrix
        #   b: ncc between bboxes
        #   c: acceleration, difference between velocities
        #   d: standard deviation of optical flow errors
        #   e: confidence score from detector on that bounding box
        # 3: Based on those feature, predict priority of each patch
        #    this uses the decision tree classifier, which estimates IoU
        #    IoU: 0 high, 0-0.5 medium, >0.5 low
        # 4: extract the patches, we get a patch for each bounding box
        #    each patch has padding equal to height and width (so 1/2 per side)
        # 5: 
