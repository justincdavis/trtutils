# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self


class PatchRecommender:
    """The patch recommenderation methodology as described by the FlexPatch paper."""

    def __init__(
            self: Self,
            frame_size: tuple[int, int],
            cell_size: tuple[int, int] = (20, 22),
            edge_threshold: float = 0.1,
            refresh_interval: int = 50,
        ) -> None:
        """
        Create an instance of PatchRecommender.

        Parameters
        ----------
        frame_size : tuple[int, int]
            The size of the frames in the sequence to recommend patches on.
            Frame size should be supplied in (width, height) form
        cell_size : tuple[int, int], optional
            The size of each cell portion to recommend or not.
            The size corresponsed to (width, height)
            Default is (20, 22)
        edge_threshold : float, optional
            The ratio of pixels within a cell which are marked as edges.
            The higher the value (up to one) the fewer cells
            will be marked for detection.
            Default is 0.1
        refresh_interval : int, optional
            The maximum amount to clip refresh interval to.
            Helps to balance the priority of recommendation.
            Default is 50

        """
        self._fsize = frame_size
        self._csize = cell_size
        self._e_thresh = edge_threshold
        self._weight = refresh_interval

        # create a grid from the cell size and frame size
        self._n_cols = math.ceil(self._fsize[0] / self._csize[0])
        self._n_rows = math.ceil(self._fsize[1] / self._csize[1])

        # step size between grid cells
        self._col_step = (
            int((self._fsize[0] - self._csize[0]) / (self._n_cols - 1))
            if self._n_cols > 1
            else self._fsize[0]
        )
        self._row_step = (
            int((self._fsize[1] - self._csize[1]) / (self._n_rows - 1))
            if self._n_rows > 1
            else self._fsize[1]
        )

        # valid cell ids, and detection count info
        self._cells = {(i, j) for j in range(self._n_cols) for i in range(self._n_rows)}
        self._grid = np.zeros((self._n_rows, self._n_cols), dtype=int)

    def run(self: Self, frame: np.ndarray, exclusions: list[tuple[int, int, int, int]] | None = None) -> list[tuple[tuple[int, int, int, int], int]]:
        """
        Get the list of recommended patches.

        Parameters
        ----------
        frame : np.ndarray
            The frame to get recommendations on.
        exclusions : list[tuple[int, int, int, int]], optional
            The list of bounding boxes to exclude from the recommendation process.
            Any cells within a bounding box will not be included.

        Returns
        -------
        list[tuple[tuple[int, int, int, int], int]]
            The list of patch recommendations as a bbox and priority level

        """
        # Step 0: get the set of cells which we can evaluate based on the exclusions
        

        # Step 1: compute Canny edge detection and sum edge pixels per cell block
        gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred: np.ndarray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges: np.ndarray = cv2.Canny(gray, int(0.5 * thresh), int(1.5 * thresh))

        
