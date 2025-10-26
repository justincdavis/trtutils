# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Object distribution extraction from historical frames.

Classes
-------
ObjectDistributionExtractor
    Extracts spatial and size distributions of objects using oracle detector.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trtutils._log import LOG

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.image.interfaces import DetectorInterface


class ObjectDistributionExtractor:
    """Extracts object size distributions from historical frames using oracle detector."""

    def __init__(self: Self, oracle: DetectorInterface, num_bins: int = 12) -> None:
        """
        Initialize the distribution extractor.

        Parameters
        ----------
        oracle : DetectorInterface
            High-accuracy detector for labeling frames.
        num_bins : int, optional
            Number of size bins, by default 12.

        """
        self.oracle = oracle
        self.num_bins = num_bins

    def extract(
        self: Self,
        frames: list[np.ndarray],
        *,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Extract object distribution from frames.

        Parameters
        ----------
        frames : list[np.ndarray]
            Historical frames to analyze.
        verbose : bool, optional
            Whether to log information, by default False.

        Returns
        -------
        np.ndarray
            Normalized distribution vector of length num_bins.

        """
        if verbose:
            LOG.info(f"Extracting object distribution from {len(frames)} frames")

        all_distributions = []
        for i, frame in enumerate(frames):
            if verbose and i % 10 == 0:
                LOG.debug(f"Processing frame {i+1}/{len(frames)}")

            detections = self.oracle.end2end(frame, verbose=False)
            dist = self._compute_size_distribution(detections)
            all_distributions.append(dist)

        # Aggregate distributions
        if len(all_distributions) == 0:
            return np.zeros(self.num_bins)

        aggregated = self._aggregate_distributions(all_distributions)

        if verbose:
            LOG.info(f"Aggregated distribution: {aggregated}")

        return aggregated

    def extract_spatial(
        self: Self,
        frames: list[np.ndarray],
        grid_size: tuple[int, int] = (2, 2),
        *,
        verbose: bool = False,
    ) -> dict[tuple[int, int], np.ndarray]:
        """
        Extract spatial object distributions across image regions.

        Parameters
        ----------
        frames : list[np.ndarray]
            Historical frames to analyze.
        grid_size : tuple[int, int], optional
            Grid dimensions (rows, cols), by default (2, 2).
        verbose : bool, optional
            Whether to log information, by default False.

        Returns
        -------
        dict[tuple[int, int], np.ndarray]
            Distribution vector for each grid cell (row, col).

        """
        if verbose:
            LOG.info(
                f"Extracting spatial distributions with grid {grid_size} "
                f"from {len(frames)} frames",
            )

        rows, cols = grid_size
        spatial_dists: dict[tuple[int, int], list[np.ndarray]] = {
            (r, c): [] for r in range(rows) for c in range(cols)
        }

        for i, frame in enumerate(frames):
            if verbose and i % 10 == 0:
                LOG.debug(f"Processing frame {i+1}/{len(frames)}")

            h, w = frame.shape[:2]
            detections = self.oracle.end2end(frame, verbose=False)

            # Bin detections into grid cells
            for (x1, y1, x2, y2), _, _ in detections:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                grid_row = min(int(cy * rows / h), rows - 1)
                grid_col = min(int(cx * cols / w), cols - 1)

                area = (x2 - x1) * (y2 - y1)
                bin_idx = min(int(np.log2(area + 1)), self.num_bins - 1)

                # Create one-hot distribution for this detection
                dist = np.zeros(self.num_bins)
                dist[bin_idx] = 1.0

                spatial_dists[(grid_row, grid_col)].append(dist)

        # Aggregate each cell's distributions
        result = {}
        for cell, dists in spatial_dists.items():
            if len(dists) > 0:
                result[cell] = self._aggregate_distributions(dists)
            else:
                result[cell] = np.zeros(self.num_bins)

        return result

    def _compute_size_distribution(
        self: Self,
        detections: list[tuple[tuple[int, int, int, int], float, int]],
    ) -> np.ndarray:
        """
        Compute size distribution from detections.

        Parameters
        ----------
        detections : list[tuple[tuple[int, int, int, int], float, int]]
            Detections as (bbox, confidence, class).

        Returns
        -------
        np.ndarray
            Normalized distribution vector.

        """
        bins = np.zeros(self.num_bins)

        for (x1, y1, x2, y2), _, _ in detections:
            area = (x2 - x1) * (y2 - y1)
            bin_idx = min(int(np.log2(area + 1)), self.num_bins - 1)
            bins[bin_idx] += 1

        # Normalize
        total = np.sum(bins)
        if total > 0:
            bins = bins / total

        return bins

    def _aggregate_distributions(
        self: Self,
        distributions: list[np.ndarray],
    ) -> np.ndarray:
        """
        Aggregate multiple distributions.

        Parameters
        ----------
        distributions : list[np.ndarray]
            List of distribution vectors.

        Returns
        -------
        np.ndarray
            Aggregated and normalized distribution.

        """
        if len(distributions) == 0:
            return np.zeros(self.num_bins)

        # Average across all distributions
        stacked = np.stack(distributions)
        aggregated = np.mean(stacked, axis=0)

        # Ensure normalized
        total = np.sum(aggregated)
        if total > 0:
            aggregated = aggregated / total

        return aggregated

