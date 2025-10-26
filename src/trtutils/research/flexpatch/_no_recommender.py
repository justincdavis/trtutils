# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
New-object patch recommender.

Classes
-------
NewObjectRecommender
    Recommends patches where new objects might appear.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self


class NewObjectRecommender:
    """Recommends patches for new object detection."""

    def __init__(
        self: Self,
        frame_size: tuple[int, int],
        cell_size: tuple[int, int] = (20, 22),
        edge_threshold: float = 0.1,
        refresh_weight: int = 50,
        edge_weight: int = 10,
    ) -> None:
        """
        Initialize the new object recommender.

        Parameters
        ----------
        frame_size : tuple[int, int]
            Size of frames as (width, height).
        cell_size : tuple[int, int], optional
            Size of each cell as (width, height), by default (20, 22).
        edge_threshold : float, optional
            Ratio of edge pixels required to consider a cell, by default 0.1.
        refresh_weight : int, optional
            Maximum refresh interval weight, by default 50.
        edge_weight : int, optional
            Weight for cells with high edge density, by default 10.

        """
        self.frame_size = frame_size
        self.cell_size = cell_size
        self.edge_threshold = edge_threshold
        self.refresh_weight = refresh_weight
        self.edge_weight = edge_weight
        
        # Create grid
        self.n_cols = (frame_size[0] + cell_size[0] - 1) // cell_size[0]
        self.n_rows = (frame_size[1] + cell_size[1] - 1) // cell_size[1]
        
        # Refresh interval map
        self.refresh_map = np.zeros((self.n_rows, self.n_cols), dtype=np.int32)

    def recommend(
        self: Self,
        frame: np.ndarray,
        exclusion_bboxes: list[tuple[int, int, int, int]] | None = None,
        min_priority: int = 20,
    ) -> list[tuple[tuple[int, int, int, int], int, str]]:
        """
        Recommend patches for new object detection.

        Parameters
        ----------
        frame : np.ndarray
            The current frame.
        exclusion_bboxes : list[tuple[int, int, int, int]], optional
            Bounding boxes to exclude from recommendations.
        min_priority : int, optional
            Minimum priority threshold for recommendations, by default 20.

        Returns
        -------
        list[tuple[tuple[int, int, int, int], int, str]]
            List of (bbox, priority_score, type) tuples.
            type is always "new-object".

        """
        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Compute adaptive threshold for Canny
        thresh_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(gray, int(0.5 * thresh_val), int(1.5 * thresh_val))
        
        # Create exclusion mask
        exclusion_mask = np.zeros((self.n_rows, self.n_cols), dtype=bool)
        if exclusion_bboxes:
            for bbox_x, bbox_y, bbox_w, bbox_h in exclusion_bboxes:
                # Convert bbox to cell coordinates
                cell_x1 = max(0, bbox_x // self.cell_size[0])
                cell_y1 = max(0, bbox_y // self.cell_size[1])
                cell_x2 = min(self.n_cols, (bbox_x + bbox_w + self.cell_size[0] - 1) // self.cell_size[0])
                cell_y2 = min(self.n_rows, (bbox_y + bbox_h + self.cell_size[1] - 1) // self.cell_size[1])
                
                exclusion_mask[cell_y1:cell_y2, cell_x1:cell_x2] = True
        
        # Compute cell priorities
        cell_priorities = np.zeros((self.n_rows, self.n_cols), dtype=np.int32)
        
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                if exclusion_mask[row, col]:
                    continue
                
                # Get cell region
                y1 = row * self.cell_size[1]
                x1 = col * self.cell_size[0]
                y2 = min(y1 + self.cell_size[1], frame.shape[0])
                x2 = min(x1 + self.cell_size[0], frame.shape[1])
                
                cell_edges = edges[y1:y2, x1:x2]
                
                # Compute edge intensity (EI)
                edge_pixels = np.sum(cell_edges > 0)
                total_pixels = cell_edges.size
                edge_ratio = edge_pixels / total_pixels if total_pixels > 0 else 0
                
                # Get refresh interval (RI)
                ri = self.refresh_map[row, col]
                
                # Compute priority
                priority = min(self.refresh_weight, ri)
                if edge_ratio > self.edge_threshold:
                    priority += self.edge_weight
                
                cell_priorities[row, col] = priority
                
                # Increment refresh interval
                self.refresh_map[row, col] += 1
        
        # Extract patches from high-priority cells
        patches = []
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                priority = cell_priorities[row, col]
                
                if priority >= min_priority:
                    # Convert cell to bbox
                    x = col * self.cell_size[0]
                    y = row * self.cell_size[1]
                    w = min(self.cell_size[0], self.frame_size[0] - x)
                    h = min(self.cell_size[1], self.frame_size[1] - y)
                    
                    bbox = (x, y, w, h)
                    patches.append((bbox, int(priority), "new-object"))
        
        return patches

    def reset_region(
        self: Self,
        bbox: tuple[int, int, int, int],
    ) -> None:
        """
        Reset refresh interval for a region (called after detection).

        Parameters
        ----------
        bbox : tuple[int, int, int, int]
            Bounding box as (x, y, w, h) to reset.

        """
        x, y, w, h = bbox
        
        # Convert to cell coordinates
        cell_x1 = max(0, x // self.cell_size[0])
        cell_y1 = max(0, y // self.cell_size[1])
        cell_x2 = min(self.n_cols, (x + w + self.cell_size[0] - 1) // self.cell_size[0])
        cell_y2 = min(self.n_rows, (y + h + self.cell_size[1] - 1) // self.cell_size[1])
        
        self.refresh_map[cell_y1:cell_y2, cell_x1:cell_x2] = 0

