# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Selective execution with AIMD partition selection.

Classes
-------
SelectiveExecutor
    Executes selective partitions with AIMD-based block skipping.

"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from trtutils._log import LOG

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.image.interfaces import DetectorInterface

    from ._plan import PartitionPlan


class SelectiveExecutor:
    """Online runtime controller for executing selective partitions."""

    def __init__(
        self: Self,
        detectors: dict[str, DetectorInterface],
        aimd_increase: int = 1,
        aimd_decrease_factor: float = 0.5,
        nms_iou_thresh: float = 0.5,
    ) -> None:
        """
        Initialize selective executor.

        Parameters
        ----------
        detectors : dict[str, DetectorInterface]
            Mapping of detector names to instances.
        aimd_increase : int, optional
            Additive increase for skip window, by default 1.
        aimd_decrease_factor : float, optional
            Multiplicative decrease factor, by default 0.5.
        nms_iou_thresh : float, optional
            IoU threshold for NMS merging, by default 0.5.

        """
        self.detectors = detectors
        self.aimd_increase = aimd_increase
        self.aimd_decrease_factor = aimd_decrease_factor
        self.nms_iou_thresh = nms_iou_thresh

        # Track skip windows per block (key: block coords tuple)
        self.skip_windows: dict[tuple[int, int, int, int], int] = defaultdict(int)

        # Track previous detections per block
        self.prev_detections: dict[tuple[int, int, int, int], int] = defaultdict(int)

    def execute(
        self: Self,
        frame: np.ndarray,
        plan: PartitionPlan,
        *,
        verbose: bool = False,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Execute selective inference on frame using partition plan.

        Parameters
        ----------
        frame : np.ndarray
            Input frame.
        plan : PartitionPlan
            Partition plan to execute.
        verbose : bool, optional
            Whether to log information, by default False.

        Returns
        -------
        list[tuple[tuple[int, int, int, int], float, int]]
            Merged detections as (bbox, confidence, class).

        """
        all_detections = []

        for block in plan.blocks:
            # Check if should skip this block
            if self._should_skip_block(block, verbose=verbose):
                # Decrease skip window
                coords = block.coords
                if self.skip_windows[coords] > 0:
                    self.skip_windows[coords] -= 1
                continue

            # Get detector for this block
            if block.detector_name is None:
                continue

            if block.detector_name not in self.detectors:
                if verbose:
                    LOG.warning(f"Detector {block.detector_name} not found")
                continue

            detector = self.detectors[block.detector_name]

            # Crop block region
            x1, y1, x2, y2 = block.coords
            cropped = frame[y1:y2, x1:x2]

            # Run detection
            try:
                detections = detector.end2end(cropped, verbose=False)
            except Exception as e:
                if verbose:
                    LOG.warning(f"Detection failed for block {block.coords}: {e}")
                continue

            # Adjust coordinates to global frame
            adjusted = []
            for (bx1, by1, bx2, by2), conf, cls in detections:
                global_bbox = (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
                adjusted.append((global_bbox, conf, cls))

            all_detections.extend(adjusted)

            # Update AIMD feedback
            self._update_aimd_feedback(block, len(detections))

        # Merge detections using NMS
        merged = self._merge_detections(all_detections)

        if verbose:
            LOG.debug(
                f"Executed {plan.num_blocks} blocks, "
                f"got {len(all_detections)} detections, "
                f"merged to {len(merged)}",
            )

        return merged

    def _should_skip_block(
        self: Self,
        block,
        *,
        verbose: bool = False,
    ) -> bool:
        """
        Determine if block should be skipped based on AIMD.

        Parameters
        ----------
        block : PartitionBlock
            Block to check.
        verbose : bool, optional
            Whether to log information, by default False.

        Returns
        -------
        bool
            True if block should be skipped.

        """
        coords = block.coords
        skip_window = self.skip_windows[coords]

        should_skip = skip_window > 0

        if verbose and should_skip:
            LOG.debug(f"Skipping block {coords}, window={skip_window}")

        return should_skip

    def _update_aimd_feedback(
        self: Self,
        block,
        num_detections: int,
    ) -> None:
        """
        Update AIMD skip window based on detection results.

        Parameters
        ----------
        block : PartitionBlock
            Block that was executed.
        num_detections : int
            Number of detections found.

        """
        coords = block.coords

        if num_detections == 0:
            # No detections: additive increase
            self.skip_windows[coords] += self.aimd_increase
        else:
            # Detections found: multiplicative decrease (reset)
            self.skip_windows[coords] = int(
                self.skip_windows[coords] * self.aimd_decrease_factor,
            )

        self.prev_detections[coords] = num_detections

    def _merge_detections(
        self: Self,
        detections: list[tuple[tuple[int, int, int, int], float, int]],
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Merge overlapping detections using NMS.

        Parameters
        ----------
        detections : list[tuple[tuple[int, int, int, int], float, int]]
            Detections to merge.

        Returns
        -------
        list[tuple[tuple[int, int, int, int], float, int]]
            Merged detections.

        """
        if len(detections) == 0:
            return []

        # Convert to arrays for NMS
        boxes = np.array([det[0] for det in detections])
        scores = np.array([det[1] for det in detections])
        classes = np.array([det[2] for det in detections])

        # Apply NMS per class
        keep_indices = []
        for cls in np.unique(classes):
            cls_mask = classes == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = np.where(cls_mask)[0]

            # NMS
            keep = self._nms(cls_boxes, cls_scores, self.nms_iou_thresh)
            keep_indices.extend(cls_indices[keep])

        # Return merged detections
        merged = [detections[i] for i in keep_indices]
        return merged

    def _nms(
        self: Self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_thresh: float,
    ) -> list[int]:
        """
        Non-maximum suppression.

        Parameters
        ----------
        boxes : np.ndarray
            Boxes as (x1, y1, x2, y2).
        scores : np.ndarray
            Confidence scores.
        iou_thresh : float
            IoU threshold.

        Returns
        -------
        list[int]
            Indices of boxes to keep.

        """
        if len(boxes) == 0:
            return []

        # Sort by score descending
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(int(i))

            if len(order) == 1:
                break

            # Compute IoU with remaining boxes
            ious = self._compute_ious(boxes[i], boxes[order[1:]])

            # Keep boxes with IoU below threshold
            order = order[1:][ious <= iou_thresh]

        return keep

    def _compute_ious(
        self: Self,
        box: np.ndarray,
        boxes: np.ndarray,
    ) -> np.ndarray:
        """
        Compute IoU between one box and multiple boxes.

        Parameters
        ----------
        box : np.ndarray
            Single box (x1, y1, x2, y2).
        boxes : np.ndarray
            Multiple boxes (N, 4).

        Returns
        -------
        np.ndarray
            IoU scores.

        """
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union = box_area + boxes_area - intersection

        return intersection / np.maximum(union, 1e-6)

    def reset(self: Self) -> None:
        """Reset executor state."""
        self.skip_windows.clear()
        self.prev_detections.clear()

