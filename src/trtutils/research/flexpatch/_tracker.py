# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Object tracker using optical flow.

Classes
-------
ObjectTracker
    Optical flow-based object tracker with feature extraction.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self


class TrackedObject:
    """Represents a tracked object with its features."""

    def __init__(
        self: Self,
        bbox: tuple[int, int, int, int],
        obj_id: int,
        confidence: float = 1.0,
    ) -> None:
        """
        Initialize a tracked object.

        Parameters
        ----------
        bbox : tuple[int, int, int, int]
            Bounding box as (x, y, w, h).
        obj_id : int
            Unique object ID.
        confidence : float, optional
            Detection confidence score, by default 1.0.

        """
        self.bbox = bbox
        self.obj_id = obj_id
        self.confidence = confidence
        self.age = 0
        self.velocity = (0.0, 0.0)
        self.prev_velocity = (0.0, 0.0)
        
        # Tracking features
        self.min_eigenvalue: float = 0.0
        self.ncc: float = 1.0
        self.acceleration: float = 0.0
        self.flow_std: float = 0.0


class ObjectTracker:
    """Optical flow-based object tracker."""

    def __init__(
        self: Self,
        max_age: int = 10,
        feature_params: dict | None = None,
        lk_params: dict | None = None,
    ) -> None:
        """
        Initialize the object tracker.

        Parameters
        ----------
        max_age : int, optional
            Maximum age for tracked objects before removal, by default 10.
        feature_params : dict, optional
            Parameters for cv2.goodFeaturesToTrack, by default None.
        lk_params : dict, optional
            Parameters for cv2.calcOpticalFlowPyrLK, by default None.

        """
        self.max_age = max_age
        self.tracked_objects: list[TrackedObject] = []
        self.prev_frame: np.ndarray | None = None
        self.prev_gray: np.ndarray | None = None
        self._next_id = 0
        
        self.feature_params = feature_params or {
            "maxCorners": 50,
            "qualityLevel": 0.01,
            "minDistance": 5,
            "blockSize": 7,
        }
        
        self.lk_params = lk_params or {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        }

    def init(
        self: Self,
        frame: np.ndarray,
        detections: list[tuple[tuple[int, int, int, int], float, int]],
    ) -> None:
        """
        Initialize tracker with detections.

        Parameters
        ----------
        frame : np.ndarray
            The frame to initialize on.
        detections : list[tuple[tuple[int, int, int, int], float, int]]
            List of detections as (bbox, confidence, class_id).

        """
        self.tracked_objects = []
        self.prev_frame = frame.copy()
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for bbox, conf, _ in detections:
            obj = TrackedObject(bbox, self._next_id, conf)
            self._next_id += 1
            self.tracked_objects.append(obj)

    def update(
        self: Self,
        frame: np.ndarray,
    ) -> list[TrackedObject]:
        """
        Update tracked objects using optical flow.

        Parameters
        ----------
        frame : np.ndarray
            The current frame.

        Returns
        -------
        list[TrackedObject]
            List of tracked objects with updated positions and features.

        """
        if self.prev_frame is None or self.prev_gray is None:
            self.prev_frame = frame.copy()
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return self.tracked_objects

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        updated_objects = []
        for obj in self.tracked_objects:
            x, y, w, h = obj.bbox
            
            # Ensure valid ROI
            x = max(0, x)
            y = max(0, y)
            w = min(w, self.prev_gray.shape[1] - x)
            h = min(h, self.prev_gray.shape[0] - y)
            
            if w <= 0 or h <= 0:
                obj.age += 1
                if obj.age < self.max_age:
                    updated_objects.append(obj)
                continue
            
            # Extract ROI
            roi_prev = self.prev_gray[y:y+h, x:x+w]
            
            # Extract features
            pts_prev = cv2.goodFeaturesToTrack(roi_prev, **self.feature_params)
            
            if pts_prev is not None and len(pts_prev) > 0:
                # Compute minimum eigenvalue
                eigen_vals = cv2.cornerMinEigenVal(roi_prev, blockSize=7)
                obj.min_eigenvalue = float(np.mean(eigen_vals))
                
                # Offset points to full frame coordinates
                pts_prev_offset = pts_prev + np.array([x, y], dtype=np.float32)
                
                # Calculate optical flow
                pts_curr, status, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray,
                    curr_gray,
                    pts_prev_offset,
                    None,
                    **self.lk_params,
                )
                
                if pts_curr is not None and status is not None:
                    # Filter valid points
                    good_prev = pts_prev_offset[status.flatten() == 1]
                    good_curr = pts_curr[status.flatten() == 1]
                    
                    if len(good_curr) > 0:
                        # Compute displacement
                        displacement = good_curr - good_prev
                        mean_disp = np.mean(displacement, axis=0).flatten()
                        dx = float(mean_disp[0]) if len(mean_disp) > 0 else 0.0
                        dy = float(mean_disp[1]) if len(mean_disp) > 1 else 0.0
                        
                        # Update velocity
                        obj.prev_velocity = obj.velocity
                        obj.velocity = (float(dx), float(dy))
                        
                        # Compute acceleration
                        obj.acceleration = float(np.sqrt(
                            (obj.velocity[0] - obj.prev_velocity[0]) ** 2 +
                            (obj.velocity[1] - obj.prev_velocity[1]) ** 2
                        ))
                        
                        # Compute flow standard deviation
                        flow_magnitudes = np.sqrt(np.sum(displacement ** 2, axis=1))
                        obj.flow_std = float(np.std(flow_magnitudes))
                        
                        # Update bounding box
                        new_x = int(x + dx)
                        new_y = int(y + dy)
                        obj.bbox = (new_x, new_y, w, h)
                        
                        # Compute NCC for appearance consistency
                        new_x_clipped = max(0, min(new_x, curr_gray.shape[1] - w))
                        new_y_clipped = max(0, min(new_y, curr_gray.shape[0] - h))
                        roi_curr = curr_gray[new_y_clipped:new_y_clipped+h, new_x_clipped:new_x_clipped+w]
                        
                        if roi_curr.shape == roi_prev.shape:
                            ncc = cv2.matchTemplate(roi_curr, roi_prev, cv2.TM_CCORR_NORMED)
                            obj.ncc = float(ncc[0, 0])
                        
                        obj.age = 0
                    else:
                        obj.age += 1
                else:
                    obj.age += 1
            else:
                obj.age += 1
            
            if obj.age < self.max_age:
                updated_objects.append(obj)
        
        self.tracked_objects = updated_objects
        self.prev_frame = frame.copy()
        self.prev_gray = curr_gray
        
        return self.tracked_objects

    def refresh(
        self: Self,
        detections: list[tuple[tuple[int, int, int, int], float, int]],
        iou_threshold: float = 0.5,
    ) -> None:
        """
        Refresh tracked objects with new detections.

        Parameters
        ----------
        detections : list[tuple[tuple[int, int, int, int], float, int]]
            New detections as (bbox, confidence, class_id).
        iou_threshold : float, optional
            IoU threshold for matching detections to tracked objects, by default 0.5.

        """
        if not detections:
            return
        
        # Match detections to tracked objects
        matched_indices = set()
        for det_bbox, det_conf, _ in detections:
            best_iou = 0.0
            best_idx = -1
            
            for idx, obj in enumerate(self.tracked_objects):
                iou = self._compute_iou(obj.bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_iou > iou_threshold and best_idx >= 0:
                # Update existing object
                self.tracked_objects[best_idx].bbox = det_bbox
                self.tracked_objects[best_idx].confidence = det_conf
                self.tracked_objects[best_idx].age = 0
                matched_indices.add(best_idx)
            else:
                # Add new object
                obj = TrackedObject(det_bbox, self._next_id, det_conf)
                self._next_id += 1
                self.tracked_objects.append(obj)

    @staticmethod
    def _compute_iou(
        bbox1: tuple[int, int, int, int],
        bbox2: tuple[int, int, int, int],
    ) -> float:
        """Compute IoU between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to (x1, y1, x2, y2) format
        box1_x1, box1_y1, box1_x2, box1_y2 = x1, y1, x1 + w1, y1 + h1
        box2_x1, box2_y1, box2_x2, box2_y2 = x2, y2, x2 + w2, y2 + h2
        
        # Compute intersection
        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return float(inter_area / union_area)

