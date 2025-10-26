# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Neural network profiling for Remix.

Classes
-------
NNProfiler
    Profiles DetectorInterface models for latency and accuracy by object size.

"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from trtutils._log import LOG

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.image.interfaces import DetectorInterface


class NNProfiler:
    """Profiles DetectorInterface models for latency and accuracy across object sizes."""

    def __init__(self: Self, detectors: list[DetectorInterface]) -> None:
        """
        Initialize the profiler.

        Parameters
        ----------
        detectors : list[DetectorInterface]
            List of detector models to profile.

        """
        self.detectors = detectors
        self.profiles: dict[str, dict] = {}

    def profile(
        self: Self,
        coco_path: Path | str,
        num_latency_runs: int = 20,
        max_images: int | None = None,
        *,
        verbose: bool = False,
    ) -> dict[str, dict]:
        """
        Profile all detectors for latency and accuracy.

        Parameters
        ----------
        coco_path : Path | str
            Path to COCO dataset directory.
        num_latency_runs : int, optional
            Number of runs for latency measurement, by default 20.
        max_images : int, optional
            Maximum number of images to evaluate, by default None (use all).
        verbose : bool, optional
            Whether to log detailed information, by default False.

        Returns
        -------
        dict[str, dict]
            Profiles mapping detector name to {latency, acc_vector}.

        """
        for det in self.detectors:
            if verbose:
                LOG.info(f"Profiling detector: {det.name}")

            lat = self._measure_latency(det, num_latency_runs, verbose=verbose)
            acc_vector = self._evaluate_accuracy(
                det,
                coco_path,
                max_images=max_images,
                verbose=verbose,
            )

            self.profiles[det.name] = {
                "latency": float(lat),
                "acc_vector": acc_vector.tolist(),
            }

            if verbose:
                LOG.info(
                    f"{det.name}: latency={lat*1000:.2f}ms, "
                    f"mean_acc={np.mean(acc_vector):.3f}",
                )

        return self.profiles

    def _measure_latency(
        self: Self,
        det: DetectorInterface,
        num_runs: int,
        *,
        verbose: bool = False,
    ) -> float:
        """
        Measure average inference latency.

        Parameters
        ----------
        det : DetectorInterface
            Detector to measure.
        num_runs : int
            Number of inference runs.
        verbose : bool, optional
            Whether to log information, by default False.

        Returns
        -------
        float
            Average latency in seconds.

        """
        if verbose:
            LOG.debug(f"Measuring latency for {det.name} over {num_runs} runs")

        # Create dummy input image (HWC format)
        h, w = det.input_shape
        dummy_image = np.zeros((h, w, 3), dtype=np.uint8)

        # Warmup
        for _ in range(5):
            det.run(dummy_image, preprocessed=False, postprocess=False)

        # Measure
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            det.run(dummy_image, preprocessed=False, postprocess=False)
            times.append(time.perf_counter() - start)

        return float(np.mean(times))

    def _evaluate_accuracy(
        self: Self,
        det: DetectorInterface,
        coco_path: Path | str,
        max_images: int | None = None,
        *,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Evaluate detector accuracy across object size bins using COCO.

        Parameters
        ----------
        det : DetectorInterface
            Detector to evaluate.
        coco_path : Path | str
            Path to COCO dataset.
        max_images : int, optional
            Maximum images to evaluate, by default None.
        verbose : bool, optional
            Whether to log information, by default False.

        Returns
        -------
        np.ndarray
            Accuracy vector of length 12 (size bins S0-S3, M0-M3, L0-L3).

        """
        if verbose:
            LOG.debug(f"Evaluating accuracy for {det.name} on COCO dataset")

        coco_path = Path(coco_path)

        # Load COCO annotations
        ann_file = coco_path / "annotations" / "instances_val2017.json"
        img_dir = coco_path / "val"

        with open(ann_file) as f:
            coco_data = json.load(f)

        # Create image ID to annotations mapping
        annotations_map = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_map:
                annotations_map[img_id] = []

            # Convert bbox from xywh to xyxy format
            x, y, w, h = ann['bbox']
            bbox = (int(x), int(y), int(x + w), int(y + h))
            annotations_map[img_id].append({
                'bbox': bbox,
                'category_id': ann['category_id'],
            })

        # Get image files
        image_list = [(img['file_name'], img['id']) for img in coco_data['images']]

        # Initialize bins for size-specific accuracy
        # 12 bins: log2(area) from 0 to 11+
        num_bins = 12
        bin_tp = np.zeros(num_bins)
        bin_fp = np.zeros(num_bins)
        bin_fn = np.zeros(num_bins)

        # Process images
        img_count = 0
        for img_filename, img_id in image_list:
            if max_images is not None and img_count >= max_images:
                break

            # Load image
            img_path = img_dir / img_filename
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Get annotations for this image
            annotations = annotations_map.get(img_id, [])

            # Run detector
            detections = det.end2end(img, verbose=False)

            # Convert ground truth to boxes (already in x1,y1,x2,y2 format)
            gt_boxes = [ann['bbox'] for ann in annotations]

            # Convert detections to boxes
            det_boxes = [bbox for bbox, _, _ in detections]

            # Match detections to ground truth
            matched_gt, matched_det = self._match_boxes(
                gt_boxes,
                det_boxes,
                iou_thresh=0.5,
            )

            # Update bins based on ground truth object sizes
            for i, gt_box in enumerate(gt_boxes):
                x1, y1, x2, y2 = gt_box
                area = (x2 - x1) * (y2 - y1)
                bin_idx = min(int(np.log2(area + 1)), num_bins - 1)

                if i in matched_gt:
                    bin_tp[bin_idx] += 1
                else:
                    bin_fn[bin_idx] += 1

            # Count false positives
            for i, det_box in enumerate(det_boxes):
                if i not in matched_det:
                    # Estimate size bin from detection
                    x1, y1, x2, y2 = det_box
                    area = (x2 - x1) * (y2 - y1)
                    bin_idx = min(int(np.log2(area + 1)), num_bins - 1)
                    bin_fp[bin_idx] += 1

            img_count += 1
            if verbose and img_count % 100 == 0:
                LOG.debug(f"Processed {img_count} images")

        # Calculate precision per bin
        acc_vector = np.zeros(num_bins)
        for i in range(num_bins):
            if bin_tp[i] + bin_fp[i] > 0:
                acc_vector[i] = bin_tp[i] / (bin_tp[i] + bin_fp[i])
            else:
                acc_vector[i] = 0.0

        if verbose:
            LOG.debug(f"Accuracy vector: {acc_vector}")

        return acc_vector

    def _match_boxes(
        self: Self,
        gt_boxes: list[tuple[int, int, int, int]],
        det_boxes: list[tuple[int, int, int, int]],
        iou_thresh: float = 0.5,
    ) -> tuple[set[int], set[int]]:
        """
        Match detection boxes to ground truth using IoU threshold.

        Parameters
        ----------
        gt_boxes : list[tuple[int, int, int, int]]
            Ground truth boxes as (x1, y1, x2, y2).
        det_boxes : list[tuple[int, int, int, int]]
            Detection boxes as (x1, y1, x2, y2).
        iou_thresh : float, optional
            IoU threshold for matching, by default 0.5.

        Returns
        -------
        tuple[set[int], set[int]]
            Sets of matched ground truth and detection indices.

        """
        matched_gt = set()
        matched_det = set()

        # Compute IoU matrix
        iou_matrix = np.zeros((len(gt_boxes), len(det_boxes)))
        for i, gt in enumerate(gt_boxes):
            for j, det in enumerate(det_boxes):
                iou_matrix[i, j] = self._compute_iou(gt, det)

        # Greedy matching: highest IoU first
        while True:
            if iou_matrix.size == 0:
                break

            max_iou = np.max(iou_matrix)
            if max_iou < iou_thresh:
                break

            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            matched_gt.add(int(i))
            matched_det.add(int(j))

            # Remove matched row and column
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0

        return matched_gt, matched_det

    def _compute_iou(
        self: Self,
        box1: tuple[int, int, int, int],
        box2: tuple[int, int, int, int],
    ) -> float:
        """
        Compute IoU between two boxes.

        Parameters
        ----------
        box1 : tuple[int, int, int, int]
            First box as (x1, y1, x2, y2).
        box2 : tuple[int, int, int, int]
            Second box as (x1, y1, x2, y2).

        Returns
        -------
        float
            IoU score between 0 and 1.

        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return float(intersection / union) if union > 0 else 0.0

    def save_profiles(self: Self, path: Path | str) -> None:
        """
        Save profiles to JSON file.

        Parameters
        ----------
        path : Path | str
            Output file path.

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(self.profiles, f, indent=2)

        LOG.info(f"Saved profiles to {path}")

    @classmethod
    def load_profiles(cls: type[Self], path: Path | str) -> dict[str, dict]:
        """
        Load profiles from JSON file.

        Parameters
        ----------
        path : Path | str
            Input file path.

        Returns
        -------
        dict[str, dict]
            Loaded profiles.

        """
        path = Path(path)
        with path.open("r") as f:
            profiles = json.load(f)

        LOG.info(f"Loaded profiles from {path}")
        return profiles

