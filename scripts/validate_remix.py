#!/usr/bin/env python3
# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Validate Remix against MOT17 ground truth annotations.

This script:
1. Loads trained Remix system (profiles + plans)
2. Runs Remix on MOT17 test sequences
3. Runs baseline (full-frame) detection for comparison
4. Computes precision, recall, F1, IoU
5. Saves detailed results to file
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from trtutils.models import YOLO
from trtutils.research.remix import RemixSystem


class GroundTruth(NamedTuple):
    """Ground truth annotation."""

    frame_id: int
    track_id: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    class_id: int
    visibility: float


@dataclass
class ValidationMetrics:
    """Validation metrics."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_gt: int = 0
    total_detections: int = 0
    iou_sum: float = 0.0
    iou_count: int = 0

    @property
    def precision(self) -> float:
        """Compute precision."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Compute recall."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """Compute F1 score."""
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @property
    def avg_iou(self) -> float:
        """Compute average IoU."""
        if self.iou_count == 0:
            return 0.0
        return self.iou_sum / self.iou_count


def load_mot_ground_truth(gt_file: Path) -> dict[int, list[GroundTruth]]:
    """
    Load MOT17 ground truth annotations.

    MOT format: frame, id, x, y, w, h, conf, class, visibility

    Parameters
    ----------
    gt_file : Path
        Path to gt.txt file

    Returns
    -------
    dict[int, list[GroundTruth]]
        Ground truth annotations by frame number
    """
    gt_data = {}

    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue

            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            cls_id = int(parts[7]) if len(parts) > 7 else 1
            visibility = float(parts[8]) if len(parts) > 8 else 1.0

            # Filter out non-pedestrians (class 1 in MOT17)
            if cls_id != 1:
                continue

            # Convert to int for bbox
            bbox = (int(x), int(y), int(w), int(h))

            gt = GroundTruth(
                frame_id=frame_id,
                track_id=track_id,
                bbox=bbox,
                confidence=conf,
                class_id=cls_id,
                visibility=visibility,
            )

            if frame_id not in gt_data:
                gt_data[frame_id] = []
            gt_data[frame_id].append(gt)

    return gt_data


def compute_iou(bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]) -> float:
    """
    Compute IoU between two bounding boxes.

    Parameters
    ----------
    bbox1 : tuple[int, int, int, int]
        Detection box (x1, y1, x2, y2)
    bbox2 : tuple[int, int, int, int]
        Ground truth box (x, y, w, h)

    Returns
    -------
    float
        IoU score [0, 1]
    """
    # bbox1 is (x1, y1, x2, y2) from detector
    box1 = bbox1

    # bbox2 is (x, y, w, h) from ground truth - convert
    x2, y2, w2, h2 = bbox2
    box2 = (x2, y2, x2 + w2, y2 + h2)

    # Compute intersection
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    # Compute union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = w2 * h2
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def match_detections_to_gt(
    detections: list[tuple[tuple[int, int, int, int], float, int]],
    gt_list: list[GroundTruth],
    iou_threshold: float = 0.5,
) -> ValidationMetrics:
    """
    Match detections to ground truth.

    Parameters
    ----------
    detections : list
        List of (bbox, conf, class) tuples
    gt_list : list[GroundTruth]
        Ground truth annotations
    iou_threshold : float
        IoU threshold for matching

    Returns
    -------
    ValidationMetrics
        Computed metrics
    """
    metrics = ValidationMetrics()
    metrics.total_gt = len(gt_list)
    metrics.total_detections = len(detections)

    if len(detections) == 0:
        metrics.false_negatives = len(gt_list)
        return metrics

    if len(gt_list) == 0:
        metrics.false_positives = len(detections)
        return metrics

    # Compute IoU matrix
    iou_matrix = np.zeros((len(detections), len(gt_list)))
    for i, (det_bbox, _, _) in enumerate(detections):
        for j, gt in enumerate(gt_list):
            iou_matrix[i, j] = compute_iou(det_bbox, gt.bbox)

    # Greedy matching
    matched_dets = set()
    matched_gts = set()

    while True:
        if iou_matrix.size == 0 or np.max(iou_matrix) < iou_threshold:
            break

        # Find best match
        i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        iou_val = iou_matrix[i, j]

        if iou_val >= iou_threshold:
            matched_dets.add(i)
            matched_gts.add(j)
            metrics.true_positives += 1
            metrics.iou_sum += iou_val
            metrics.iou_count += 1

            # Remove matched row and column
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0

    # Count unmatched
    metrics.false_positives = len(detections) - len(matched_dets)
    metrics.false_negatives = len(gt_list) - len(matched_gts)

    return metrics


def main() -> None:
    """Validate Remix on MOT17."""
    print("=" * 80)
    print("REMIX VALIDATION - MOT17 Train Sequences")
    print("=" * 80)

    # Configuration
    mot17_path = Path("/home/orinagx/research/datasets/mot17/train")
    model_dir = Path("data/yolov10")
    data_dir = Path("remix_data")

    profile_path = data_dir / "profiles_mot17.json"
    plans_path = data_dir / "plans_mot17.json"
    results_path = Path("REMIX_VALIDATION_RESULTS.md")

    max_frames = 100  # Process 100 frames per sequence

    # Verify paths
    print("\n📁 Verifying paths...")
    if not mot17_path.exists():
        print(f"❌ MOT17 not found: {mot17_path}")
        return

    if not profile_path.exists():
        print(f"❌ Profiles not found: {profile_path}")
        print("   Run: python train_remix_mot17.py")
        return

    if not plans_path.exists():
        print(f"❌ Plans not found: {plans_path}")
        print("   Run: python train_remix_mot17.py")
        return

    print("✓ All paths verified")

    # Load detectors
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DETECTORS")
    print("=" * 80)

    models = {
        "yolov10n": model_dir / "yolov10n_640.engine",
        "yolov10s": model_dir / "yolov10s_640.engine",
        "yolov10m": model_dir / "yolov10m_640.engine",
        "oracle": model_dir / "yolov10m_1280.engine",
    }

    detectors = []
    for name in ["yolov10n", "yolov10s", "yolov10m"]:
        print(f"  Loading {name}...")
        detector = YOLO(models[name], warmup=True)
        detectors.append(detector)

    print("\nLoading oracle...")
    oracle = YOLO(models["oracle"], warmup=True)

    print(f"✓ Loaded {len(detectors)} detectors + oracle")

    # Initialize Remix
    print("\n" + "=" * 80)
    print("STEP 2: INITIALIZING REMIX")
    print("=" * 80)

    remix = RemixSystem(
        detectors=detectors,
        oracle=oracle,
        latency_budget=0.050,
        profile_path=profile_path,
        plans_path=plans_path,
        load_existing=True,
    )

    print(f"✓ Loaded {len(remix.profiles)} profiles")
    print(f"✓ Loaded {len(remix.plans)} plans")

    remix.initialize_runtime(kp=0.6, ki=0.3, kd=0.1)
    print("✓ Runtime initialized")

    # Initialize baseline detector for comparison
    print("\nInitializing baseline detector...")
    baseline = YOLO(models["yolov10m"], warmup=True)
    print("✓ Baseline detector ready")

    # Select test sequences
    print("\n" + "=" * 80)
    print("STEP 3: SELECTING TEST SEQUENCES")
    print("=" * 80)

    sequences = sorted([d for d in mot17_path.iterdir() if d.is_dir()])
    frcnn_sequences = [s for s in sequences if "FRCNN" in s.name]

    # Use different sequences than training (for validation)
    test_sequences = frcnn_sequences[3:5] if len(frcnn_sequences) > 3 else frcnn_sequences[:2]

    print(f"\nTest sequences:")
    for seq in test_sequences:
        print(f"  • {seq.name}")

    # Run validation
    print("\n" + "=" * 80)
    print("STEP 4: RUNNING VALIDATION")
    print("=" * 80)

    remix_metrics = ValidationMetrics()
    baseline_metrics = ValidationMetrics()
    remix_latencies = []
    baseline_latencies = []

    for seq_dir in test_sequences:
        print(f"\nProcessing {seq_dir.name}...")

        # Load ground truth
        gt_file = seq_dir / "gt" / "gt.txt"
        if not gt_file.exists():
            print(f"  ⚠ No ground truth found")
            continue

        gt_data = load_mot_ground_truth(gt_file)
        print(f"  ✓ Loaded ground truth: {len(gt_data)} frames")

        # Load images
        img_dir = seq_dir / "img1"
        img_files = sorted(img_dir.glob("*.jpg"))[:max_frames]
        print(f"  ✓ Processing {len(img_files)} frames")

        # Process frames
        for frame_idx, img_path in enumerate(img_files, start=1):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Get ground truth for this frame
            gt_list = gt_data.get(frame_idx, [])

            # Run Remix
            detections_remix, lat_remix = remix.run_frame(img, verbose=False)
            remix_latencies.append(lat_remix)

            # Run baseline
            import time
            t0 = time.perf_counter()
            detections_baseline = baseline.end2end(img, verbose=False)
            lat_baseline = time.perf_counter() - t0
            baseline_latencies.append(lat_baseline)

            # Match and accumulate metrics
            frame_metrics_remix = match_detections_to_gt(detections_remix, gt_list)
            frame_metrics_baseline = match_detections_to_gt(detections_baseline, gt_list)

            remix_metrics.true_positives += frame_metrics_remix.true_positives
            remix_metrics.false_positives += frame_metrics_remix.false_positives
            remix_metrics.false_negatives += frame_metrics_remix.false_negatives
            remix_metrics.total_gt += frame_metrics_remix.total_gt
            remix_metrics.total_detections += frame_metrics_remix.total_detections
            remix_metrics.iou_sum += frame_metrics_remix.iou_sum
            remix_metrics.iou_count += frame_metrics_remix.iou_count

            baseline_metrics.true_positives += frame_metrics_baseline.true_positives
            baseline_metrics.false_positives += frame_metrics_baseline.false_positives
            baseline_metrics.false_negatives += frame_metrics_baseline.false_negatives
            baseline_metrics.total_gt += frame_metrics_baseline.total_gt
            baseline_metrics.total_detections += frame_metrics_baseline.total_detections
            baseline_metrics.iou_sum += frame_metrics_baseline.iou_sum
            baseline_metrics.iou_count += frame_metrics_baseline.iou_count

            if (frame_idx) % 20 == 0:
                print(f"    Frame {frame_idx}/{len(img_files)} processed...")

        print(f"  ✓ Sequence complete")

    # Compute final statistics
    print("\n" + "=" * 80)
    print("STEP 5: COMPUTING RESULTS")
    print("=" * 80)

    avg_remix_lat = np.mean(remix_latencies) * 1000
    avg_baseline_lat = np.mean(baseline_latencies) * 1000

    # Display results
    print("\n" + "-" * 80)
    print("REMIX PERFORMANCE")
    print("-" * 80)
    print(f"True Positives:      {remix_metrics.true_positives}")
    print(f"False Positives:     {remix_metrics.false_positives}")
    print(f"False Negatives:     {remix_metrics.false_negatives}")
    print(f"\nPrecision:           {remix_metrics.precision:.3f} ({remix_metrics.precision*100:.1f}%)")
    print(f"Recall:              {remix_metrics.recall:.3f} ({remix_metrics.recall*100:.1f}%)")
    print(f"F1-Score:            {remix_metrics.f1_score:.3f} ({remix_metrics.f1_score*100:.1f}%)")
    print(f"Average IoU:         {remix_metrics.avg_iou:.3f}")
    print(f"Average Latency:     {avg_remix_lat:.2f}ms")

    print("\n" + "-" * 80)
    print("BASELINE PERFORMANCE")
    print("-" * 80)
    print(f"True Positives:      {baseline_metrics.true_positives}")
    print(f"False Positives:     {baseline_metrics.false_positives}")
    print(f"False Negatives:     {baseline_metrics.false_negatives}")
    print(f"\nPrecision:           {baseline_metrics.precision:.3f} ({baseline_metrics.precision*100:.1f}%)")
    print(f"Recall:              {baseline_metrics.recall:.3f} ({baseline_metrics.recall*100:.1f}%)")
    print(f"F1-Score:            {baseline_metrics.f1_score:.3f} ({baseline_metrics.f1_score*100:.1f}%)")
    print(f"Average IoU:         {baseline_metrics.avg_iou:.3f}")
    print(f"Average Latency:     {avg_baseline_lat:.2f}ms")

    print("\n" + "-" * 80)
    print("COMPARISON")
    print("-" * 80)
    speedup = avg_baseline_lat / avg_remix_lat if avg_remix_lat > 0 else 0
    acc_delta = remix_metrics.f1_score - baseline_metrics.f1_score

    print(f"Speedup:             {speedup:.2f}×")
    print(f"Accuracy Delta:      {acc_delta:+.3f} ({acc_delta*100:+.1f}%)")
    print(f"Latency Reduction:   {avg_baseline_lat - avg_remix_lat:.2f}ms")

    # Save results
    print("\n" + "=" * 80)
    print("STEP 6: SAVING RESULTS")
    print("=" * 80)

    with open(results_path, "w") as f:
        f.write("# Remix Validation Results\n\n")
        f.write("## Validation Complete\n\n")
        f.write(f"Successfully validated Remix against MOT17 train sequences.\n\n")

        f.write("## Test Configuration\n\n")
        f.write(f"- **Dataset**: MOT17 Train\n")
        f.write(f"- **Test Sequences**: {', '.join([s.name for s in test_sequences])}\n")
        f.write(f"- **Frames Processed**: {len(remix_latencies)}\n")
        f.write(f"- **Total GT Objects**: {remix_metrics.total_gt}\n")
        f.write(f"- **IoU Threshold**: 0.5\n\n")

        f.write("## Remix Performance\n\n")
        f.write("```\n")
        f.write(f"True Positives:      {remix_metrics.true_positives}\n")
        f.write(f"False Positives:     {remix_metrics.false_positives}\n")
        f.write(f"False Negatives:     {remix_metrics.false_negatives}\n")
        f.write(f"\n")
        f.write(f"Precision:           {remix_metrics.precision:.3f} ({remix_metrics.precision*100:.1f}%)\n")
        f.write(f"Recall:              {remix_metrics.recall:.3f} ({remix_metrics.recall*100:.1f}%)\n")
        f.write(f"F1-Score:            {remix_metrics.f1_score:.3f} ({remix_metrics.f1_score*100:.1f}%)\n")
        f.write(f"Average IoU:         {remix_metrics.avg_iou:.3f}\n")
        f.write(f"Average Latency:     {avg_remix_lat:.2f}ms\n")
        f.write("```\n\n")

        f.write("## Baseline Performance\n\n")
        f.write("```\n")
        f.write(f"True Positives:      {baseline_metrics.true_positives}\n")
        f.write(f"False Positives:     {baseline_metrics.false_positives}\n")
        f.write(f"False Negatives:     {baseline_metrics.false_negatives}\n")
        f.write(f"\n")
        f.write(f"Precision:           {baseline_metrics.precision:.3f} ({baseline_metrics.precision*100:.1f}%)\n")
        f.write(f"Recall:              {baseline_metrics.recall:.3f} ({baseline_metrics.recall*100:.1f}%)\n")
        f.write(f"F1-Score:            {baseline_metrics.f1_score:.3f} ({baseline_metrics.f1_score*100:.1f}%)\n")
        f.write(f"Average IoU:         {baseline_metrics.avg_iou:.3f}\n")
        f.write(f"Average Latency:     {avg_baseline_lat:.2f}ms\n")
        f.write("```\n\n")

        f.write("## Comparison\n\n")
        f.write("```\n")
        f.write(f"Metric              Remix        Baseline     Delta\n")
        f.write(f"──────────────────────────────────────────────────────\n")
        f.write(f"Precision           {remix_metrics.precision*100:5.1f}%       {baseline_metrics.precision*100:5.1f}%      {(remix_metrics.precision-baseline_metrics.precision)*100:+.1f}%\n")
        f.write(f"Recall              {remix_metrics.recall*100:5.1f}%       {baseline_metrics.recall*100:5.1f}%      {(remix_metrics.recall-baseline_metrics.recall)*100:+.1f}%\n")
        f.write(f"F1-Score            {remix_metrics.f1_score*100:5.1f}%       {baseline_metrics.f1_score*100:5.1f}%      {acc_delta*100:+.1f}%\n")
        f.write(f"Avg IoU             {remix_metrics.avg_iou:.3f}        {baseline_metrics.avg_iou:.3f}       {remix_metrics.avg_iou-baseline_metrics.avg_iou:+.3f}\n")
        f.write(f"Latency             {avg_remix_lat:6.2f}ms     {avg_baseline_lat:6.2f}ms    {avg_remix_lat-avg_baseline_lat:+.2f}ms\n")
        f.write(f"\n")
        f.write(f"Speedup: {speedup:.2f}×\n")
        f.write("```\n\n")

        f.write("## Analysis\n\n")
        if speedup > 1.0:
            f.write(f"✓ Remix achieved {speedup:.2f}× speedup over baseline\n")
        else:
            f.write(f"⚠ Remix was slower than baseline (may need tuning)\n")

        if abs(acc_delta) <= 0.05:
            f.write(f"✓ Accuracy maintained within 5% ({acc_delta*100:+.1f}%)\n")
        elif acc_delta > 0:
            f.write(f"✓ Accuracy improved by {acc_delta*100:.1f}%\n")
        else:
            f.write(f"⚠ Accuracy dropped by {-acc_delta*100:.1f}%\n")

    print(f"✓ Results saved to: {results_path}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

