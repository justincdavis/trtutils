#!/usr/bin/env python3
# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Validate FlexPatch against MOT17 ground truth annotations.

This script compares FlexPatch detections with ground truth to compute:
- Precision, Recall, F1-score
- Average IoU
- Detection accuracy
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from trtutils.models import YOLO
from trtutils.research.flexpatch import FlexPatch


class GroundTruth(NamedTuple):
    """Ground truth annotation."""

    frame_id: int
    track_id: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    class_id: int
    visibility: float


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
        Bounding box from detector in format (x1, y1, x2, y2)
    bbox2 : tuple[int, int, int, int]
        Bounding box from ground truth in format (x, y, w, h)
        
    Returns
    -------
    float
        IoU score [0, 1]
    """
    # bbox1 is (x1, y1, x2, y2) from detector
    box1 = bbox1
    
    # bbox2 is (x, y, w, h) from ground truth - convert to (x1, y1, x2, y2)
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
    # box1 is (x1, y1, x2, y2), so area = (x2-x1) * (y2-y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    # box2 was converted from (x, y, w, h), so area = w * h
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def match_detections_to_gt(
    detections: list[tuple[tuple[int, int, int, int], float, int]],
    gt_list: list[GroundTruth],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int, list[float]]:
    """
    Match detections to ground truth using IoU.
    
    NOTE: We only match on IoU, ignoring class labels since:
    - YOLO uses COCO classes (person=0)
    - MOT17 uses different class IDs (person=1)
    
    Parameters
    ----------
    detections : list
        List of (bbox, conf, class) tuples
    gt_list : list[GroundTruth]
        List of ground truth annotations
    iou_threshold : float
        IoU threshold for matching
        
    Returns
    -------
    tuple[int, int, int, list[float]]
        (true_positives, false_positives, false_negatives, ious)
    """
    if not detections and not gt_list:
        return 0, 0, 0, []
    
    if not detections:
        return 0, 0, len(gt_list), []
    
    if not gt_list:
        return 0, len(detections), 0, []
    
    # Filter ground truth to only pedestrians (class 1 in MOT17)
    # MOT17 classes: 1=pedestrian, 2=person on vehicle, 7=static person, etc.
    gt_list = [gt for gt in gt_list if gt.class_id == 1]
    
    if not gt_list:
        return 0, len(detections), 0, []
    
    # Build IoU matrix (class-agnostic matching)
    iou_matrix = np.zeros((len(detections), len(gt_list)))
    for i, (det_bbox, _, _) in enumerate(detections):
        for j, gt in enumerate(gt_list):
            iou_matrix[i, j] = compute_iou(det_bbox, gt.bbox)
    
    # Greedy matching: match highest IoU first
    matched_dets = set()
    matched_gts = set()
    ious = []
    
    while True:
        # Find maximum IoU
        max_iou = iou_matrix.max()
        if max_iou < iou_threshold:
            break
        
        # Get indices of maximum
        i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        
        # Mark as matched
        matched_dets.add(i)
        matched_gts.add(j)
        ious.append(max_iou)
        
        # Remove from consideration
        iou_matrix[i, :] = 0
        iou_matrix[:, j] = 0
    
    true_positives = len(matched_dets)
    false_positives = len(detections) - true_positives
    false_negatives = len(gt_list) - len(matched_gts)
    
    return true_positives, false_positives, false_negatives, ious


def main() -> None:
    """Validate FlexPatch against ground truth."""
    print("=" * 80)
    print("FLEXPATCH VALIDATION - Ground Truth Comparison")
    print("=" * 80)
    
    # Configuration
    video_path = Path("mot17_13.mp4")
    model_path = Path("data/yolov10/yolov10m_640.engine")
    trained_model_path = Path("flexpatch_training/flexpatch_model.joblib")
    gt_file = Path("/home/orinagx/research/datasets/mot17det/train/MOT17-13/gt/gt.txt")
    
    # Verify files
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    if not gt_file.exists():
        print(f"❌ Ground truth not found: {gt_file}")
        return
    
    print(f"\n📹 Video: {video_path}")
    print(f"🤖 Model: {model_path}")
    print(f"🧠 Trained Model: {trained_model_path}")
    print(f"📊 Ground Truth: {gt_file}")
    
    # Load ground truth
    print("\n" + "=" * 80)
    print("1. LOADING GROUND TRUTH")
    print("=" * 80)
    
    gt_data = load_mot_ground_truth(gt_file)
    total_gt_objects = sum(len(gts) for gts in gt_data.values())
    
    print(f"✓ Loaded ground truth")
    print(f"  • Frames with annotations: {len(gt_data)}")
    print(f"  • Total ground truth objects: {total_gt_objects}")
    print(f"  • Average objects per frame: {total_gt_objects / len(gt_data):.2f}")
    
    # Load video
    print("\n" + "=" * 80)
    print("2. LOADING VIDEO")
    print("=" * 80)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ Failed to open video")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✓ Resolution: {width}×{height}")
    print(f"✓ FPS: {fps:.2f}")
    print(f"✓ Total frames: {total_frames}")
    
    # Load detector
    print("\n" + "=" * 80)
    print("3. LOADING DETECTOR")
    print("=" * 80)
    
    detector = YOLO(
        engine_path=model_path,
        conf_thres=0.25,
        nms_iou_thres=0.45,
    )
    print(f"✓ Detector loaded: {detector.name}")
    
    # Initialize FlexPatch
    print("\n" + "=" * 80)
    print("4. INITIALIZING FLEXPATCH")
    print("=" * 80)
    
    use_trained_model = trained_model_path.exists()
    if use_trained_model:
        print(f"✓ Using trained model: {trained_model_path}")
        flexpatch = FlexPatch(
            detector=detector,
            frame_size=(width, height),
            tf_model_path=trained_model_path,
        )
    else:
        print("⚠ Trained model not found, using default")
        flexpatch = FlexPatch(
            detector=detector,
            frame_size=(width, height),
        )
    
    print("✓ FlexPatch initialized")
    
    # Validate FlexPatch
    print("\n" + "=" * 80)
    print("5. VALIDATING FLEXPATCH")
    print("=" * 80)
    
    num_test_frames = 100
    print(f"Testing on {num_test_frames} frames...")
    
    flexpatch_stats = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "ious": [],
        "frames_tested": 0,
    }
    
    for frame_idx in range(1, num_test_frames + 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get ground truth for this frame
        gt_list = gt_data.get(frame_idx, [])
        
        # Run FlexPatch
        detections = flexpatch.process_frame(frame)
        
        # Match detections to ground truth
        tp, fp, fn, ious = match_detections_to_gt(detections, gt_list, iou_threshold=0.5)
        
        flexpatch_stats["tp"] += tp
        flexpatch_stats["fp"] += fp
        flexpatch_stats["fn"] += fn
        flexpatch_stats["ious"].extend(ious)
        flexpatch_stats["frames_tested"] += 1
        
        if frame_idx % 20 == 0:
            print(f"  Frame {frame_idx}: {len(detections)} dets, {len(gt_list)} GT, TP={tp}, FP={fp}, FN={fn}")
    
    cap.release()
    
    # Compute metrics for FlexPatch
    tp = flexpatch_stats["tp"]
    fp = flexpatch_stats["fp"]
    fn = flexpatch_stats["fn"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou = np.mean(flexpatch_stats["ious"]) if flexpatch_stats["ious"] else 0.0
    
    print(f"\n✓ FlexPatch Validation Results:")
    print(f"  • Frames tested: {flexpatch_stats['frames_tested']}")
    print(f"  • True Positives: {tp}")
    print(f"  • False Positives: {fp}")
    print(f"  • False Negatives: {fn}")
    print(f"  • Precision: {precision:.3f}")
    print(f"  • Recall: {recall:.3f}")
    print(f"  • F1-Score: {f1:.3f}")
    print(f"  • Average IoU: {avg_iou:.3f}")
    
    # Validate Baseline Detector
    print("\n" + "=" * 80)
    print("6. VALIDATING BASELINE DETECTOR")
    print("=" * 80)
    
    cap = cv2.VideoCapture(str(video_path))
    
    baseline_stats = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "ious": [],
        "frames_tested": 0,
    }
    
    for frame_idx in range(1, num_test_frames + 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get ground truth for this frame
        gt_list = gt_data.get(frame_idx, [])
        
        # Run baseline detector
        detections = detector.end2end(frame)
        
        # Match detections to ground truth
        tp, fp, fn, ious = match_detections_to_gt(detections, gt_list, iou_threshold=0.5)
        
        baseline_stats["tp"] += tp
        baseline_stats["fp"] += fp
        baseline_stats["fn"] += fn
        baseline_stats["ious"].extend(ious)
        baseline_stats["frames_tested"] += 1
        
        if frame_idx % 20 == 0:
            print(f"  Frame {frame_idx}: {len(detections)} dets, {len(gt_list)} GT, TP={tp}, FP={fp}, FN={fn}")
    
    cap.release()
    
    # Compute metrics for Baseline
    tp = baseline_stats["tp"]
    fp = baseline_stats["fp"]
    fn = baseline_stats["fn"]
    
    baseline_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    baseline_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    baseline_f1 = 2 * baseline_precision * baseline_recall / (baseline_precision + baseline_recall) if (baseline_precision + baseline_recall) > 0 else 0.0
    baseline_avg_iou = np.mean(baseline_stats["ious"]) if baseline_stats["ious"] else 0.0
    
    print(f"\n✓ Baseline Validation Results:")
    print(f"  • Frames tested: {baseline_stats['frames_tested']}")
    print(f"  • True Positives: {tp}")
    print(f"  • False Positives: {fp}")
    print(f"  • False Negatives: {fn}")
    print(f"  • Precision: {baseline_precision:.3f}")
    print(f"  • Recall: {baseline_recall:.3f}")
    print(f"  • F1-Score: {baseline_f1:.3f}")
    print(f"  • Average IoU: {baseline_avg_iou:.3f}")
    
    # Comparison
    print("\n" + "=" * 80)
    print("7. COMPARISON")
    print("=" * 80)
    
    print("\nFlexPatch vs Baseline:")
    print(f"  • Precision: {precision:.3f} vs {baseline_precision:.3f} (Δ {precision - baseline_precision:+.3f})")
    print(f"  • Recall: {recall:.3f} vs {baseline_recall:.3f} (Δ {recall - baseline_recall:+.3f})")
    print(f"  • F1-Score: {f1:.3f} vs {baseline_f1:.3f} (Δ {f1 - baseline_f1:+.3f})")
    print(f"  • Avg IoU: {avg_iou:.3f} vs {baseline_avg_iou:.3f} (Δ {avg_iou - baseline_avg_iou:+.3f})")
    
    # Analysis
    print("\n" + "=" * 80)
    print("8. ANALYSIS")
    print("=" * 80)
    
    if abs(f1 - baseline_f1) < 0.05:
        print("✓ FlexPatch maintains similar detection quality to baseline")
        print("  → Tracking and patch-based detection working correctly")
    elif f1 < baseline_f1:
        print("⚠ FlexPatch F1-score lower than baseline")
        print("  → This is expected: FlexPatch runs detector less frequently")
        print("  → Trade-off: Lower computational cost for slightly reduced recall")
    else:
        print("✓ FlexPatch F1-score higher than baseline")
        print("  → Temporal consistency from tracking may help reduce false positives")
    
    if avg_iou > 0.5:
        print(f"\n✓ Average IoU ({avg_iou:.3f}) indicates good localization")
    
    print(f"\nDetection Coverage:")
    coverage = flexpatch_stats["tp"] / (flexpatch_stats["tp"] + flexpatch_stats["fn"]) if (flexpatch_stats["tp"] + flexpatch_stats["fn"]) > 0 else 0.0
    print(f"  • FlexPatch detected {coverage*100:.1f}% of ground truth objects")
    
    baseline_coverage = baseline_stats["tp"] / (baseline_stats["tp"] + baseline_stats["fn"]) if (baseline_stats["tp"] + baseline_stats["fn"]) > 0 else 0.0
    print(f"  • Baseline detected {baseline_coverage*100:.1f}% of ground truth objects")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"✓ Tested {flexpatch_stats['frames_tested']} frames against ground truth")
    print(f"✓ FlexPatch F1-Score: {f1:.3f}")
    print(f"✓ Baseline F1-Score: {baseline_f1:.3f}")
    
    if baseline_f1 > 0:
        print(f"✓ FlexPatch maintains {(f1/baseline_f1)*100:.1f}% of baseline accuracy")
    else:
        print(f"⚠ Unable to compute accuracy ratio (baseline F1 is zero)")
    
    if use_trained_model:
        print(f"✓ Trained model used successfully")
    
    print("\n" + "=" * 80)
    print("✓ VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

