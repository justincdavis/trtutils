#!/usr/bin/env python3
# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Comprehensive FlexPatch test using real video and YOLO detector.

This script tests FlexPatch on MOT17 dataset with YOLOv10.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from trtutils.models import YOLO
from trtutils.research.flexpatch import FlexPatch


def main() -> None:
    """Run comprehensive FlexPatch test."""
    print("=" * 80)
    print("FLEXPATCH COMPREHENSIVE TEST")
    print("=" * 80)
    
    # Configuration
    video_path = Path("mot17_13.mp4")
    model_path = Path("data/yolov10/yolov10m_640.engine")
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Verify files exist
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\n📹 Video: {video_path}")
    print(f"🤖 Model: {model_path}")
    print(f"📁 Output: {output_dir}")
    
    # Load video
    print("\n" + "=" * 80)
    print("1. LOADING VIDEO")
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
    print("2. LOADING YOLO DETECTOR")
    print("=" * 80)
    detector = YOLO(
        engine_path=model_path,
        conf_thres=0.25,
        nms_iou_thres=0.45,
        verbose=True,
    )
    print(f"✓ Detector loaded: {detector.name}")
    print(f"✓ Input shape: {detector.input_shape}")
    
    # Initialize FlexPatch
    print("\n" + "=" * 80)
    print("3. INITIALIZING FLEXPATCH")
    print("=" * 80)
    flexpatch = FlexPatch(
        detector=detector,
        frame_size=(width, height),
        cluster_size=(640, 360),
        cell_size=(20, 22),
        max_age=10,
        tf_ratio=0.75,
        use_ratio_packing=True,
    )
    print("✓ FlexPatch initialized")
    print(f"  • Frame size: {width}×{height}")
    print(f"  • Cluster size: 640×360")
    print(f"  • Max age: 10 frames")
    print(f"  • TF:NO ratio: 0.75 (3:1)")
    
    # Test 1: FlexPatch inference
    print("\n" + "=" * 80)
    print("4. TESTING FLEXPATCH (Processing 100 frames)")
    print("=" * 80)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    flexpatch_times = []
    flexpatch_detections_count = []
    frames_processed = 0
    max_test_frames = 100
    
    print("Processing frames with FlexPatch...")
    while frames_processed < max_test_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.perf_counter()
        detections = flexpatch.process_frame(frame, verbose=False)
        end_time = time.perf_counter()
        
        elapsed = (end_time - start_time) * 1000  # Convert to ms
        flexpatch_times.append(elapsed)
        flexpatch_detections_count.append(len(detections))
        frames_processed += 1
        
        if frames_processed % 10 == 0:
            avg_time = np.mean(flexpatch_times[-10:])
            avg_dets = np.mean(flexpatch_detections_count[-10:])
            print(f"  Frame {frames_processed}: {avg_time:.1f}ms, {avg_dets:.1f} dets")
    
    flexpatch_mean = np.mean(flexpatch_times)
    flexpatch_std = np.std(flexpatch_times)
    flexpatch_fps = 1000.0 / flexpatch_mean
    
    print(f"\n✓ FlexPatch Results ({frames_processed} frames):")
    print(f"  • Mean latency: {flexpatch_mean:.2f}ms ± {flexpatch_std:.2f}ms")
    print(f"  • Min latency: {np.min(flexpatch_times):.2f}ms")
    print(f"  • Max latency: {np.max(flexpatch_times):.2f}ms")
    print(f"  • Throughput: {flexpatch_fps:.2f} FPS")
    print(f"  • Avg detections: {np.mean(flexpatch_detections_count):.2f}")
    
    # Test 2: Baseline (full-frame) inference
    print("\n" + "=" * 80)
    print("5. TESTING BASELINE (Full-frame detection)")
    print("=" * 80)
    
    # Reset FlexPatch for fair comparison
    flexpatch.reset()
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    baseline_times = []
    baseline_detections_count = []
    frames_processed = 0
    
    print("Processing frames with baseline detector...")
    while frames_processed < max_test_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.perf_counter()
        detections = detector.end2end(frame)
        end_time = time.perf_counter()
        
        elapsed = (end_time - start_time) * 1000
        baseline_times.append(elapsed)
        baseline_detections_count.append(len(detections))
        frames_processed += 1
        
        if frames_processed % 10 == 0:
            avg_time = np.mean(baseline_times[-10:])
            avg_dets = np.mean(baseline_detections_count[-10:])
            print(f"  Frame {frames_processed}: {avg_time:.1f}ms, {avg_dets:.1f} dets")
    
    baseline_mean = np.mean(baseline_times)
    baseline_std = np.std(baseline_times)
    baseline_fps = 1000.0 / baseline_mean
    
    print(f"\n✓ Baseline Results ({frames_processed} frames):")
    print(f"  • Mean latency: {baseline_mean:.2f}ms ± {baseline_std:.2f}ms")
    print(f"  • Min latency: {np.min(baseline_times):.2f}ms")
    print(f"  • Max latency: {np.max(baseline_times):.2f}ms")
    print(f"  • Throughput: {baseline_fps:.2f} FPS")
    print(f"  • Avg detections: {np.mean(baseline_detections_count):.2f}")
    
    # Comparison
    print("\n" + "=" * 80)
    print("6. PERFORMANCE COMPARISON")
    print("=" * 80)
    
    speedup = baseline_mean / flexpatch_mean
    latency_reduction = ((baseline_mean - flexpatch_mean) / baseline_mean) * 100
    fps_improvement = ((flexpatch_fps - baseline_fps) / baseline_fps) * 100
    
    print(f"FlexPatch vs Baseline:")
    print(f"  • Speedup: {speedup:.2f}×")
    print(f"  • Latency reduction: {latency_reduction:.1f}%")
    print(f"  • FPS improvement: {fps_improvement:.1f}%")
    print(f"  • Detection consistency: {np.mean(baseline_detections_count) - np.mean(flexpatch_detections_count):.2f} avg diff")
    
    # Visual test with output video
    print("\n" + "=" * 80)
    print("7. GENERATING OUTPUT VIDEO")
    print("=" * 80)
    
    output_video_path = output_dir / "flexpatch_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (width, height)
    )
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    flexpatch.reset()
    
    frames_to_save = min(300, total_frames)  # Save first 300 frames
    print(f"Saving {frames_to_save} frames with detections...")
    
    for i in range(frames_to_save):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run FlexPatch
        detections = flexpatch.process_frame(frame, verbose=False)
        
        # Draw detections
        for bbox, conf, cls_id in detections:
            x, y, w, h = bbox
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw label
            label = f"ID{cls_id}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x, y - label_h - 5),
                (x + label_w, y),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        # Add frame info
        info_text = f"Frame: {i+1}/{frames_to_save} | Detections: {len(detections)}"
        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        out.write(frame)
        
        if (i + 1) % 50 == 0:
            print(f"  Saved {i+1}/{frames_to_save} frames...")
    
    out.release()
    cap.release()
    
    print(f"✓ Output video saved: {output_video_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✓ Video processed: {video_path}")
    print(f"✓ Model used: {model_path}")
    print(f"✓ FlexPatch latency: {flexpatch_mean:.2f}ms ({flexpatch_fps:.2f} FPS)")
    print(f"✓ Baseline latency: {baseline_mean:.2f}ms ({baseline_fps:.2f} FPS)")
    print(f"✓ Performance gain: {speedup:.2f}× faster")
    print(f"✓ Output saved: {output_video_path}")
    print("\n" + "=" * 80)
    print("✓ FLEXPATCH TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

