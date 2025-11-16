#!/usr/bin/env python3
# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
FlexPatch test with available resources or synthetic data.

This script tests FlexPatch using:
1. Available YOLO engine (searches for one)
2. Synthetic video generated from images or created from scratch
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from trtutils.models import YOLO
from trtutils.research.flexpatch import FlexPatch


def find_yolo_engine() -> Path | None:
    """Find an available YOLO engine."""
    search_paths = [
        Path("data/yolov10/yolov10n_640.engine"),
        Path("data/ultralytics/yolov10n_640.engine"),
        Path("data/yolov8/yolov8n_640.engine"),
        Path("data/ultralytics/yolov8n_640.engine"),
        Path("data/engines/trt_yolov10n.engine"),
        Path("data/engines/trt_yolov8n.engine"),
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None


def create_synthetic_video(output_path: Path, num_frames: int = 200) -> tuple[int, int]:
    """Create a synthetic video with moving objects."""
    width, height = 1920, 1080
    fps = 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"Generating {num_frames} frames of synthetic video...")
    
    for i in range(num_frames):
        # Create frame with gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(50, 100, width, dtype=np.uint8)  # Blue
        frame[:, :, 1] = np.linspace(80, 120, height, dtype=np.uint8).reshape(-1, 1)  # Green
        frame[:, :, 2] = 60  # Red
        
        # Add moving objects (simulated cars/people)
        for obj_id in range(5):
            # Object moves across screen
            x = int((i * 10 + obj_id * 300) % (width + 200) - 100)
            y = 200 + obj_id * 150
            
            # Draw rectangle (simulated object)
            color = ((obj_id * 50) % 256, (obj_id * 100) % 256, (obj_id * 150) % 256)
            cv2.rectangle(frame, (x, y), (x + 100, y + 80), color, -1)
            cv2.rectangle(frame, (x, y), (x + 100, y + 80), (255, 255, 255), 2)
        
        # Add some noise
        noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{num_frames} frames...")
    
    out.release()
    print(f"✓ Synthetic video created: {output_path}")
    
    return width, height


def main() -> None:
    """Run FlexPatch test with available resources."""
    print("=" * 80)
    print("FLEXPATCH TEST - SEARCHING FOR RESOURCES")
    print("=" * 80)
    
    # Find YOLO engine
    print("\n1. Searching for YOLO engine...")
    model_path = find_yolo_engine()
    if model_path is None:
        print("❌ No YOLO engine found. Please ensure you have a model in:")
        print("   - data/yolov10/yolov10n_640.engine")
        print("   - data/ultralytics/yolov10n_640.engine")
        print("   - data/engines/trt_yolov10n.engine")
        return
    
    print(f"✓ Found YOLO engine: {model_path}")
    
    # Check for video or create synthetic
    print("\n2. Checking for video...")
    video_path = Path("data/mot17_13.mp4")
    
    if not video_path.exists():
        print(f"⚠  Video not found: {video_path}")
        print("   Creating synthetic video for testing...")
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        video_path = output_dir / "synthetic_test.mp4"
        width, height = create_synthetic_video(video_path, num_frames=200)
    else:
        print(f"✓ Found video: {video_path}")
        # Get video dimensions
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    
    # Now run the comprehensive test
    print("\n" + "=" * 80)
    print("STARTING FLEXPATCH TEST")
    print("=" * 80)
    print(f"Video: {video_path} ({width}×{height})")
    print(f"Model: {model_path}")
    
    # Load detector
    print("\nLoading detector...")
    detector = YOLO(
        engine_path=model_path,
        conf_thres=0.25,
        nms_iou_thres=0.45,
    )
    print(f"✓ Detector: {detector.name}")
    print(f"✓ Input shape: {detector.input_shape}")
    
    # Initialize FlexPatch
    print("\nInitializing FlexPatch...")
    flexpatch = FlexPatch(
        detector=detector,
        frame_size=(width, height),
        cluster_size=(640, 360),
        max_age=10,
        tf_ratio=0.75,
    )
    print("✓ FlexPatch initialized")
    
    # Test FlexPatch
    print("\n" + "=" * 80)
    print("TESTING FLEXPATCH (100 frames)")
    print("=" * 80)
    
    cap = cv2.VideoCapture(str(video_path))
    flexpatch_times = []
    frames_processed = 0
    max_frames = 100
    
    while frames_processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.perf_counter()
        detections = flexpatch.process_frame(frame)
        end = time.perf_counter()
        
        elapsed_ms = (end - start) * 1000
        flexpatch_times.append(elapsed_ms)
        frames_processed += 1
        
        if frames_processed % 10 == 0:
            avg = np.mean(flexpatch_times[-10:])
            print(f"Frame {frames_processed}: {avg:.1f}ms avg")
    
    cap.release()
    
    fp_mean = np.mean(flexpatch_times)
    fp_std = np.std(flexpatch_times)
    fp_fps = 1000.0 / fp_mean
    
    print(f"\n✓ FlexPatch Results:")
    print(f"  Latency: {fp_mean:.2f}ms ± {fp_std:.2f}ms")
    print(f"  FPS: {fp_fps:.2f}")
    print(f"  Min: {np.min(flexpatch_times):.2f}ms")
    print(f"  Max: {np.max(flexpatch_times):.2f}ms")
    
    # Test Baseline
    print("\n" + "=" * 80)
    print("TESTING BASELINE (100 frames)")
    print("=" * 80)
    
    cap = cv2.VideoCapture(str(video_path))
    baseline_times = []
    frames_processed = 0
    
    while frames_processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.perf_counter()
        detections = detector.end2end(frame)
        end = time.perf_counter()
        
        elapsed_ms = (end - start) * 1000
        baseline_times.append(elapsed_ms)
        frames_processed += 1
        
        if frames_processed % 10 == 0:
            avg = np.mean(baseline_times[-10:])
            print(f"Frame {frames_processed}: {avg:.1f}ms avg")
    
    cap.release()
    
    bl_mean = np.mean(baseline_times)
    bl_std = np.std(baseline_times)
    bl_fps = 1000.0 / bl_mean
    
    print(f"\n✓ Baseline Results:")
    print(f"  Latency: {bl_mean:.2f}ms ± {bl_std:.2f}ms")
    print(f"  FPS: {bl_fps:.2f}")
    print(f"  Min: {np.min(baseline_times):.2f}ms")
    print(f"  Max: {np.max(baseline_times):.2f}ms")
    
    # Comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    speedup = bl_mean / fp_mean
    latency_reduction = ((bl_mean - fp_mean) / bl_mean) * 100
    
    print(f"FlexPatch vs Baseline:")
    print(f"  Speedup: {speedup:.2f}×")
    print(f"  Latency reduction: {latency_reduction:.1f}%")
    print(f"  FPS gain: {fp_fps - bl_fps:.2f} FPS")
    
    if speedup > 1.0:
        print(f"\n✓ FlexPatch is {speedup:.2f}× FASTER than baseline!")
    else:
        print(f"\n⚠ FlexPatch is slower (overhead from tracking/packing)")
        print("  Note: This is expected for the first few frames during initialization")
    
    print("\n" + "=" * 80)
    print("✓ TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()


