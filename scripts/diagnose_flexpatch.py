#!/usr/bin/env python3
# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Diagnose FlexPatch performance bottlenecks.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from trtutils.models import YOLO
from trtutils.research.flexpatch import FlexPatch


def main() -> None:
    """Diagnose FlexPatch performance."""
    print("=" * 80)
    print("FLEXPATCH PERFORMANCE DIAGNOSTIC")
    print("=" * 80)
    
    # Load resources
    video_path = Path("mot17_13.mp4")
    model_path = Path("data/yolov10/yolov10n_640.engine")
    
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    detector = YOLO(model_path, conf_thres=0.25)
    flexpatch = FlexPatch(detector, frame_size=(width, height))
    
    print(f"\nVideo: {width}×{height}")
    print(f"Model: {detector.name}")
    print(f"Input shape: {detector.input_shape}")
    
    # Profile individual components
    print("\n" + "=" * 80)
    print("PROFILING COMPONENTS (50 frames)")
    print("=" * 80)
    
    times = {
        "total": [],
        "tracker_update": [],
        "tf_recommend": [],
        "no_recommend": [],
        "aggregator": [],
        "detection": [],
    }
    
    for i in range(50):
        ret, frame = cap.read()
        if not ret:
            break
        
        t_start = time.perf_counter()
        
        # Track
        t0 = time.perf_counter()
        tracked = flexpatch.tracker.update(frame)
        t1 = time.perf_counter()
        times["tracker_update"].append((t1 - t0) * 1000)
        
        # TF recommend
        t0 = time.perf_counter()
        tf_patches = flexpatch.tf_recommender.recommend(tracked)
        t1 = time.perf_counter()
        times["tf_recommend"].append((t1 - t0) * 1000)
        
        # NO recommend
        t0 = time.perf_counter()
        exclusions = [obj.bbox for obj in tracked]
        no_patches = flexpatch.no_recommender.recommend(frame, exclusions)
        t1 = time.perf_counter()
        times["no_recommend"].append((t1 - t0) * 1000)
        
        # Aggregate
        t0 = time.perf_counter()
        all_patches = tf_patches + no_patches
        if all_patches:
            packed = flexpatch.aggregator.pack(all_patches)
        else:
            packed = []
        t1 = time.perf_counter()
        times["aggregator"].append((t1 - t0) * 1000)
        
        # Detection (if patches exist)
        if packed:
            t0 = time.perf_counter()
            # Simulate detection on patches
            cluster = np.zeros((360, 640, 3), dtype=np.uint8)
            _ = detector.end2end(cluster)
            t1 = time.perf_counter()
            times["detection"].append((t1 - t0) * 1000)
        else:
            times["detection"].append(0.0)
        
        t_end = time.perf_counter()
        times["total"].append((t_end - t_start) * 1000)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/50 frames...")
    
    cap.release()
    
    # Print results
    print("\n" + "=" * 80)
    print("COMPONENT BREAKDOWN")
    print("=" * 80)
    
    total_mean = np.mean(times["total"])
    
    for component, timings in times.items():
        if component == "total":
            continue
        mean_time = np.mean(timings)
        percentage = (mean_time / total_mean) * 100
        print(f"{component:20s}: {mean_time:7.2f}ms ({percentage:5.1f}%)")
    
    print(f"{'=' * 20}   {'=' * 7}    {'=' * 7}")
    print(f"{'TOTAL':20s}: {total_mean:7.2f}ms (100.0%)")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    baseline_time = 3.96  # From previous test
    flexpatch_time = total_mean
    overhead = flexpatch_time - baseline_time
    
    print(f"Baseline detector: {baseline_time:.2f}ms")
    print(f"FlexPatch total: {flexpatch_time:.2f}ms")
    print(f"Overhead: {overhead:.2f}ms ({(overhead/flexpatch_time)*100:.1f}%)")
    
    print("\nBottlenecks:")
    for component, timings in sorted(times.items(), key=lambda x: np.mean(x[1]), reverse=True):
        if component == "total":
            continue
        mean_time = np.mean(timings)
        if mean_time > 10:
            print(f"  • {component}: {mean_time:.2f}ms")
    
    print("\nConclusion:")
    if overhead > baseline_time:
        print("  The overhead from FlexPatch components exceeds the baseline detection time.")
        print("  FlexPatch is designed for scenarios where detection is the bottleneck.")
        print("  Your YOLOv10n model is extremely fast (3.96ms), making FlexPatch overhead visible.")
        print("\n  FlexPatch would be beneficial when:")
        print("    1. Using a slower/larger detector (e.g., YOLOv8x, RT-DETR)")
        print("    2. Running on resource-constrained hardware (mobile/edge devices)")
        print("    3. Processing even higher resolution (e.g., 4K)")
        print("    4. Optimizing the Python overhead with Cython/C++")
    else:
        print("  FlexPatch successfully reduces overall latency!")


if __name__ == "__main__":
    main()


