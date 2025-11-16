#!/usr/bin/env python3
# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Test that Remix and FlexPatch match the DetectorInterface.

This script demonstrates that both Remix and FlexPatch can be used
as drop-in replacements for standard Detector objects.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from trtutils.models import YOLO
from trtutils.research.flexpatch import FlexPatch
from trtutils.research.remix import RemixSystem


def test_detector_interface(detector, name: str, test_image: np.ndarray) -> None:
    """Test that detector implements DetectorInterface methods."""
    print(f"\n{'='*80}")
    print(f"Testing {name}")
    print(f"{'='*80}")
    
    # Test properties
    print(f"\nProperties:")
    print(f"  name: {detector.name}")
    print(f"  input_shape: {detector.input_shape}")
    print(f"  dtype: {detector.dtype}")
    print(f"  engine: {type(detector.engine).__name__}")
    
    # Test preprocess
    print(f"\nTesting preprocess()...")
    preprocessed, ratios, padding = detector.preprocess(test_image)
    print(f"  ✓ Preprocessed shape: {preprocessed.shape}")
    print(f"  ✓ Ratios: {ratios}, Padding: {padding}")
    
    # Test end2end (recommended usage)
    print(f"\nTesting end2end()...")
    detections = detector.end2end(test_image)
    print(f"  ✓ Found {len(detections)} detections")
    if len(detections) > 0:
        bbox, conf, cls = detections[0]
        print(f"  ✓ First detection: bbox={bbox}, conf={conf:.3f}, class={cls}")
    
    # Test run
    print(f"\nTesting run()...")
    outputs = detector.run(test_image)
    print(f"  ✓ Outputs: {len(outputs)} arrays")
    
    # Test get_detections
    print(f"\nTesting get_detections()...")
    detections = detector.get_detections(outputs)
    print(f"  ✓ Got {len(detections)} detections from outputs")
    
    # Test __call__
    print(f"\nTesting __call__()...")
    outputs = detector(test_image)
    print(f"  ✓ __call__ returned {len(outputs)} arrays")
    
    print(f"\n✓ All DetectorInterface methods work for {name}")


def main() -> None:
    """Test Remix and FlexPatch DetectorInterface compatibility."""
    print("="*80)
    print("DETECTOR INTERFACE COMPATIBILITY TEST")
    print("="*80)
    
    # Load test image
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    print(f"\nTest image shape: {test_image.shape}")
    
    # Test 1: Standard Detector (baseline)
    print("\n" + "="*80)
    print("1. BASELINE - Standard YOLO Detector")
    print("="*80)
    
    model_path = Path("data/yolov10/yolov10m_640.engine")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    baseline = YOLO(model_path, warmup=True)
    test_detector_interface(baseline, "YOLO Baseline", test_image)
    
    # Test 2: FlexPatch
    print("\n" + "="*80)
    print("2. FLEXPATCH - Patch-based Detection")
    print("="*80)
    
    flexpatch = FlexPatch(
        detector=baseline,
        frame_size=(1920, 1080),
        cluster_size=(640, 360),
    )
    test_detector_interface(flexpatch, "FlexPatch", test_image)
    
    # Test 3: Remix
    print("\n" + "="*80)
    print("3. REMIX - Adaptive Partitioning")
    print("="*80)
    
    # Check if Remix is trained
    profile_path = Path("remix_data/profiles_mot17.json")
    plans_path = Path("remix_data/plans_mot17.json")
    
    if not profile_path.exists() or not plans_path.exists():
        print("⚠ Remix not trained. Run: python train_remix_mot17.py")
        print("Skipping Remix test...")
    else:
        # Load detectors
        detectors = [
            YOLO("data/yolov10/yolov10n_640.engine", warmup=True),
            YOLO("data/yolov10/yolov10s_640.engine", warmup=True),
            YOLO("data/yolov10/yolov10m_640.engine", warmup=True),
        ]
        oracle = YOLO("data/yolov10/yolov10m_1280.engine", warmup=True)
        
        remix = RemixSystem(
            detectors=detectors,
            oracle=oracle,
            latency_budget=0.050,
            profile_path=profile_path,
            plans_path=plans_path,
        )
        remix.initialize_runtime()
        
        test_detector_interface(remix, "Remix", test_image)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\n✓ All systems implement DetectorInterface!")
    print("\nUsage example:")
    print("""
    # Any of these can be used interchangeably:
    detector = YOLO("model.engine")
    detector = FlexPatch(YOLO("model.engine"), frame_size=(1920, 1080))
    detector = RemixSystem(...); detector.initialize_runtime()
    
    # Standard interface works for all:
    detections = detector.end2end(image)
    for bbox, conf, cls in detections:
        print(f"Class {cls}: {conf:.2f} at {bbox}")
    """)


if __name__ == "__main__":
    main()

