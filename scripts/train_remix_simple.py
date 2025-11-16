#!/usr/bin/env python3
# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Simple Remix training script with minimal configuration.

This script provides a quick way to profile detectors and generate plans
with sensible defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from trtutils.models import YOLO
from trtutils.research.remix import RemixSystem


def main() -> None:
    """Simple Remix training with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile detectors and generate Remix partition plans",
    )
    parser.add_argument(
        "--coco",
        type=Path,
        default=Path("/home/orinagx/research/data/coco"),
        help="Path to COCO dataset",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=Path("data/yolov10"),
        help="Directory containing model engines",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Video file for plan generation (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("remix_data"),
        help="Output directory for profiles and plans",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=50.0,
        help="Latency budget in milliseconds (default: 50ms)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
        help="Max COCO images for profiling (default: 500)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="3840x2160",
        help="Target resolution WxH (default: 3840x2160 for 4K)",
    )
    parser.add_argument(
        "--skip-profiling",
        action="store_true",
        help="Skip profiling if profiles.json exists",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("REMIX SIMPLE TRAINING")
    print("=" * 80)
    
    # Create output directory
    args.output.mkdir(exist_ok=True)
    profile_path = args.output / "profiles.json"
    plans_path = args.output / "plans.json"
    
    # Parse resolution
    w, h = map(int, args.resolution.split("x"))
    view_shape = (w, h)
    
    print(f"\nConfiguration:")
    print(f"  COCO: {args.coco}")
    print(f"  Models: {args.models}")
    print(f"  Output: {args.output}")
    print(f"  Budget: {args.budget}ms")
    print(f"  Max images: {args.max_images}")
    print(f"  Resolution: {w}x{h}")
    
    # Check if profiling can be skipped
    if args.skip_profiling and profile_path.exists():
        print(f"\n✓ Skipping profiling (using existing {profile_path})")
        skip_profiling = True
    else:
        skip_profiling = False
    
    # Load detectors
    print("\nLoading detectors...")
    
    detector_models = [
        ("yolov10n", "yolov10n_640.engine"),
        ("yolov10s", "yolov10s_640.engine"),
        ("yolov10m", "yolov10m_640.engine"),
    ]
    
    detectors = []
    for name, filename in detector_models:
        path = args.models / filename
        if not path.exists():
            print(f"⚠ {name} not found, skipping: {path}")
            continue
        
        print(f"  Loading {name}...")
        detector = YOLO(path, warmup=True)
        detectors.append(detector)
    
    if not detectors:
        print("❌ No detectors loaded")
        return
    
    print(f"✓ Loaded {len(detectors)} detectors")
    
    # Load oracle
    oracle_path = args.models / "yolov10x_640.engine"
    if not oracle_path.exists():
        print(f"❌ Oracle not found: {oracle_path}")
        return
    
    print("  Loading oracle...")
    oracle = YOLO(oracle_path, warmup=True)
    print(f"✓ Oracle loaded: {oracle.name}")
    
    # Initialize Remix
    print("\nInitializing Remix...")
    remix = RemixSystem(
        detectors=detectors,
        oracle=oracle,
        latency_budget=args.budget / 1000.0,  # Convert ms to seconds
        profile_path=profile_path,
        plans_path=plans_path,
        load_existing=skip_profiling,
    )
    
    # Profile networks
    if not skip_profiling:
        print("\nProfiling networks (this may take 15-30 minutes)...")
        
        if not args.coco.exists():
            print(f"❌ COCO dataset not found: {args.coco}")
            return
        
        try:
            remix.profile_networks(
                coco_path=args.coco,
                max_images=args.max_images,
                verbose=True,
            )
            print(f"✓ Profiles saved to {profile_path}")
        except Exception as e:
            print(f"❌ Profiling failed: {e}")
            return
    
    # Load historical frames
    print("\nLoading historical frames...")
    
    if args.video and args.video.exists():
        print(f"  Loading from video: {args.video}")
        cap = cv2.VideoCapture(str(args.video))
        
        frames = []
        for i in range(20):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        print(f"  ✓ Loaded {len(frames)} frames")
        
    else:
        print("  No video provided, using synthetic frames")
        frames = []
        for i in range(10):
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            frames.append(frame)
        print(f"  ✓ Generated {len(frames)} synthetic frames")
    
    # Generate plans
    print("\nGenerating partition plans...")
    
    try:
        remix.generate_plans(
            view_shape=view_shape,
            historical_frames=frames,
            max_plans=10,
            verbose=True,
        )
        print(f"✓ Plans saved to {plans_path}")
    except Exception as e:
        print(f"❌ Plan generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n✓ Profiles: {profile_path}")
    print(f"✓ Plans: {plans_path}")
    print(f"\nRun example:")
    print(f"  python examples/remix_example.py")


if __name__ == "__main__":
    main()

