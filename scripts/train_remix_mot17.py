#!/usr/bin/env python3
# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Train Remix system using COCO for profiling and MOT17 for plan generation.

This script:
1. Profiles detectors on COCO dataset (accuracy + latency)
2. Extracts object distributions from MOT17 training sequences
3. Generates partition plans optimized for MOT17 content
4. Saves all data for runtime inference and validation
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from trtutils.models import YOLO
from trtutils.research.remix import RemixSystem


def main() -> None:
    """Train Remix with COCO profiling and MOT17 plan generation."""
    print("=" * 80)
    print("REMIX TRAINING - COCO Profiling + MOT17 Plan Generation")
    print("=" * 80)
    
    # Configuration
    coco_path = Path("/home/orinagx/research/datasets/coco17")
    mot17_path = Path("/home/orinagx/research/datasets/mot17/train")
    model_dir = Path("data/yolov10")
    output_dir = Path("remix_data")
    output_dir.mkdir(exist_ok=True)
    
    profile_path = output_dir / "profiles_mot17.json"
    plans_path = output_dir / "plans_mot17.json"
    
    # Model configuration
    models = {
        "yolov10n": model_dir / "yolov10n_640.engine",
        "yolov10s": model_dir / "yolov10s_640.engine",
        "yolov10m": model_dir / "yolov10m_640.engine",
        "oracle": model_dir / "yolov10m_1280.engine",  # Use 1280 as oracle
    }
    
    latency_budget = 0.050  # 50ms
    max_coco_images = 500  # Number of COCO images for profiling
    
    # Verify paths
    print("\n📁 Verifying paths...")
    if not coco_path.exists():
        print(f"❌ COCO dataset not found: {coco_path}")
        return
    
    if not mot17_path.exists():
        print(f"❌ MOT17 dataset not found: {mot17_path}")
        return
    
    missing_models = []
    for name, path in models.items():
        if not path.exists():
            missing_models.append(f"{name}: {path}")
    
    if missing_models:
        print(f"❌ Missing models:")
        for model in missing_models:
            print(f"   - {model}")
        return
    
    print(f"✓ COCO dataset: {coco_path}")
    print(f"✓ MOT17 dataset: {mot17_path}")
    print(f"✓ All models found")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Latency budget: {latency_budget*1000:.0f}ms")
    
    # Step 1: Load detectors
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DETECTORS")
    print("=" * 80)
    
    print("\nLoading candidate detectors (640x640)...")
    detectors = []
    for name in ["yolov10n", "yolov10s", "yolov10m"]:
        print(f"  Loading {name}...")
        detector = YOLO(
            engine_path=models[name],
            conf_thres=0.25,
            nms_iou_thres=0.45,
            warmup_iterations=10,
            warmup=True,
        )
        detectors.append(detector)
        print(f"  ✓ {detector.name} - Input: {detector.input_shape}")
    
    print(f"\n✓ Loaded {len(detectors)} candidate detectors")
    
    print("\nLoading oracle detector (1280x1280)...")
    oracle = YOLO(
        engine_path=models["oracle"],
        conf_thres=0.25,
        nms_iou_thres=0.45,
        warmup_iterations=10,
        warmup=True,
    )
    print(f"✓ Oracle loaded: {oracle.name} - Input: {oracle.input_shape}")
    
    # Step 2: Initialize Remix
    print("\n" + "=" * 80)
    print("STEP 2: INITIALIZING REMIX SYSTEM")
    print("=" * 80)
    
    remix = RemixSystem(
        detectors=detectors,
        oracle=oracle,
        latency_budget=latency_budget,
        profile_path=profile_path,
        plans_path=plans_path,
        load_existing=False,  # Force new profiling
    )
    print("✓ Remix system initialized")
    
    # Step 3: Profile networks on COCO
    print("\n" + "=" * 80)
    print("STEP 3: PROFILING NETWORKS ON COCO")
    print("=" * 80)
    print("\nThis will:")
    print("  1. Measure inference latency (20 runs per detector)")
    print("  2. Evaluate accuracy on COCO across 12 object size bins")
    print("  3. Save profiles to disk")
    print(f"\nEvaluating on {max_coco_images} images from COCO validation set...")
    print("⚠ This may take 20-30 minutes depending on hardware.\n")
    
    try:
        profiles = remix.profile_networks(
            coco_path=coco_path,
            num_latency_runs=20,
            max_images=max_coco_images,
            save=True,
            verbose=True,
        )
        
        print("\n✓ Network profiling complete!")
        print(f"✓ Profiles saved to: {profile_path}")
        
        # Display profile summary
        print("\n" + "-" * 80)
        print("PROFILE SUMMARY")
        print("-" * 80)
        for name, profile in profiles.items():
            lat_ms = profile['latency'] * 1000
            mean_acc = sum(profile['acc_vector']) / len(profile['acc_vector'])
            print(f"\n{name}:")
            print(f"  Latency: {lat_ms:.2f}ms")
            print(f"  Mean Accuracy: {mean_acc:.3f}")
            
            # Show size-specific accuracy
            acc_vec = profile['acc_vector']
            print(f"  Small objects (bins 0-3): {np.mean(acc_vec[0:4]):.3f}")
            print(f"  Medium objects (bins 4-7): {np.mean(acc_vec[4:8]):.3f}")
            print(f"  Large objects (bins 8-11): {np.mean(acc_vec[8:12]):.3f}")
        
    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Load MOT17 frames for distribution extraction
    print("\n" + "=" * 80)
    print("STEP 4: LOADING MOT17 FRAMES FOR PLAN GENERATION")
    print("=" * 80)
    
    # Get MOT17 sequences
    sequences = sorted([d for d in mot17_path.iterdir() if d.is_dir()])
    
    # Use FRCNN variants (best detection baseline in MOT17)
    frcnn_sequences = [s for s in sequences if "FRCNN" in s.name]
    
    if not frcnn_sequences:
        print("⚠ No FRCNN sequences found, using all sequences")
        selected_sequences = sequences[:3]
    else:
        selected_sequences = frcnn_sequences[:3]
    
    print(f"\nSelected sequences for distribution extraction:")
    for seq in selected_sequences:
        print(f"  • {seq.name}")
    
    # Load historical frames from MOT17
    print("\nLoading frames...")
    historical_frames = []
    frames_per_seq = 10  # Load 10 frames per sequence
    view_shape = None
    
    for seq_dir in selected_sequences:
        img_dir = seq_dir / "img1"
        if not img_dir.exists():
            continue
        
        img_files = sorted(img_dir.glob("*.jpg"))
        
        # Load frames uniformly spaced across sequence
        indices = np.linspace(0, len(img_files) - 1, frames_per_seq, dtype=int)
        
        print(f"  Loading {frames_per_seq} frames from {seq_dir.name}...")
        for idx in indices:
            img_path = img_files[idx]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            historical_frames.append(img)
            
            # Set view shape from first frame
            if view_shape is None:
                h, w = img.shape[:2]
                view_shape = (w, h)
        
        if len(historical_frames) >= 30:  # Limit total frames
            break
    
    print(f"✓ Loaded {len(historical_frames)} frames")
    print(f"✓ View shape: {view_shape}")
    
    # Step 5: Generate partition plans
    print("\n" + "=" * 80)
    print("STEP 5: GENERATING PARTITION PLANS")
    print("=" * 80)
    print("\nThis will:")
    print("  1. Run oracle detector on historical frames")
    print("  2. Extract object size distributions")
    print("  3. Generate candidate partition plans")
    print("  4. Prune and select best plans")
    print("  5. Save plans to disk\n")
    
    try:
        plans = remix.generate_plans(
            view_shape=view_shape,
            historical_frames=historical_frames,
            max_plans=10,
            save=True,
            verbose=True,
        )
        
        print("\n✓ Partition plan generation complete!")
        print(f"✓ Plans saved to: {plans_path}")
        
        # Display plan summary
        print("\n" + "-" * 80)
        print("PLAN SUMMARY")
        print("-" * 80)
        print(f"Generated {len(plans)} partition plans for {view_shape} resolution:\n")
        
        for plan in plans:
            detectors_used = plan.get_detector_names()
            lat_ms = plan.est_lat * 1000
            within_budget = "✓" if plan.est_lat <= latency_budget else "✗"
            
            print(f"Plan {plan.plan_id}:")
            print(f"  Blocks: {plan.num_blocks}")
            print(f"  Detectors: {', '.join(detectors_used)}")
            print(f"  Est. Accuracy: {plan.est_ap:.3f}")
            print(f"  Est. Latency: {lat_ms:.2f}ms {within_budget}")
            
            # Show block details for non-full-frame plans
            if plan.num_blocks > 1:
                print(f"  Block layout:")
                for i, block in enumerate(plan.blocks):
                    print(f"    Block {i}: {block.coords}, detector: {block.detector_name}")
            print()
        
    except Exception as e:
        print(f"\n❌ Plan generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    print(f"\n✓ Detectors profiled: {len(detectors)}")
    print(f"✓ Profiles saved: {profile_path}")
    print(f"✓ Profile size: {profile_path.stat().st_size / 1024:.2f} KB")
    
    print(f"\n✓ Partition plans generated: {len(plans)}")
    print(f"✓ Plans saved: {plans_path}")
    print(f"✓ Plans size: {plans_path.stat().st_size / 1024:.2f} KB")
    
    print(f"\n✓ View shape: {view_shape}")
    print(f"✓ Latency budget: {latency_budget*1000:.0f}ms")
    print(f"✓ Historical frames used: {len(historical_frames)}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Validate on MOT17:")
    print("   python validate_remix.py")
    print("\n2. Run on video:")
    print("   python examples/remix_example.py")
    print("\n3. Compare with baseline detection")
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

