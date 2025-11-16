#!/usr/bin/env python3
# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Profile and train Remix system using COCO dataset.

This script profiles detectors on COCO, extracts object distributions
from video, generates partition plans, and saves everything to disk.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from trtutils.models import YOLO
from trtutils.research.remix import RemixSystem


def main() -> None:
    """Profile detectors and generate partition plans."""
    print("=" * 80)
    print("REMIX TRAINING - Network Profiling & Plan Generation")
    print("=" * 80)
    
    # Configuration
    coco_path = Path("/home/orinagx/research/data/coco")
    data_dir = Path("remix_data")
    data_dir.mkdir(exist_ok=True)
    
    profile_path = data_dir / "profiles.json"
    plans_path = data_dir / "plans.json"
    
    # Model paths
    model_dir = Path("data/yolov10")
    models = {
        "yolov10n": model_dir / "yolov10n_640.engine",
        "yolov10s": model_dir / "yolov10s_640.engine",
        "yolov10m": model_dir / "yolov10m_640.engine",
        "yolov10x": model_dir / "yolov10x_640.engine",  # Oracle
    }
    
    # Verify paths
    print("\n📁 Checking paths...")
    if not coco_path.exists():
        print(f"❌ COCO dataset not found: {coco_path}")
        print(f"   Please download COCO2017 dataset to: {coco_path}")
        print(f"   Expected structure:")
        print(f"     {coco_path}/annotations/instances_val2017.json")
        print(f"     {coco_path}/val2017/")
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
    print(f"✓ All models found")
    print(f"✓ Output directory: {data_dir}")
    
    # Step 1: Load detectors
    print("\n" + "=" * 80)
    print("1. LOADING DETECTORS")
    print("=" * 80)
    
    print("\nLoading candidate detectors...")
    detectors = []
    for name in ["yolov10n", "yolov10s", "yolov10m"]:
        print(f"  Loading {name}...")
        detector = YOLO(
            engine_path=models[name],
            conf_thres=0.25,
            nms_iou_thres=0.45,
            warmup_iterations=5,
            warmup=True,
        )
        detectors.append(detector)
        print(f"  ✓ {detector.name} loaded")
    
    print(f"\n✓ Loaded {len(detectors)} detectors")
    
    print("\nLoading oracle detector...")
    oracle = YOLO(
        engine_path=models["yolov10x"],
        conf_thres=0.25,
        nms_iou_thres=0.45,
        warmup_iterations=5,
        warmup=True,
    )
    print(f"✓ Oracle loaded: {oracle.name}")
    
    # Step 2: Initialize Remix system
    print("\n" + "=" * 80)
    print("2. INITIALIZING REMIX SYSTEM")
    print("=" * 80)
    
    latency_budget = 0.050  # 50ms
    print(f"\nLatency budget: {latency_budget*1000:.0f}ms")
    
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
    print("3. PROFILING NETWORKS ON COCO")
    print("=" * 80)
    print("\nThis will evaluate each detector on COCO validation set.")
    print("Process:")
    print("  1. Measure inference latency (20+ runs per detector)")
    print("  2. Evaluate accuracy across 12 object size bins")
    print("  3. Save profiles to disk")
    print("\n⚠ This may take 15-30 minutes depending on hardware and dataset size.")
    
    # Choose number of images for profiling
    max_images = 500  # Use 500 images for reasonable accuracy
    print(f"\nEvaluating on {max_images} images from COCO validation set...")
    
    try:
        profiles = remix.profile_networks(
            coco_path=coco_path,
            num_latency_runs=20,
            max_images=max_images,
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
            print(f"  Accuracy Vector: {profile['acc_vector'][:4]}... (12 bins)")
        
    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Load video for plan generation
    print("\n" + "=" * 80)
    print("4. LOADING VIDEO FOR PLAN GENERATION")
    print("=" * 80)
    
    video_path = Path("path/to/video.mp4")
    
    # Try to find a video
    possible_videos = [
        Path("mot17_13.mp4"),
        Path("test_video.mp4"),
        Path("sample.mp4"),
    ]
    
    for vid in possible_videos:
        if vid.exists():
            video_path = vid
            break
    
    if not video_path.exists():
        print(f"⚠ No video found. Using synthetic frames for demonstration.")
        print(f"  For production use, provide a video with:")
        print(f"    - Similar resolution to target deployment")
        print(f"    - Representative scene content")
        print(f"    - 10-20 frames sufficient for distribution extraction")
        
        # Create synthetic frames (for demonstration)
        import numpy as np
        historical_frames = []
        view_shape = (3840, 2160)  # 4K
        
        print(f"\nGenerating {10} synthetic frames at {view_shape}...")
        for i in range(10):
            frame = np.random.randint(0, 255, (*view_shape[::-1], 3), dtype=np.uint8)
            historical_frames.append(frame)
        
        print(f"✓ Created {len(historical_frames)} synthetic frames")
    else:
        print(f"✓ Found video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        view_shape = (w, h)
        
        print(f"\nVideo properties:")
        print(f"  Resolution: {w}x{h}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Total frames: {total_frames}")
        
        # Load historical frames for distribution extraction
        num_historical = 20
        print(f"\nLoading {num_historical} frames for object distribution extraction...")
        
        historical_frames = []
        frame_indices = list(range(0, min(total_frames, 1000), 50))[:num_historical]
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            historical_frames.append(frame)
            
            if len(historical_frames) % 5 == 0:
                print(f"  Loaded {len(historical_frames)} frames...")
        
        cap.release()
        
        print(f"✓ Loaded {len(historical_frames)} historical frames")
    
    # Step 5: Generate partition plans
    print("\n" + "=" * 80)
    print("5. GENERATING PARTITION PLANS")
    print("=" * 80)
    print("\nThis will:")
    print("  1. Run oracle detector on historical frames")
    print("  2. Extract object size distributions")
    print("  3. Generate partition plans (full-frame + subdivisions)")
    print("  4. Prune and select best plans")
    print("  5. Save plans to disk")
    
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
        print(f"Generated {len(plans)} partition plans:\n")
        
        for plan in plans:
            detectors_used = plan.get_detector_names()
            lat_ms = plan.est_lat * 1000
            within_budget = "✓" if plan.est_lat <= latency_budget else "✗"
            
            print(f"Plan {plan.plan_id}:")
            print(f"  Blocks: {plan.num_blocks}")
            print(f"  Detectors: {', '.join(detectors_used)}")
            print(f"  Est. Accuracy: {plan.est_ap:.3f}")
            print(f"  Est. Latency: {lat_ms:.2f}ms {within_budget}")
            print()
        
    except Exception as e:
        print(f"\n❌ Plan generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Summary and usage instructions
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    print(f"\n✓ Detectors profiled: {len(detectors)}")
    print(f"✓ Profiles saved: {profile_path}")
    print(f"✓ Profile size: {profile_path.stat().st_size / 1024:.2f} KB")
    
    print(f"\n✓ Partition plans generated: {len(plans)}")
    print(f"✓ Plans saved: {plans_path}")
    print(f"✓ Plans size: {plans_path.stat().st_size / 1024:.2f} KB")
    
    print("\n" + "=" * 80)
    print("USAGE INSTRUCTIONS")
    print("=" * 80)
    print("\nTo use the trained Remix system:")
    print(f"""
from pathlib import Path
from trtutils.models import YOLO
from trtutils.research.remix import RemixSystem

# Load detectors
detectors = [
    YOLO("data/yolov10/yolov10n_640.engine", warmup=True),
    YOLO("data/yolov10/yolov10s_640.engine", warmup=True),
    YOLO("data/yolov10/yolov10m_640.engine", warmup=True),
]

oracle = YOLO("data/yolov10/yolov10x_640.engine", warmup=True)

# Initialize Remix with saved profiles and plans
remix = RemixSystem(
    detectors=detectors,
    oracle=oracle,
    latency_budget=0.050,  # 50ms
    profile_path=Path("{profile_path}"),
    plans_path=Path("{plans_path}"),
    load_existing=True,  # Load saved data
)

# Initialize runtime
remix.initialize_runtime()

# Process video
stats = remix.run_video(
    video_path="input.mp4",
    output_path="output.mp4",
    verbose=True,
)

print(f"Average latency: {{stats['avg_latency']*1000:.2f}}ms")
print(f"Average detections: {{stats['avg_detections']:.1f}}")
""")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Run the example: python examples/remix_example.py")
    print("2. Adjust latency budget in RemixSystem if needed")
    print("3. Re-generate plans for different scenes/resolutions")
    print("4. Tune PID parameters (kp, ki, kd) for your use case")
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

