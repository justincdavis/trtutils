#!/usr/bin/env python3
# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Train FlexPatch model using available MOT17 video with detector as ground truth.

This script generates training data by:
1. Running the detector on MOT17 video to get pseudo ground truth
2. Using FlexPatch's automatic training data generation
3. Training and saving a joblib model
"""

from __future__ import annotations

from pathlib import Path

import cv2

from trtutils.models import YOLO
from trtutils.research.flexpatch import FlexPatch


def main() -> None:
    """Train FlexPatch model from video."""
    print("=" * 80)
    print("FLEXPATCH TRAINING - MOT17 Video")
    print("=" * 80)
    
    # Configuration
    video_path = Path("mot17_13.mp4")
    model_path = Path("data/yolov10/yolov10m_640.engine")
    output_dir = Path("flexpatch_training")
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / "training_data.csv"
    model_save_path = output_dir / "flexpatch_model.joblib"
    
    # Verify paths
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\n📹 Video: {video_path}")
    print(f"🤖 Model: {model_path}")
    print(f"💾 Output: {output_dir}")
    
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
    print("2. LOADING DETECTOR")
    print("=" * 80)
    
    detector = YOLO(
        engine_path=model_path,
        conf_thres=0.25,
        nms_iou_thres=0.45,
    )
    print(f"✓ Detector loaded: {detector.name}")
    print(f"✓ Input shape: {detector.input_shape}")
    
    # Generate pseudo ground truth by running detector on frames
    print("\n" + "=" * 80)
    print("3. GENERATING PSEUDO GROUND TRUTH")
    print("=" * 80)
    
    max_training_frames = 200  # Use first 200 frames for training
    print(f"Processing {max_training_frames} frames...")
    print("(Using detector output as ground truth)")
    
    images = []
    ground_truth = []
    
    for i in range(max_training_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detector to get "ground truth"
        detections = detector.end2end(frame)
        
        images.append(frame)
        ground_truth.append(detections)
        
        if (i + 1) % 20 == 0:
            avg_dets = sum(len(gt) for gt in ground_truth[-20:]) / 20
            print(f"  Processed {i+1}/{max_training_frames} frames (avg {avg_dets:.1f} dets/frame)")
    
    cap.release()
    
    print(f"\n✓ Loaded {len(images)} frames")
    print(f"✓ Total detections: {sum(len(gt) for gt in ground_truth)}")
    print(f"✓ Average detections per frame: {sum(len(gt) for gt in ground_truth) / len(ground_truth):.2f}")
    
    # Generate training data and train model
    print("\n" + "=" * 80)
    print("4. GENERATING TRAINING DATA & TRAINING MODEL")
    print("=" * 80)
    print(f"Processing {len(images)} frames for training...")
    print("This will:")
    print("  1. Track objects across frames")
    print("  2. Compare tracked positions with ground truth")
    print("  3. Extract tracking quality features")
    print("  4. Label samples based on IoU")
    print("  5. Train decision tree model")
    print("\nThis may take several minutes...")
    
    try:
        recommender = FlexPatch.generate_training_data(
            images=images,
            ground_truth=ground_truth,
            detector=detector,
            csv_path=csv_path,
            model_save_path=model_save_path,
            max_age=10,
            iou_threshold_high=0.0,      # IoU = 0 → "high" (complete failure)
            iou_threshold_medium=0.5,    # IoU ≤ 0.5 → "medium" (partial failure)
            train_model=True,
            max_depth=5,
            min_samples_split=10,
            random_state=42,
            verbose=True,
        )
        
        print("\n✓ Training complete!")
        
        if recommender is not None and recommender.model is not None:
            print(f"✓ Model saved to: {model_save_path}")
            print(f"✓ Training data saved to: {csv_path}")
            
            # Show model info
            print(f"\nModel Information:")
            print(f"  • Type: {type(recommender.model).__name__}")
            print(f"  • Max depth: {recommender.model.max_depth}")
            print(f"  • Number of features: 5")
            print(f"  • Features: min_eig, ncc, accel, flow_std, confidence")
            print(f"  • Classes: {list(recommender.model.classes_)}")
            
            # Check CSV
            if csv_path.exists():
                with open(csv_path) as f:
                    lines = f.readlines()
                print(f"\nTraining Data:")
                print(f"  • CSV rows: {len(lines) - 1} (excluding header)")
                print(f"  • CSV size: {csv_path.stat().st_size / 1024:.2f} KB")
                
                # Count label distribution
                label_counts = {"high": 0, "medium": 0, "low": 0}
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        label = line.strip().split(',')[-1]
                        if label in label_counts:
                            label_counts[label] += 1
                
                print(f"  • Label distribution:")
                total = sum(label_counts.values())
                for label, count in label_counts.items():
                    pct = (count / total * 100) if total > 0 else 0
                    print(f"    - {label}: {count} ({pct:.1f}%)")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test the trained model
    print("\n" + "=" * 80)
    print("5. TESTING TRAINED MODEL")
    print("=" * 80)
    
    # Create FlexPatch with trained model
    test_flexpatch = FlexPatch(
        detector=detector,
        frame_size=(width, height),
        tf_model_path=model_save_path,  # Use trained model
    )
    
    print(f"✓ FlexPatch initialized with trained model")
    print(f"✓ Model loaded from: {model_save_path}")
    
    # Test on video
    print("\nTesting on 50 frames...")
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)  # Start from frame 300
    
    test_frames = 50
    detections_count = []
    
    for i in range(test_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = test_flexpatch.process_frame(frame)
        detections_count.append(len(detections))
        
        if (i + 1) % 10 == 0:
            avg = sum(detections_count[-10:]) / 10
            print(f"  Frames {i-8}-{i+1}: {avg:.1f} avg detections")
    
    cap.release()
    
    print(f"\n✓ Tested on {len(detections_count)} frames")
    print(f"✓ Average detections: {sum(detections_count) / len(detections_count):.2f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"✓ Training video: {video_path}")
    print(f"✓ Frames processed: {len(images)}")
    print(f"✓ Training samples: {len(lines) - 1 if csv_path.exists() else 0}")
    print(f"✓ CSV saved: {csv_path}")
    print(f"✓ Model saved: {model_save_path}")
    print(f"✓ Model size: {model_save_path.stat().st_size / 1024:.2f} KB")
    
    print("\n" + "=" * 80)
    print("USAGE INSTRUCTIONS")
    print("=" * 80)
    print("\nTo use the trained model in your code:")
    print(f"""
from trtutils.models import YOLO
from trtutils.research.flexpatch import FlexPatch

# Load detector
detector = YOLO("path/to/yolo.engine")

# Initialize FlexPatch with trained model
flexpatch = FlexPatch(
    detector=detector,
    frame_size=(1920, 1080),
    tf_model_path="{model_save_path}",
)

# Process video
cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    detections = flexpatch.process_frame(frame)
    # Use detections...
""")
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

