#!/usr/bin/env python3
# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Train FlexPatch model using MOT17Det dataset.

This script loads MOT17Det data using cv2ext, generates training data,
and trains a tracking-failure model for FlexPatch.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from cv2ext.datasets.mot import MOT17Det

from trtutils.models import YOLO
from trtutils.research.flexpatch import FlexPatch


def main() -> None:
    """Train FlexPatch model from MOT17Det dataset."""
    print("=" * 80)
    print("FLEXPATCH TRAINING - MOT17Det Dataset")
    print("=" * 80)
    
    # Configuration
    data_root = Path("/home/orinagx/research/data/mot17det")
    model_path = Path("data/yolov10/yolov10m_640.engine")
    output_dir = Path("flexpatch_training")
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / "training_data.csv"
    model_save_path = output_dir / "flexpatch_model.joblib"
    
    # Verify paths
    if not data_root.exists():
        print(f"❌ Dataset not found: {data_root}")
        return
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\n📁 Dataset: {data_root}")
    print(f"🤖 Model: {model_path}")
    print(f"💾 Output: {output_dir}")
    
    # Load MOT17Det dataset
    print("\n" + "=" * 80)
    print("1. LOADING MOT17Det DATASET")
    print("=" * 80)
    
    try:
        dataset = MOT17Det(root=str(data_root))
        print(f"✓ Dataset loaded")
        print(f"  • Number of sequences: {len(dataset.sequences)}")
        
        # Show available sequences
        print("\n  Available sequences:")
        for seq in dataset.sequences:
            print(f"    - {seq}")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Select training sequences (use first few for training)
    print("\n" + "=" * 80)
    print("2. SELECTING TRAINING SEQUENCES")
    print("=" * 80)
    
    # Use 2-3 sequences for training to get diverse data
    num_training_seqs = min(3, len(dataset.sequences))
    training_sequences = dataset.sequences[:num_training_seqs]
    
    print(f"Training on {num_training_seqs} sequences:")
    for seq in training_sequences:
        print(f"  • {seq}")
    
    # Load frames and ground truth from training sequences
    print("\n" + "=" * 80)
    print("3. LOADING FRAMES AND GROUND TRUTH")
    print("=" * 80)
    
    all_images = []
    all_ground_truth = []
    max_frames_per_seq = 100  # Limit frames per sequence to manage memory
    
    for seq_name in training_sequences:
        print(f"\nProcessing sequence: {seq_name}")
        
        try:
            # Get sequence data
            seq_data = dataset.get_sequence(seq_name)
            
            if seq_data is None:
                print(f"  ⚠ Skipping {seq_name} - no data returned")
                continue
            
            # Load frames
            seq_dir = data_root / seq_name
            img_dir = seq_dir / "img1"
            
            if not img_dir.exists():
                print(f"  ⚠ Image directory not found: {img_dir}")
                continue
            
            # Get ground truth
            gt_file = seq_dir / "gt" / "gt.txt"
            if not gt_file.exists():
                print(f"  ⚠ Ground truth not found: {gt_file}")
                continue
            
            # Load ground truth data
            # MOT format: frame, id, x, y, w, h, conf, class, visibility
            gt_data = {}
            with open(gt_file) as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 7:
                        continue
                    
                    frame_id = int(parts[0])
                    x, y, w, h = map(float, parts[2:6])
                    conf = float(parts[6]) if len(parts) > 6 else 1.0
                    cls_id = int(parts[7]) if len(parts) > 7 else 1
                    
                    if frame_id not in gt_data:
                        gt_data[frame_id] = []
                    
                    # Convert to int for bbox
                    bbox = (int(x), int(y), int(w), int(h))
                    gt_data[frame_id].append((bbox, conf, cls_id))
            
            # Load images
            img_files = sorted(img_dir.glob("*.jpg"))[:max_frames_per_seq]
            
            print(f"  • Found {len(img_files)} frames")
            print(f"  • Loading up to {max_frames_per_seq} frames...")
            
            seq_images = []
            seq_gt = []
            
            for idx, img_path in enumerate(img_files, start=1):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                seq_images.append(img)
                
                # Get ground truth for this frame
                frame_gt = gt_data.get(idx, [])
                seq_gt.append(frame_gt)
                
                if len(seq_images) % 20 == 0:
                    print(f"    Loaded {len(seq_images)} frames...")
            
            print(f"  ✓ Loaded {len(seq_images)} frames from {seq_name}")
            
            all_images.extend(seq_images)
            all_ground_truth.extend(seq_gt)
            
        except Exception as e:
            print(f"  ❌ Error processing {seq_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_images:
        print("\n❌ No frames loaded. Cannot proceed with training.")
        return
    
    print(f"\n✓ Total frames loaded: {len(all_images)}")
    print(f"✓ Total ground truth annotations: {sum(len(gt) for gt in all_ground_truth)}")
    
    # Load detector
    print("\n" + "=" * 80)
    print("4. LOADING DETECTOR")
    print("=" * 80)
    
    detector = YOLO(
        engine_path=model_path,
        conf_thres=0.25,
        nms_iou_thres=0.45,
    )
    print(f"✓ Detector loaded: {detector.name}")
    
    # Generate training data and train model
    print("\n" + "=" * 80)
    print("5. GENERATING TRAINING DATA & TRAINING MODEL")
    print("=" * 80)
    print(f"Processing {len(all_images)} frames...")
    print(f"This may take several minutes...")
    
    try:
        recommender = FlexPatch.generate_training_data(
            images=all_images,
            ground_truth=all_ground_truth,
            detector=detector,
            csv_path=csv_path,
            model_save_path=model_save_path,
            max_age=10,
            iou_threshold_high=0.0,      # Complete failure
            iou_threshold_medium=0.5,    # Partial failure
            train_model=True,
            max_depth=5,
            min_samples_split=10,
            random_state=42,
            verbose=True,
        )
        
        print("\n✓ Training complete!")
        
        if recommender is not None:
            print(f"✓ Model saved to: {model_save_path}")
            print(f"✓ Training data saved to: {csv_path}")
            
            # Show model info
            if recommender.model is not None:
                print(f"\nModel Information:")
                print(f"  • Type: {type(recommender.model).__name__}")
                print(f"  • Max depth: {recommender.model.max_depth}")
                print(f"  • Features: 5 (min_eig, ncc, accel, flow_std, confidence)")
                print(f"  • Classes: {recommender.model.classes_}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test the trained model
    print("\n" + "=" * 80)
    print("6. TESTING TRAINED MODEL")
    print("=" * 80)
    
    # Load the saved model
    from trtutils.research.flexpatch import TrackingFailureRecommender
    
    test_recommender = TrackingFailureRecommender(model_path=model_save_path)
    print(f"✓ Model loaded from disk: {model_save_path}")
    
    # Test on a small video
    print("\nTesting on video...")
    test_video = Path("mot17_13.mp4")
    
    if test_video.exists():
        test_flexpatch = FlexPatch(
            detector=detector,
            frame_size=(1920, 1080),
            tf_model_path=model_save_path,  # Use trained model
        )
        
        cap = cv2.VideoCapture(str(test_video))
        test_frames = 10
        
        print(f"Processing {test_frames} test frames...")
        for i in range(test_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = test_flexpatch.process_frame(frame)
            print(f"  Frame {i+1}: {len(detections)} detections")
        
        cap.release()
        print("✓ Model tested successfully on video")
    else:
        print(f"⚠ Test video not found: {test_video}")
        print("  Skipping video test")
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"✓ Training sequences: {len(training_sequences)}")
    print(f"✓ Frames processed: {len(all_images)}")
    print(f"✓ CSV saved: {csv_path}")
    print(f"✓ Model saved: {model_save_path}")
    print(f"✓ Model size: {model_save_path.stat().st_size / 1024:.2f} KB")
    
    print("\n" + "=" * 80)
    print("USAGE INSTRUCTIONS")
    print("=" * 80)
    print("\nTo use the trained model:")
    print(f"""
from trtutils.models import YOLO
from trtutils.research.flexpatch import FlexPatch

detector = YOLO("path/to/yolo.engine")
flexpatch = FlexPatch(
    detector=detector,
    frame_size=(1920, 1080),
    tf_model_path="{model_save_path}",  # Use trained model
)

# Process video frames
for frame in video_frames:
    detections = flexpatch.process_frame(frame)
""")
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

