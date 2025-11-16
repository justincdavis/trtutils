# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Example usage of FlexPatch for efficient object detection.

This example demonstrates how to use FlexPatch with a TensorRT detector
for efficient real-time object detection on high-resolution video.
"""

from __future__ import annotations

import cv2
import numpy as np

from trtutils.models import YOLO
from trtutils.research.flexpatch import FlexPatch, TrackingFailureRecommender


def main() -> None:
    """Run FlexPatch example."""
    # Initialize detector (using YOLO as an example)
    detector = YOLO(
        engine_path="path/to/yolo.engine",
        conf_thres=0.25,
        nms_iou_thres=0.45,
    )
    
    # Initialize FlexPatch system
    frame_size = (1920, 1080)  # 1080p video
    
    # Option 1: Initialize with saved model
    flexpatch = FlexPatch(
        detector=detector,
        frame_size=frame_size,
        cluster_size=(640, 360),
        cell_size=(20, 22),
        max_age=10,
        tf_ratio=0.75,  # 3:1 ratio of tracking-failure to new-object patches
        tf_model_path="path/to/trained_model.joblib",  # Load from disk
    )
    
    # Option 2: Initialize without model, then set it later
    # flexpatch = FlexPatch(detector, frame_size)
    # model = TrackingFailureRecommender.train_from_csv(
    #     "training_data.csv",
    #     model_save_path="model.joblib"
    # )
    # flexpatch.set_tf_model(model.model)
    
    # Process video frames
    cap = cv2.VideoCapture("path/to/video.mp4")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        detections = flexpatch.process_frame(frame, verbose=True)
        
        # Draw detections
        for bbox, conf, cls_id in detections:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Class {cls_id}: {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        
        # Display result
        cv2.imshow("FlexPatch Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    del detector


def train_tf_model_example() -> None:
    """Example of training a tracking-failure model from CSV."""
    # Create training data CSV with the following format:
    # min_eig,ncc,accel,flow_std,confidence,iou_label
    # 0.002,0.85,2.3,1.2,0.9,low
    # 0.001,0.45,15.7,8.5,0.7,high
    # 0.003,0.72,5.2,3.1,0.8,medium
    
    # Train model and save to disk
    recommender = TrackingFailureRecommender.train_from_csv(
        csv_path="training_data.csv",
        model_save_path="flexpatch_model.joblib",  # Save model
        max_depth=5,
        min_samples_split=10,
        random_state=42,
    )
    
    print("Model trained and saved successfully!")
    print(f"Model saved to: flexpatch_model.joblib")
    
    # Later, load the model
    loaded_recommender = TrackingFailureRecommender(
        model_path="flexpatch_model.joblib"
    )
    print("Model loaded successfully!")


def generate_training_data_example() -> None:
    """Example of automatically generating training data from annotated video."""
    import cv2
    
    # Load video frames
    images = []
    cap = cv2.VideoCapture("path/to/training_video.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        images.append(frame)
    cap.release()
    
    # Load ground truth annotations
    # This could come from COCO format, YOLO format, or custom annotations
    # Format: list of lists, one per frame
    # Each frame has: [(bbox, confidence, class_id), ...]
    ground_truth = [
        [((100, 100, 50, 50), 1.0, 0), ((200, 200, 60, 60), 1.0, 1)],  # Frame 0
        [((105, 102, 50, 50), 1.0, 0), ((198, 205, 60, 60), 1.0, 1)],  # Frame 1
        # ... more frames
    ]
    
    # Initialize detector
    detector = YOLO("path/to/yolo.engine")
    
    # Generate training data and train model automatically
    recommender = FlexPatch.generate_training_data(
        images=images,
        ground_truth=ground_truth,
        detector=detector,
        csv_path="flexpatch_training.csv",
        model_save_path="flexpatch_model.joblib",  # Save model to disk
        max_age=10,
        iou_threshold_high=0.0,      # IoU <= 0.0 → "high" (complete failure)
        iou_threshold_medium=0.5,    # IoU <= 0.5 → "medium" (partial failure)
        train_model=True,             # Train model automatically
        verbose=True,
    )
    
    print(f"✓ Training complete! Model saved to: flexpatch_model.joblib")
    
    # Use the trained model (Option 1: load from disk)
    flexpatch = FlexPatch(
        detector=detector,
        frame_size=(1920, 1080),
        cluster_size=(640, 360),
        tf_model_path="flexpatch_model.joblib",  # Load from disk
    )
    
    # Or Option 2: set model directly
    # flexpatch = FlexPatch(detector, frame_size=(1920, 1080))
    # flexpatch.set_tf_model(recommender.model)
    
    print("✓ FlexPatch ready with trained model!")


if __name__ == "__main__":
    # Run main example
    main()
    
    # Uncomment to run training examples:
    # train_tf_model_example()
    # generate_training_data_example()

