# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Example usage of Remix for adaptive high-resolution object detection.

This example demonstrates how to use Remix with TensorRT detectors
for efficient real-time object detection under latency constraints.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from trtutils.models import YOLO
from trtutils.research.remix import RemixSystem


def main() -> None:
    """Run Remix example."""
    print("=" * 80)
    print("REMIX EXAMPLE - Adaptive High-Resolution Object Detection")
    print("=" * 80)
    
    # Configuration
    data_dir = Path("remix_data")
    data_dir.mkdir(exist_ok=True)
    
    # Initialize detectors (fast to slow)
    print("\n1. Loading detectors...")
    detectors = [
        YOLO("data/yolov10/yolov10n_640.engine", warmup=True),
        YOLO("data/yolov10/yolov10s_640.engine", warmup=True),
        YOLO("data/yolov10/yolov10m_640.engine", warmup=True),
    ]
    print(f"✓ Loaded {len(detectors)} detectors")
    
    # High-accuracy oracle for profiling
    print("\n2. Loading oracle detector...")
    oracle = YOLO("data/yolov10/yolov10x_640.engine", warmup=True)
    print("✓ Oracle detector loaded")
    
    # Initialize Remix with 50ms latency budget
    print("\n3. Initializing Remix system...")
    remix = RemixSystem(
        detectors=detectors,
        oracle=oracle,
        latency_budget=0.050,  # 50ms in seconds
        profile_path=data_dir / "profiles.json",
        plans_path=data_dir / "plans.json",
        load_existing=True,  # Load if available
    )
    print("✓ Remix system initialized")
    
    # Check if we need to profile networks
    if not remix.profiles:
        print("\n⚠ No existing profiles found. Please run train_remix.py first.")
        print("  This will profile the detectors on COCO dataset.")
        return
    
    print(f"✓ Loaded profiles for {len(remix.profiles)} detectors")
    
    # Check if we need to generate plans
    if not remix.plans:
        print("\n⚠ No existing plans found. Generating plans from video...")
        
        # Load historical frames for plan generation
        video_path = Path("path/to/video.mp4")
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            print("  Please provide a video file for plan generation.")
            return
        
        cap = cv2.VideoCapture(str(video_path))
        historical_frames = []
        
        # Load 10-20 frames for distribution extraction
        print("  Loading historical frames...")
        for i in range(20):
            ret, frame = cap.read()
            if not ret:
                break
            historical_frames.append(frame)
        
        cap.release()
        
        if not historical_frames:
            print("❌ Failed to load frames from video")
            return
        
        # Get view shape from first frame
        h, w = historical_frames[0].shape[:2]
        view_shape = (w, h)
        
        print(f"  Generating plans for {view_shape} resolution...")
        remix.generate_plans(
            view_shape=view_shape,
            historical_frames=historical_frames,
            max_plans=10,
            verbose=True,
        )
        print("✓ Plans generated and saved")
    
    print(f"✓ Loaded {len(remix.plans)} partition plans")
    
    # Initialize runtime components
    print("\n4. Initializing runtime...")
    remix.initialize_runtime(kp=0.6, ki=0.3, kd=0.1)
    print("✓ Runtime initialized")
    
    # Process video
    print("\n5. Processing video...")
    video_path = Path("path/to/video.mp4")
    output_path = Path("remix_output.mp4")
    
    if not video_path.exists():
        print(f"⚠ Video not found: {video_path}")
        print("  Running on webcam instead...")
        video_source = 0
    else:
        video_source = str(video_path)
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("❌ Failed to open video source")
        return
    
    # Setup video writer if processing file
    writer = None
    if video_source != 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    frame_count = 0
    total_latency = 0.0
    total_detections = 0
    
    print("\nProcessing frames (press 'q' to quit)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run Remix inference
        detections, latency = remix.run_frame(frame, verbose=False)
        
        total_latency += latency
        total_detections += len(detections)
        frame_count += 1
        
        # Draw detections
        for (x1, y1, x2, y2), conf, cls_id in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {cls_id}: {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        
        # Draw stats
        avg_lat = (total_latency / frame_count) * 1000  # ms
        stats_text = f"Frame: {frame_count} | Latency: {latency*1000:.1f}ms (avg: {avg_lat:.1f}ms) | Detections: {len(detections)}"
        cv2.putText(
            frame,
            stats_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        
        # Write to output if processing file
        if writer:
            writer.write(frame)
        
        # Display
        cv2.imshow("Remix Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        # Progress update
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames, avg latency: {avg_lat:.1f}ms")
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total frames: {frame_count}")
    print(f"Average latency: {(total_latency / frame_count)*1000:.2f}ms")
    print(f"Average detections: {total_detections / frame_count:.1f}")
    print(f"Target budget: {remix.latency_budget*1000:.2f}ms")
    
    if writer:
        print(f"\n✓ Output saved to: {output_path}")


def quick_inference_example() -> None:
    """Quick example of running inference on a single frame."""
    # Load pre-trained Remix system
    data_dir = Path("remix_data")
    
    detectors = [
        YOLO("data/yolov10/yolov10n_640.engine", warmup=True),
        YOLO("data/yolov10/yolov10m_640.engine", warmup=True),
    ]
    
    oracle = YOLO("data/yolov10/yolov10x_640.engine", warmup=True)
    
    remix = RemixSystem(
        detectors=detectors,
        oracle=oracle,
        latency_budget=0.050,
        profile_path=data_dir / "profiles.json",
        plans_path=data_dir / "plans.json",
    )
    
    # Initialize runtime
    remix.initialize_runtime()
    
    # Load image
    frame = cv2.imread("path/to/image.jpg")
    
    # Run inference
    detections, latency = remix.run_frame(frame)
    
    # Process results
    print(f"Latency: {latency*1000:.2f}ms")
    print(f"Detections: {len(detections)}")
    
    for (x1, y1, x2, y2), confidence, class_id in detections:
        print(f"  Class {class_id}: {confidence:.2f} at ({x1}, {y1}, {x2}, {y2})")


def video_processing_example() -> None:
    """Example of processing entire video with statistics."""
    data_dir = Path("remix_data")
    
    detectors = [
        YOLO("data/yolov10/yolov10n_640.engine", warmup=True),
        YOLO("data/yolov10/yolov10s_640.engine", warmup=True),
        YOLO("data/yolov10/yolov10m_640.engine", warmup=True),
    ]
    
    oracle = YOLO("data/yolov10/yolov10x_640.engine", warmup=True)
    
    remix = RemixSystem(
        detectors=detectors,
        oracle=oracle,
        latency_budget=0.050,
        profile_path=data_dir / "profiles.json",
        plans_path=data_dir / "plans.json",
    )
    
    remix.initialize_runtime()
    
    # Process video with built-in method
    stats = remix.run_video(
        video_path="input.mp4",
        output_path="output.mp4",
        max_frames=None,  # Process all frames
        verbose=True,
    )
    
    print("\nVideo processing statistics:")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Average latency: {stats['avg_latency']*1000:.2f}ms")
    print(f"  Average detections: {stats['avg_detections']:.1f}")
    print(f"  Total time: {stats['total_time']:.2f}s")


if __name__ == "__main__":
    # Run main example
    main()
    
    # Uncomment to run other examples:
    # quick_inference_example()
    # video_processing_example()

