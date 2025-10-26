# FlexPatch Implementation

Complete implementation of the FlexPatch paper for efficient real-time object detection on high-resolution video.

**Paper:** [FlexPatch: Fast and Accurate Object Detection in Heterogeneous Drone Imagery](https://juheonyi.github.io/files/FlexPatch.pdf)

## Overview

FlexPatch enables real-time object detection on high-resolution video (e.g., 1080p) by combining tracking and selective patch-based detection. Instead of running detection on every full frame, FlexPatch:

1. Tracks objects between frames using optical flow
2. Identifies patches where tracking might fail or new objects may appear
3. Packs high-priority patches into a compact cluster
4. Runs detection only on the cluster (much faster than full frame)
5. Maps detections back to the original frame

## Components

### ObjectTracker
- Optical flow-based tracking using Lucas-Kanade
- Tracks bounding boxes between frames
- Extracts tracking quality features (min eigenvalue, NCC, acceleration, flow std)
- Configurable box aging and removal

### TrackingFailureRecommender
- Recommends patches where tracking likely failed
- Uses ML (sklearn DecisionTreeClassifier) for priority prediction
- Static method `train_from_csv()` for model training
- Falls back to heuristics if no model provided

### NewObjectRecommender
- Detects regions where new objects might appear
- Uses Canny edge detection per cell
- Tracks refresh interval (frames since last detection)
- Priority formula: `min(50, RI) + W × (EI > T)`

### PatchAggregator
- Guillotine 2D bin packing algorithm
- Packs high-priority patches into compact clusters
- Supports ratio-based packing (e.g., 3:1 tracking-failure to new-object)

### FlexPatch
- Main orchestrator integrating all components
- Works with any `DetectorInterface` implementation
- Handles initialization, tracking, patch selection, and detection

## Installation

```bash
# Install trtutils with FlexPatch dependencies
pip install trtutils[flexpatch]
```

## Usage

### Basic Usage

```python
from trtutils.models import YOLO
from trtutils.research.flexpatch import FlexPatch

# Initialize detector
detector = YOLO("path/to/yolo.engine", conf_thres=0.25)

# Initialize FlexPatch
flexpatch = FlexPatch(
    detector=detector,
    frame_size=(1920, 1080),  # 1080p
    cluster_size=(640, 360),
    cell_size=(20, 22),
    max_age=10,
    tf_ratio=0.75,  # 3:1 ratio TF:NO patches
)

# Process video frames
for frame in video_frames:
    detections = flexpatch.process_frame(frame)
    # detections: list[tuple[tuple[int, int, int, int], float, int]]
    # Format: [(bbox, confidence, class_id), ...]
```

### Training Tracking-Failure Model

### Automatic Training Data Generation (Recommended)

FlexPatch provides a static method to automatically generate training data from annotated videos:

```python
import cv2
from trtutils.models import YOLO
from trtutils.research.flexpatch import FlexPatch

# Load images and ground truth
images = [cv2.imread(f"frame_{i:04d}.jpg") for i in range(100)]
ground_truth = load_your_annotations()  # Format: list[list[tuple]]

# Generate training data and train model
detector = YOLO("yolo.engine")
recommender = FlexPatch.generate_training_data(
    images=images,
    ground_truth=ground_truth,
    detector=detector,
    csv_path="training.csv",
    iou_threshold_high=0.0,    # Complete failure
    iou_threshold_medium=0.5,  # Partial failure
    train_model=True,          # Auto-train
    verbose=True,
)

# Use immediately
flexpatch = FlexPatch(detector, frame_size=(1920, 1080))
flexpatch.set_tf_model(recommender.model)
```

### Manual Training from CSV

Alternatively, create a CSV file with tracking features and IoU labels:

```csv
min_eig,ncc,accel,flow_std,confidence,iou_label
0.002,0.85,2.3,1.2,0.9,low
0.001,0.45,15.7,8.5,0.7,high
0.003,0.72,5.2,3.1,0.8,medium
```

Labels:
- `high`: IoU = 0 (complete tracking failure)
- `medium`: 0 < IoU ≤ 0.5 (partial failure)
- `low`: IoU > 0.5 (tracking succeeded)

Train and use the model:

```python
from trtutils.research.flexpatch import TrackingFailureRecommender

# Train model from CSV
recommender = TrackingFailureRecommender.train_from_csv(
    csv_path="training_data.csv",
    max_depth=5,
    min_samples_split=10,
    random_state=42,
)

# Use with FlexPatch
flexpatch.set_tf_model(recommender.model)
```

## Parameters

### FlexPatch Parameters

- `detector`: DetectorInterface instance
- `frame_size`: (width, height) of input frames
- `cluster_size`: (width, height) of patch cluster (default: 640×360)
- `cell_size`: (width, height) of cells for new-object detection (default: 20×22)
- `max_age`: Maximum frames to track object without re-detection (default: 10)
- `tf_ratio`: Ratio of tracking-failure to new-object patches (default: 0.75)
- `use_ratio_packing`: Whether to use ratio-based packing (default: True)

### ObjectTracker Parameters

- `max_age`: Maximum age before removing tracked objects (default: 10)
- `feature_params`: OpenCV goodFeaturesToTrack parameters
- `lk_params`: OpenCV calcOpticalFlowPyrLK parameters

### NewObjectRecommender Parameters

- `frame_size`: (width, height) of frames
- `cell_size`: (width, height) of detection cells (default: 20×22)
- `edge_threshold`: Edge pixel ratio threshold (default: 0.1)
- `refresh_weight`: Max refresh interval weight (default: 50)
- `edge_weight`: Weight for high-edge cells (default: 10)

## Performance

Expected performance improvements over full-frame detection:

- **Latency**: ~85% reduction (1029ms → 139ms for 1080p→640×360)
- **Accuracy**: Up to 146% AP improvement over tracking-only
- **Power**: ~63% reduction in power consumption

## Architecture

```
Video Frame
    ↓
ObjectTracker (optical flow)
    ↓
┌─────────────────────────────────┐
│ Patch Recommendation            │
│  ├─ TrackingFailureRecommender  │
│  └─ NewObjectRecommender        │
└─────────────────────────────────┘
    ↓
PatchAggregator (bin packing)
    ↓
Patched Detection (DetectorInterface)
    ↓
Map back to full frame
    ↓
Update tracker & return detections
```

## Code Style

Follows trtutils conventions:
- Copyright header with MIT license
- Type hints with `from __future__ import annotations`
- Google-style docstrings
- `typing_extensions.Self` for self-reference
- Compatible with Python 3.8+

## References

Yi, J., Kim, Y., Choi, Y., Kim, Y. (2020). "FlexPatch: Fast and Accurate Object Detection in Heterogeneous Drone Imagery." arXiv preprint arXiv:2006.07417.

