# Remix: Flexible High-resolution Object Detection

Implementation of the Remix system from *"Remix: Flexible High-resolution Object Detection on Edge Devices with Tunable Latency"* (ACM MobiCom '21).

## Overview

Remix enables efficient high-resolution object detection on edge devices by:

1. **Adaptive Non-uniform Image Partitioning**: Divides images into regions optimized for different detector models
2. **Selective Execution**: Dynamically skips uninformative regions using AIMD control
3. **PID-based Plan Control**: Adjusts partition plans at runtime to meet latency budgets

### Key Features

- **1.7×–8.1× speedup** over uniform partitioning
- **≤0.2% accuracy loss** compared to full-frame detection
- **Tunable latency** that adapts to user-specified constraints
- **TensorRT integration** via `DetectorInterface`

## Architecture

```
┌────────────────────────────────┐
│     Historical Frames          │
└──────────────┬─────────────────┘
               │
┌──────────────▼─────────────────┐
│  Adaptive Partition Module      │
│  (Offline / Initialization)     │
├─────────────────────────────────┤
│  1. NN Profiling                │
│  2. Object Distribution Extract │
│  3. Performance Estimation      │
│  4. Partition Plan Generation   │
└──────────────┬─────────────────┘
               │
      Partition Plan Pool
               │
┌──────────────▼─────────────────┐
│     Selective Execution         │
│     (Online / Runtime)          │
├─────────────────────────────────┤
│  1. Partition Selection (AIMD)  │
│  2. Batch Execution             │
│  3. Plan Control (PID loop)     │
│  4. BBox Merging (NMS)          │
└──────────────┬─────────────────┘
               │
         Final Detections
```

## Installation

Remix is part of the `trtutils.research` module:

```python
from trtutils.research.remix import RemixSystem
```

## Usage

### Quick Start

```python
from pathlib import Path
from trtutils.impls.yolo import YOLO
from trtutils.research.remix import RemixSystem

# Initialize detectors (fast to slow)
detectors = [
    YOLO("yolov10n.engine", warmup=True),
    YOLO("yolov10s.engine", warmup=True),
    YOLO("yolov10m.engine", warmup=True),
]

# High-accuracy oracle for profiling
oracle = YOLO("yolov10x.engine", warmup=True)

# Initialize Remix with 50ms latency budget
remix = RemixSystem(
    detectors=detectors,
    oracle=oracle,
    latency_budget=0.050,  # 50ms in seconds
    profile_path="profiles.json",
    plans_path="plans.json",
)

# Profile networks (do this once)
remix.profile_networks(
    coco_path="/path/to/coco",
    max_images=500,
    verbose=True,
)

# Generate partition plans (do this once per scene)
import cv2
historical_frames = [cv2.imread(f"frame_{i}.jpg") for i in range(10)]

remix.generate_plans(
    view_shape=(3840, 2160),  # 4K resolution
    historical_frames=historical_frames,
    max_plans=10,
    verbose=True,
)

# Initialize runtime
remix.initialize_runtime(kp=0.6, ki=0.3, kd=0.1)

# Process video
stats = remix.run_video(
    video_path="input.mp4",
    output_path="output.mp4",
    verbose=True,
)

print(f"Average latency: {stats['avg_latency']*1000:.2f}ms")
print(f"Average detections: {stats['avg_detections']:.1f}")
```

### Single Frame Inference

```python
import cv2

# Read frame
frame = cv2.imread("frame.jpg")

# Run inference
detections, latency = remix.run_frame(frame, verbose=True)

# Process detections
for (x1, y1, x2, y2), confidence, class_id in detections:
    print(f"Class {class_id}: {confidence:.2f} at ({x1}, {y1}, {x2}, {y2})")
```

### Advanced: Custom Profiling

```python
from trtutils.research.remix import NNProfiler

# Profile specific detectors
profiler = NNProfiler(detectors)
profiles = profiler.profile(
    coco_path="/path/to/coco",
    num_latency_runs=50,
    max_images=1000,
    verbose=True,
)

# Save profiles
profiler.save_profiles("custom_profiles.json")

# Load later
loaded_profiles = NNProfiler.load_profiles("custom_profiles.json")
```

### Advanced: Custom Plan Generation

```python
from trtutils.research.remix import (
    AdaptivePartitionPlanner,
    ObjectDistributionExtractor,
    PerformanceEstimator,
)

# Extract distribution
extractor = ObjectDistributionExtractor(oracle)
distribution = extractor.extract(historical_frames, verbose=True)

# Generate plans
estimator = PerformanceEstimator(profiles)
planner = AdaptivePartitionPlanner(detectors, profiles, estimator)

plans = planner.generate(
    view_shape=(3840, 2160),
    distribution=distribution,
    latency_budget=0.050,
    max_plans=20,
    verbose=True,
)

# Save plans
planner.save_plans(plans, "custom_plans.json")
```

### Advanced: Manual Runtime Control

```python
from trtutils.research.remix import (
    SelectiveExecutor,
    PlanController,
    PIDController,
)

# Create detector mapping
detector_map = {d.name: d for d in detectors}

# Initialize executor
executor = SelectiveExecutor(
    detectors=detector_map,
    aimd_increase=1,
    aimd_decrease_factor=0.5,
    nms_iou_thresh=0.5,
)

# Initialize controller
pid = PIDController(kp=0.6, ki=0.3, kd=0.1)
controller = PlanController(plans, pid, latency_budget=0.050)

# Process frame
plan = controller.get_current_plan()
detections = executor.execute(frame, plan)

# Adjust based on actual latency
controller.adjust(actual_latency=0.048)
```

## Components

### NNProfiler
Profiles detector networks for latency and size-specific accuracy using COCO dataset.

### ObjectDistributionExtractor
Analyzes historical frames to determine spatial and size distributions of objects.

### PerformanceEstimator
Estimates accuracy and latency for partition plans without execution.

### AdaptivePartitionPlanner
Generates candidate partition plans using recursive subdivision and dynamic programming.

### SelectiveExecutor
Executes partitions with AIMD-based block skipping to reduce computation.

### PlanController
Adjusts plan selection at runtime using PID feedback control.

## Performance Tuning

### Latency Budget
- Lower budgets → faster detectors, fewer blocks
- Higher budgets → slower detectors, more subdivisions

### PID Parameters
- **Kp (0.4-0.8)**: Proportional gain, how aggressively to respond to error
- **Ki (0.1-0.5)**: Integral gain, correction for sustained error
- **Kd (0.05-0.2)**: Derivative gain, dampening rapid changes

### AIMD Parameters
- **aimd_increase (1-3)**: How quickly to skip uninformative blocks
- **aimd_decrease_factor (0.3-0.7)**: How quickly to re-check skipped blocks

### Plan Generation
- **max_plans (5-20)**: More plans → finer control, higher memory
- **prune_threshold (0.001-0.01)**: Larger → fewer similar plans

## Profiling Workflow

1. **Profile Networks** (once per detector set)
   ```bash
   # Takes ~30 min per detector on 500 COCO images
   remix.profile_networks(coco_path, max_images=500)
   ```

2. **Generate Plans** (once per scene/view)
   ```bash
   # Takes ~1-5 seconds
   remix.generate_plans(view_shape, historical_frames)
   ```

3. **Run Inference** (continuous)
   ```bash
   # Real-time per frame
   remix.run_frame(frame)
   ```

## Limitations

- **No batching**: Each block runs sequentially (TensorRT limitation)
- **Fixed input sizes**: Detectors must handle cropped regions
- **Scene-specific**: Plans optimized for similar content
- **Overhead**: Python implementation adds ~2-5ms per frame

## Citation

If you use Remix in your research, please cite:

```bibtex
@inproceedings{remix2021,
  title={Remix: Flexible High-resolution Object Detection on Edge Devices with Tunable Latency},
  author={...},
  booktitle={ACM MobiCom},
  year={2021}
}
```

## References

- Original Paper: [ACM MobiCom '21](https://doi.org/10.1145/3447993.3483274)
- TensorRT: [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- COCO Dataset: [COCO](https://cocodataset.org/)

