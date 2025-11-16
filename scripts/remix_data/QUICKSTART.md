# Remix Quick Start

## One-Time Setup

### 1. Profile Networks (~20 minutes)

```bash
python train_remix_simple.py \
    --coco /path/to/coco \
    --models data/yolov10 \
    --max-images 500
```

**Output:** `remix_data/profiles.json`

### 2. Generate Plans (~1 minute)

```bash
python train_remix_simple.py \
    --video your_video.mp4 \
    --resolution 1920x1080 \
    --budget 50 \
    --skip-profiling
```

**Output:** `remix_data/plans.json`

## Runtime Usage

### Python Script

```python
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

# Initialize with saved data
remix = RemixSystem(
    detectors=detectors,
    oracle=oracle,
    latency_budget=0.050,  # 50ms
    profile_path=Path("remix_data/profiles.json"),
    plans_path=Path("remix_data/plans.json"),
)

remix.initialize_runtime()

# Process video
stats = remix.run_video("input.mp4", output_path="output.mp4")
print(f"Avg latency: {stats['avg_latency']*1000:.2f}ms")
```

### Single Frame

```python
import cv2

frame = cv2.imread("image.jpg")
detections, latency = remix.run_frame(frame)

for (x1, y1, x2, y2), conf, cls_id in detections:
    print(f"Class {cls_id}: {conf:.2f} at ({x1},{y1},{x2},{y2})")
```

## When to Re-train

### Re-profile (profiles.json)
- ✓ New detectors added
- ✓ Different hardware
- ✓ TensorRT version changed

### Re-generate plans (plans.json)
- ✓ Different scene (indoor → outdoor)
- ✓ Different resolution (1080p → 4K)
- ✓ Different latency budget

## Command Reference

### Full Training
```bash
python train_remix.py
```

### Quick Training
```bash
python train_remix_simple.py --coco /path/to/coco --video video.mp4
```

### Run Example
```bash
python examples/remix_example.py
```

## Key Parameters

### Latency Budget
```python
latency_budget=0.050  # 50ms
```
- 16ms for 60fps
- 33ms for 30fps
- 50-100ms for video processing

### PID Tuning
```python
remix.initialize_runtime(kp=0.6, ki=0.3, kd=0.1)
```
- Kp: Response speed
- Ki: Steady-state correction
- Kd: Oscillation dampening

## File Locations

```
remix_data/
├── profiles.json       # Network profiles (1-5 KB)
├── plans.json          # Partition plans (5-20 KB)
└── README.md           # Detailed documentation
```

## Expected Performance

- **Speedup**: 1.7×–8.1× over uniform partitioning
- **Accuracy loss**: ≤0.2% vs full-frame detection
- **Latency**: Meets user-specified budget

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Latency too high | Lower budget, add faster detectors |
| Poor accuracy | Increase budget, use better detectors |
| Plans not switching | Increase Ki (integral gain) |
| Oscillating latency | Increase Kd (derivative gain) |

## More Information

- Full usage guide: `REMIX_USAGE.md`
- Implementation: `src/trtutils/research/remix/`
- Paper: ACM MobiCom '21

