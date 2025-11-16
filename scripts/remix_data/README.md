# Remix Training Data

This directory stores profiling data and partition plans for the Remix system.

## Files

### `profiles.json`
Network profiling data containing latency and accuracy measurements for each detector.

**Format:**
```json
{
  "detector_name": {
    "latency": 0.045,  // seconds
    "acc_vector": [0.1, 0.2, ..., 0.8]  // 12 accuracy values (one per size bin)
  }
}
```

**Size bins:**
- Bins 0-3: Small objects (S0-S3)
- Bins 4-7: Medium objects (M0-M3)
- Bins 8-11: Large objects (L0-L3)

Bins are based on log2(bbox_area), covering objects from tiny to very large.

### `plans.json`
Partition plans generated for specific view resolution and scene content.

**Format:**
```json
[
  {
    "plan_id": 0,
    "est_ap": 0.75,
    "est_lat": 0.048,
    "blocks": [
      {
        "x1": 0, "y1": 0, "x2": 1920, "y2": 1080,
        "detector_name": "yolov10n",
        "skip_window": 0
      }
    ]
  }
]
```

## Training Workflow

### 1. Profile Networks (Once per detector set)

Run profiling on COCO dataset:

```bash
python train_remix.py
```

This will:
- Measure latency for each detector (20+ runs)
- Evaluate accuracy on COCO validation set (500 images)
- Save results to `profiles.json`

**Time:** ~15-30 minutes depending on hardware

### 2. Generate Plans (Once per scene/resolution)

Plans are generated automatically during training, or can be regenerated:

```bash
python train_remix_simple.py --video path/to/video.mp4 --skip-profiling
```

This will:
- Extract object distributions from video frames
- Generate candidate partition plans
- Prune and select best plans
- Save results to `plans.json`

**Time:** ~10-60 seconds depending on resolution

### 3. Use Trained System

```python
from trtutils.research.remix import RemixSystem

remix = RemixSystem(
    detectors=detectors,
    oracle=oracle,
    latency_budget=0.050,
    profile_path="remix_data/profiles.json",
    plans_path="remix_data/plans.json",
    load_existing=True,
)

remix.initialize_runtime()
detections, latency = remix.run_frame(frame)
```

## Re-training

### When to re-profile networks
- New detector models added/changed
- Different hardware platform
- Significant TensorRT version change

### When to regenerate plans
- Different scene content (e.g., indoor vs outdoor)
- Different resolution
- Different latency budget
- Object distribution changes significantly

## File Sizes

Typical file sizes:
- `profiles.json`: 1-5 KB (depends on number of detectors)
- `plans.json`: 5-20 KB (depends on number of plans and blocks)

Both files are small and can be version controlled.

## Tips

1. **Profiling accuracy**: Use more COCO images (--max-images 1000+) for better accuracy estimates
2. **Plan diversity**: Use video with representative scene content
3. **Historical frames**: 10-20 frames sufficient for distribution extraction
4. **Latency budget**: Start with 50ms, adjust based on requirements
5. **Multiple scenes**: Generate separate plans for different scenes/resolutions

