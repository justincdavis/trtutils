# Remix Usage Guide

Complete guide for using Remix adaptive object detection system.

## Quick Start

### 1. Train (Profile Networks & Generate Plans)

```bash
# Full training with COCO profiling
python train_remix.py

# Or simple training with command-line options
python train_remix_simple.py \
    --coco /path/to/coco \
    --models data/yolov10 \
    --video sample_video.mp4 \
    --budget 50 \
    --resolution 3840x2160
```

**Time:** 15-30 minutes for profiling + 1 minute for plan generation

**Output:**
- `remix_data/profiles.json` - Network profiles (latency + accuracy)
- `remix_data/plans.json` - Partition plans

### 2. Run Inference

```bash
# Run example on video
python examples/remix_example.py
```

## Training Details

### Network Profiling

Profiling measures two key metrics for each detector:

1. **Latency**: Average inference time (seconds)
   - Measured over 20+ runs on dummy inputs
   - Warmup period to stabilize GPU

2. **Accuracy Vector**: Size-specific accuracy (12 bins)
   - Evaluated on COCO validation set
   - Bins based on log2(object_area)
   - Bins 0-3: Small, 4-7: Medium, 8-11: Large

**Example profile:**
```json
{
  "yolov10n": {
    "latency": 0.008,
    "acc_vector": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
  }
}
```

### Plan Generation

Plans define how to partition the image and which detector to use for each region.

**Process:**
1. Extract object distribution from historical frames
2. Generate candidate plans:
   - Full-frame with each detector
   - 2x2, 3x3, 4x4 grid subdivisions
   - Horizontal/vertical splits
   - Homogeneous (same detector) and heterogeneous combinations
3. Prune similar plans (within 1ms latency)
4. Select best plans by accuracy within budget

**Example plan:**
```json
{
  "plan_id": 0,
  "est_ap": 0.75,
  "est_lat": 0.048,
  "blocks": [
    {"x1": 0, "y1": 0, "x2": 1920, "y2": 1080, "detector_name": "yolov10n"}
  ]
}
```

## Runtime Behavior

### Selective Execution (AIMD)

Remix dynamically skips uninformative regions using AIMD control:

- **Additive Increase**: If block has no detections → increase skip window
- **Multiplicative Decrease**: If detections found → reset skip window
- **Skip**: Don't run detector while skip window > 0
- **Decrease**: Reduce skip window by 1 each frame

**Benefits:**
- Saves computation on static regions
- Focuses resources on active areas
- Adapts to scene changes

### Plan Control (PID)

Adjusts plan selection to meet latency budget:

1. Measure actual latency
2. Compute error: `e = budget - actual`
3. Update PID: `control = Kp*e + Ki*∫e + Kd*de/dt`
4. Adjust target budget
5. Select plan closest to adjusted budget

**PID Parameters:**
- **Kp (0.6)**: Proportional - immediate response
- **Ki (0.3)**: Integral - corrects sustained error
- **Kd (0.1)**: Derivative - dampens oscillations

## Configuration

### Latency Budget

Target inference time per frame.

```python
remix = RemixSystem(
    detectors=detectors,
    oracle=oracle,
    latency_budget=0.050,  # 50ms
)
```

**Recommendations:**
- Real-time (30 fps): 33ms
- High frame rate (60 fps): 16ms
- Video processing: 50-100ms
- Interactive applications: 20-30ms

### Detector Selection

Choose detectors with varying speed/accuracy:

```python
detectors = [
    YOLO("yolov10n_640.engine"),  # Fast, lower accuracy
    YOLO("yolov10s_640.engine"),  # Balanced
    YOLO("yolov10m_640.engine"),  # Slower, higher accuracy
]

oracle = YOLO("yolov10x_640.engine")  # Highest accuracy
```

**Guidelines:**
- 3-5 detectors provide good coverage
- Oracle should be most accurate (for profiling)
- Faster detectors used for large subdivisions
- Slower detectors for full-frame or critical regions

### PID Tuning

Adjust PID parameters for your use case:

```python
remix.initialize_runtime(
    kp=0.6,  # Increase for faster response
    ki=0.3,  # Increase if consistently over/under budget
    kd=0.1,  # Increase to reduce oscillations
)
```

**Tuning tips:**
- Start with defaults (0.6, 0.3, 0.1)
- If latency oscillates: increase Kd
- If consistently off-target: increase Ki
- If response too slow: increase Kp

### AIMD Parameters

Adjust block skipping behavior:

```python
executor = SelectiveExecutor(
    detectors=detector_map,
    aimd_increase=1,        # How fast to skip blocks
    aimd_decrease_factor=0.5,  # How fast to re-check
    nms_iou_thresh=0.5,     # Detection merging
)
```

## Advanced Usage

### Multiple Scenes

Generate separate plans for different scenes:

```python
# Indoor scene
remix_indoor = RemixSystem(
    detectors=detectors,
    oracle=oracle,
    latency_budget=0.050,
    plans_path="plans_indoor.json",
)
remix_indoor.generate_plans(view_shape, indoor_frames)

# Outdoor scene
remix_outdoor = RemixSystem(
    detectors=detectors,
    oracle=oracle,
    latency_budget=0.050,
    plans_path="plans_outdoor.json",
)
remix_outdoor.generate_plans(view_shape, outdoor_frames)
```

### Custom Resolution

Generate plans for specific resolution:

```python
# 4K (3840x2160)
remix.generate_plans(
    view_shape=(3840, 2160),
    historical_frames=frames_4k,
)

# 1080p (1920x1080)
remix.generate_plans(
    view_shape=(1920, 1080),
    historical_frames=frames_1080p,
)
```

### Re-profiling

Re-profile networks when:
- New detector added
- Hardware changed
- TensorRT version updated

```python
# Force new profiling
remix = RemixSystem(
    detectors=detectors,
    oracle=oracle,
    latency_budget=0.050,
    profile_path="profiles.json",
    load_existing=False,  # Don't load existing
)

remix.profile_networks(coco_path, max_images=1000)
```

### Inspection

Inspect generated plans:

```python
# Load plans
from trtutils.research.remix import AdaptivePartitionPlanner

plans = AdaptivePartitionPlanner.load_plans("remix_data/plans.json")

for plan in plans:
    print(f"Plan {plan.plan_id}:")
    print(f"  Blocks: {plan.num_blocks}")
    print(f"  Detectors: {plan.get_detector_names()}")
    print(f"  Est. Accuracy: {plan.est_ap:.3f}")
    print(f"  Est. Latency: {plan.est_lat*1000:.2f}ms")
    
    for block in plan.blocks:
        print(f"    Block: {block.coords}, detector: {block.detector_name}")
```

## Performance Optimization

### Tips for Best Results

1. **Profile with representative data**
   - Use sufficient COCO images (500+)
   - Ensure good GPU utilization during profiling

2. **Choose good historical frames**
   - Use 10-20 frames from target scene
   - Include variety of object positions/sizes
   - Should be representative of deployment

3. **Tune for your hardware**
   - Profile on target device
   - Adjust latency budget for device capabilities
   - Consider power constraints

4. **Monitor runtime behavior**
   - Track actual vs target latency
   - Watch plan switching frequency
   - Observe AIMD skip patterns

### Expected Speedup

From paper: **1.7×–8.1× speedup** over uniform partitioning

Factors affecting speedup:
- Scene complexity
- Object distribution
- Latency budget
- Detector speed differences

## Troubleshooting

### Issue: Latency exceeds budget

**Solutions:**
- Lower latency budget (use faster plans)
- Add faster detectors to pool
- Reduce max_plans (less runtime overhead)
- Tune PID parameters (increase Kp)

### Issue: Poor accuracy

**Solutions:**
- Use slower/more accurate detectors
- Increase latency budget
- Re-generate plans with better historical frames
- Check oracle accuracy on COCO

### Issue: Plans not switching

**Solutions:**
- Increase Ki (integral gain)
- Verify multiple plans within budget
- Check if latency consistently on-target

### Issue: Oscillating latency

**Solutions:**
- Increase Kd (derivative gain)
- Decrease Kp (proportional gain)
- Reduce plan pool diversity

## Comparison with Alternatives

### vs. Full-Frame Detection
- **Remix**: Adaptive, meets latency budget
- **Full-Frame**: Fixed cost, may be too slow for high-res

### vs. Uniform Partitioning (SAHI)
- **Remix**: Non-uniform, detector-aware
- **SAHI**: Uniform grid, same detector everywhere

### vs. FlexPatch
- **Remix**: No tracking, partition-based
- **FlexPatch**: Tracking-based, patch recommendation

## References

- Paper: *Remix: Flexible High-resolution Object Detection on Edge Devices with Tunable Latency* (ACM MobiCom '21)
- Implementation: `/home/orinagx/trtutils/src/trtutils/research/remix/`
- Examples: `/home/orinagx/trtutils/examples/remix_example.py`

