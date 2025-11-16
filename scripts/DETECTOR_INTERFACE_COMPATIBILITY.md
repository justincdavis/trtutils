# DetectorInterface Compatibility

Both **Remix** and **FlexPatch** now fully implement the `DetectorInterface`, making them **drop-in replacements** for standard TensorRT detectors like YOLO.

## Overview

All three systems now share the same interface:
- **Standard Detector** (YOLO, etc.)
- **FlexPatch** (patch-based detection with tracking)
- **RemixSystem** (adaptive partitioning with selective execution)

## Interface Methods

### Properties

```python
detector.engine          # Underlying TRTEngine
detector.name            # Model name
detector.input_shape     # Expected input shape (H, W)
detector.dtype           # Required data type
```

### Methods

```python
# Preprocessing
preprocessed, ratios, padding = detector.preprocess(image)

# Postprocessing
outputs = detector.postprocess(outputs, ratios, padding)

# Full pipeline (run with prep/post)
outputs = detector.run(image)

# Call operator (same as run)
outputs = detector(image)

# Get formatted detections
detections = detector.get_detections(outputs)

# End-to-end inference (RECOMMENDED)
detections = detector.end2end(image)
# Returns: [(bbox, confidence, class_id), ...]
```

## Usage Examples

### 1. Standard Detector

```python
from trtutils.models import YOLO

detector = YOLO("yolov10m_640.engine", warmup=True)

# End-to-end inference
detections = detector.end2end(image)
for (x1, y1, x2, y2), conf, cls in detections:
    print(f"Class {cls}: {conf:.2f} at ({x1}, {y1}, {x2}, {y2})")
```

### 2. FlexPatch

```python
from trtutils.models import YOLO
from trtutils.research.flexpatch import FlexPatch

base_detector = YOLO("yolov10m_640.engine", warmup=True)

detector = FlexPatch(
    detector=base_detector,
    frame_size=(1920, 1080),
    cluster_size=(640, 360),
    tf_model_path="flexpatch_training/flexpatch_model.joblib",
)

# Same interface as standard detector!
detections = detector.end2end(image)
for (x1, y1, x2, y2), conf, cls in detections:
    print(f"Class {cls}: {conf:.2f} at ({x1}, {y1}, {x2}, {y2})")
```

### 3. Remix

```python
from pathlib import Path
from trtutils.models import YOLO
from trtutils.research.remix import RemixSystem

detectors = [
    YOLO("yolov10n_640.engine", warmup=True),
    YOLO("yolov10s_640.engine", warmup=True),
    YOLO("yolov10m_640.engine", warmup=True),
]
oracle = YOLO("yolov10m_1280.engine", warmup=True)

detector = RemixSystem(
    detectors=detectors,
    oracle=oracle,
    latency_budget=0.050,  # 50ms
    profile_path=Path("remix_data/profiles_mot17.json"),
    plans_path=Path("remix_data/plans_mot17.json"),
)
detector.initialize_runtime()

# Same interface as standard detector!
detections = detector.end2end(image)
for (x1, y1, x2, y2), conf, cls in detections:
    print(f"Class {cls}: {conf:.2f} at ({x1}, {y1}, {x2}, {y2})")
```

## Drop-in Replacement Example

```python
def process_video(detector, video_path):
    """Process video with any detector implementation."""
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Works with YOLO, FlexPatch, or Remix!
        detections = detector.end2end(frame)
        
        for (x1, y1, x2, y2), conf, cls in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# Use with any detector!
detector = YOLO("model.engine")
# detector = FlexPatch(YOLO("model.engine"), frame_size=(1920, 1080))
# detector = RemixSystem(...); detector.initialize_runtime()

process_video(detector, "video.mp4")
```

## Method Behavior

### Standard Detector (YOLO)

- **preprocess**: Letterbox resize, normalize, convert to CHW
- **run**: Full inference pipeline
- **postprocess**: NMS, coordinate scaling
- **end2end**: Complete detection pipeline

### FlexPatch

- **preprocess**: Delegates to wrapped detector
- **run**: Calls `process_frame()` (patch-based detection + tracking)
- **postprocess**: Delegates to wrapped detector
- **end2end**: Calls `process_frame()` (RECOMMENDED)

**Note**: FlexPatch maintains tracking state, so `end2end()` gives best results.

### RemixSystem

- **preprocess**: Delegates to primary detector
- **run**: Calls `run_frame()` (adaptive partitioning + selective execution)
- **postprocess**: Delegates to primary detector  
- **end2end**: Calls `run_frame()` (RECOMMENDED)

**Note**: Remix adapts plans at runtime using PID control, so `end2end()` gives best results.

## Compatibility Notes

### Return Format

All systems return detections in the same format:
```python
detections: list[tuple[tuple[int, int, int, int], float, int]]
# Each detection: ((x1, y1, x2, y2), confidence, class_id)
```

### Properties

| Property | YOLO | FlexPatch | Remix |
|----------|------|-----------|-------|
| `name` | Model name | `FlexPatch(model)` | `RemixSystem` |
| `input_shape` | Model input | Frame size | Primary detector |
| `dtype` | Model dtype | Wrapped detector | Primary detector |
| `engine` | TRTEngine | Wrapped detector | Primary detector |

### Ignored Parameters

Some parameters are ignored by FlexPatch/Remix:

**FlexPatch & Remix ignore:**
- `conf_thres` in `end2end()` (use system's internal thresholds)
- `nms_iou_thres` in `end2end()` (use system's internal NMS)
- `preprocessed` in `run()` (always preprocess)
- `ratios`/`padding` in `run()` (computed internally)

## Testing

Run the compatibility test:

```bash
python test_detector_interface.py
```

This tests all three systems with the same interface:
- ✓ Properties (name, input_shape, dtype, engine)
- ✓ preprocess()
- ✓ postprocess()
- ✓ run()
- ✓ __call__()
- ✓ get_detections()
- ✓ end2end()

## Benefits

### 1. Interchangeable Usage

Switch between detectors without code changes:

```python
# Easy A/B testing
detectors = {
    "baseline": YOLO("model.engine"),
    "flexpatch": FlexPatch(...),
    "remix": remix_instance,
}

for name, detector in detectors.items():
    detections = detector.end2end(image)
    print(f"{name}: {len(detections)} detections")
```

### 2. Unified Benchmarking

```python
def benchmark(detector, images):
    total_time = 0
    total_detections = 0
    
    for img in images:
        start = time.time()
        detections = detector.end2end(img)
        total_time += time.time() - start
        total_detections += len(detections)
    
    return {
        "avg_latency": total_time / len(images),
        "avg_detections": total_detections / len(images),
    }

# Works with all detectors!
stats = benchmark(detector, test_images)
```

### 3. Framework Integration

Both systems can now be used with any framework expecting a `DetectorInterface`:

```python
from trtutils.image.sahi import SAHI

# Works with any detector!
sahi = SAHI(detector, slice_size=640)
detections = sahi.predict(large_image)
```

## Implementation Details

### FlexPatch Compatibility

Added to `src/trtutils/research/flexpatch/_flexpatch.py`:
- Properties: `engine`, `name`, `input_shape`, `dtype`
- Methods: `preprocess()`, `postprocess()`, `run()`, `__call__()`, `get_detections()`, `end2end()`

**Key design**: 
- Wraps a single detector
- `end2end()` calls `process_frame()` (patch-based detection + tracking)
- Maintains tracking state across frames

### Remix Compatibility

Added to `src/trtutils/research/remix/_remix.py`:
- Properties: `engine`, `name`, `input_shape`, `dtype`
- Methods: `preprocess()`, `postprocess()`, `run()`, `__call__()`, `get_detections()`, `end2end()`

**Key design**:
- Wraps multiple detectors
- Uses primary (first) detector for properties
- `end2end()` calls `run_frame()` (adaptive partitioning + selective execution)
- Adapts plans at runtime using PID control

## Recommendations

### Best Practices

1. **Use `end2end()`** for inference (recommended for all systems)
   ```python
   detections = detector.end2end(image)
   ```

2. **System-specific methods** for advanced control
   ```python
   # FlexPatch: Direct control
   detections = flexpatch.process_frame(frame, verbose=True)
   
   # Remix: With latency tracking
   detections, latency = remix.run_frame(frame, verbose=True)
   ```

3. **Check system type** if needed
   ```python
   if isinstance(detector, RemixSystem):
       detections, latency = detector.run_frame(frame)
   else:
       detections = detector.end2end(frame)
   ```

## Summary

✓ **FlexPatch** and **Remix** now fully implement `DetectorInterface`

✓ **Drop-in compatible** with standard TRT detectors

✓ **Same return format**: `[(bbox, conf, cls), ...]`

✓ **Interchangeable usage** in all contexts

✓ **Tested and validated** with compatibility test

---

**Files:**
- Interface: `src/trtutils/image/interfaces.py`
- FlexPatch: `src/trtutils/research/flexpatch/_flexpatch.py`
- Remix: `src/trtutils/research/remix/_remix.py`
- Test: `test_detector_interface.py`

