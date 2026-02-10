# Tests

## Directory Structure

```
tests/
├── conftest.py                  # Root fixtures (test_images, horse_image, classifier_engine_path, etc.)
├── helpers.py                   # Shared constants and utilities (importable module)
├── ground_truth.py              # Structured ground-truth data for test images
├── common.py                    # build_engine() for simple TRT engine
├── paths.py                     # Engine/ONNX path definitions with TRT version tagging
├── test_engine.py               # Exhaustive TRTEngine tests (basic, backends, pagelocked, CUDA graph)
├── inspect/
│   └── test_inspect.py          # Engine inspection tests
├── models/
│   ├── common.py                # DETECTOR_CONFIG, build_detector(), helper functions
│   ├── paths.py                 # Model-specific engine/ONNX paths
│   ├── test_detector_inference.py  # Unified GPU detector/classifier inference tests
│   ├── test_correctness.py      # Detection correctness and regression tests
│   └── test_download_build_inference.py  # Full download/build/inference workflow
└── dla/
    ├── conftest.py              # Auto-skip if not Jetson
    └── test_detector_dla.py     # DLA-specific detector tests
```

## Markers

| Marker        | Description                                          |
|---------------|------------------------------------------------------|
| `gpu`         | Tests that require a CUDA-capable GPU                |
| `dla`         | Tests that run on NVIDIA DLA cores (Jetson only)     |
| `performance` | Benchmarks and performance regression tests          |
| `slow`        | Tests that take more than 30 seconds                 |
| `download`    | Tests that download models from the internet         |
| `cuda_graph`  | Tests requiring CUDA graph support (async_v3)        |
| `correctness` | Tests that validate detection/classification output  |
| `regression`  | Tests for specific bug regressions                   |

### Usage Examples

```bash
# Run all tests except slow ones
pytest -m "not slow"

# Run only correctness tests
pytest -m correctness

# Run only GPU tests, skip DLA
pytest -m "gpu and not dla"

# Run only performance benchmarks
pytest -m performance

# Run everything except download tests
pytest -m "not download"
```

## Fixtures

- `test_images` - Returns `[horse_image, people_image]` as numpy arrays
- `horse_image` - Returns only the horse image
- `batch_size` - Parametrized: 1, 2, 4
- `random_images` - Factory for random uint8 image arrays
- `classifier_engine_path` - Session-scoped: builds classifier engine once (or None if ONNX unavailable)

## Engine Caching

Test engines are cached in `data/engines/` with TRT version tagging (e.g., `simple_10.14.1.48.engine`).
The `build_engine()` and `build_detector()` functions skip building if the engine already exists.

## Ground Truth

`ground_truth.py` provides structured data for test images:

```python
from tests.ground_truth import HORSE_GT, PEOPLE_GT, HORSE_CLASS_ID

assert len(detections) >= HORSE_GT.min_detections
assert any(det[2] == HORSE_CLASS_ID for det in detections)
```
