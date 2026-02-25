# Phase 5: Download, Inspect, Models Tests (100% Branch Coverage)

**Branch:** `test-overhaul/phase-5-download-inspect-models`
**Goal:** Achieve 100% branch coverage on `src/trtutils/download/`, `src/trtutils/inspect/`, and `src/trtutils/models/`. Add all three to coverage ratchet.

---

## Prerequisites
- Phase 0 (infrastructure)
- Phase 1-2 (core/engine tests — engines available for inspect tests)
- Phase 3 (coverage ratchet operational)

## Deliverables
- [ ] `tests/download/conftest.py` — Download-specific fixtures
- [ ] `tests/download/test_config.py` — Config loading and model listing
- [ ] `tests/download/test_download.py` — Download API, validation, error paths
- [ ] `tests/inspect/conftest.py` — Inspect-specific fixtures
- [ ] `tests/inspect/test_inspect.py` — Engine/ONNX inspection
- [ ] `tests/models/conftest.py` — Model-specific fixtures
- [ ] `tests/models/test_models.py` — Parametrized across all model classes
- [ ] `tests/models/test_detector_correctness.py` — Detector correctness
- [ ] `tests/models/test_classifier_correctness.py` — Classifier correctness
- [ ] `tests/models/test_depth_estimator_correctness.py` — DepthEstimator correctness
- [ ] Coverage ratchet updated for all three modules
- [ ] 100% branch coverage verified

---

## Part A: Download Module

### tests/download/conftest.py

```python
"""Download test fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def download_tmp_dir(tmp_path) -> Path:
    """Temporary directory for download operations."""
    return tmp_path / "downloads"


@pytest.fixture(scope="session")
def model_configs():
    """Load model configurations once per session."""
    from trtutils.download import load_model_configs
    return load_model_configs()


@pytest.fixture(scope="session")
def supported_models():
    """Get all supported model names once per session."""
    from trtutils.download import get_supported_models
    return get_supported_models()
```

### tests/download/test_config.py — Config Loading

**Marker:** `@pytest.mark.cpu`

#### Functions to Test

| Function | Key Branches |
|----------|-------------|
| `load_model_configs()` | All JSON files parse, handles missing files, caching |
| `get_supported_models()` | Returns flat list, no duplicates, caching |

#### Test Cases

```
class TestLoadModelConfigs:
    test_returns_dict                      # Returns dict[str, dict[str, dict]]
    test_all_model_families_present        # All 17 config files loaded
    test_config_structure                  # Each entry has required keys
    test_caching                           # Second call returns same object (lru_cache)
    test_no_empty_families                 # Every family has at least 1 model

    @parametrize("family", [
        "yolov3", "yolov5", "yolov7", "yolov8", "yolov9",
        "yolov10", "yolov11", "yolov12", "yolov13", "yolox",
        "rtdetrv1", "rtdetrv2", "rtdetrv3", "dfine", "deim",
        "deimv2", "rfdetr",
    ])
    test_family_loaded(family)             # Each family exists in configs

class TestGetSupportedModels:
    test_returns_list                      # Returns list[str]
    test_no_duplicates                     # All names unique
    test_count_at_least_100                # At least 100 models (currently 118)
    test_caching                           # lru_cache works
    test_known_models_present              # Spot check: yolov8n, rfdetr_n, etc.

    @parametrize("model_name", [
        "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
        "yolov10n", "yolov11n", "yolov12n", "yolov13n",
        "rtdetrv1_r18", "rtdetrv2_r18", "rtdetrv3_r18",
        "dfine_n", "deim_dfine_n", "deimv2_atto",
        "rfdetr_n", "rfdetr_s", "rfdetr_m",
        "yoloxn", "yolov3tu",
    ])
    test_specific_model_present(model_name)  # Known model in list
```

### tests/download/test_download.py — Download API

**Marker:** Mixed (`@pytest.mark.cpu` for validation, `@pytest.mark.download` + `@pytest.mark.slow` for actual downloads)

#### Functions to Test

| Function | Key Branches |
|----------|-------------|
| `download()` | uv check, temp dir, delegates to download_model, copy output |
| `download_model()` | Model validation, license acceptance, export function routing |
| `_handle_imgsz()` | None → default, enforce mode, adjust_div mode |
| `_check_uv_version()` | Version parse, too old, valid |
| `get_valid_models()` (in models/_utils.py) | Returns models for family |
| `download_model_internal()` | Validates model name, delegates |

#### Validation Tests (CPU, fast)

```
class TestHandleImgsz:
    test_none_uses_default                 # imgsz=None → default
    test_explicit_value_used               # imgsz=512 → 512
    test_enforce_overrides                 # enforce=True + imgsz≠default → default
    test_adjust_div_rounds                 # adjust_div=32 rounds to nearest

class TestModelNameValidation:
    test_valid_model_accepted              # "yolov8n" → no error
    test_invalid_model_raises              # "nonexistent" → ValueError
    test_accept_false_raises               # accept=False → ValueError

class TestGetValidModels:
    @parametrize("model_type,expected_count", [
        pytest.param("yolov8", 5, id="yolov8"),
        pytest.param("yolov10", 11, id="yolov10"),
        pytest.param("rtdetrv1", 7, id="rtdetrv1"),
        pytest.param("rfdetr", 3, id="rfdetr"),
        pytest.param("deimv2", 8, id="deimv2"),
    ])
    test_model_count_per_family            # Correct number per family

class TestExportFunctionRouting:
    # Test that download_model routes to correct export function
    # (These test the routing logic, not actual downloads)
    @parametrize("model,expected_route", [
        pytest.param("yolov8n", "ultralytics", id="ultralytics"),
        pytest.param("yolov7t", "yolov7", id="yolov7"),
        pytest.param("yolov9t", "yolov9", id="yolov9"),
        pytest.param("yolov10n", "yolov10", id="yolov10"),
        pytest.param("rtdetrv1_r18", "rtdetrv1", id="rtdetrv1"),
        pytest.param("rfdetr_n", "rfdetr", id="rfdetr"),
        pytest.param("deimv2_atto", "deimv2", id="deimv2"),
    ])
    test_routing_logic                     # Correct export function selected
```

#### Download Tests (GPU, slow, requires network)

```
@pytest.mark.download
@pytest.mark.slow
class TestDownloadIntegration:
    # NOTE: Only run on CI with network access
    # Select 1-2 small models to minimize download time
    test_download_small_model              # Download a small YOLO model
    test_download_creates_onnx             # Output .onnx file exists
```

---

## Part B: Inspect Module

### tests/inspect/conftest.py

```python
"""Inspect test fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def inspectable_engine_path(build_test_engine) -> Path:
    """Built engine for inspection tests."""
    # tests/inspect/conftest.py -> tests/inspect -> tests -> project_root -> data/
    onnx = Path(__file__).parent.parent.parent / "data" / "simple.onnx"
    return build_test_engine(onnx)


@pytest.fixture(scope="session")
def inspectable_onnx_path() -> Path:
    """ONNX model for layer inspection."""
    path = Path(__file__).parent.parent.parent / "data" / "simple.onnx"
    if not path.exists():
        pytest.skip("Test ONNX not found")
    return path
```

### tests/inspect/test_inspect.py

**Marker:** `@pytest.mark.gpu`

#### Functions to Test

| Function | Key Branches |
|----------|-------------|
| `inspect_engine()` | Path input vs ICudaEngine input; FLAGS.TRT_10 vs older; FLAGS.MEMSIZE_V2 vs older; verbose logging; cleanup on loaded |
| `get_engine_names()` | TRTEngine input vs Path input; multiple I/O tensors |
| `inspect_onnx_layers()` | Path input vs INetworkDefinition; verbose logging |

#### Test Cases

```
class TestInspectEngine:
    test_from_path                         # Accepts Path, returns tuple
    test_returns_correct_structure         # (memsize, batch_size, inputs, outputs)
    test_memsize_positive                  # Memory size > 0
    test_batch_size_valid                  # Batch size >= 1
    test_input_tensors_structure           # Each: (name, shape, dtype, format)
    test_output_tensors_structure          # Each: (name, shape, dtype, format)
    test_input_names_are_strings           # Tensor names are str
    test_verbose_no_error                  # verbose=True works
    test_cleanup_on_path_input             # Engine freed after inspection

class TestGetEngineNames:
    test_from_path                         # Accepts Path → (inputs, outputs)
    test_from_engine_object                # Accepts TRTEngine → (inputs, outputs)
    test_names_are_lists_of_strings        # Both lists contain strings
    test_input_output_count_matches        # Matches engine spec

class TestInspectOnnxLayers:
    test_from_path                         # Accepts Path → list of layer tuples
    test_returns_layer_info                # Each: (idx, name, type, precision)
    test_layer_indices_sequential          # Indices 0, 1, 2, ...
    test_verbose_no_error                  # verbose=True works
    test_from_network                      # Accepts INetworkDefinition
```

---

## Part C: Models Module

### tests/models/conftest.py

```python
"""Model test fixtures — engine paths, test images, ground truth."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import version_engine_path

# tests/models/conftest.py -> tests/models -> tests -> project_root -> data/
DATA_DIR = Path(__file__).parent.parent.parent / "data"
ONNX_DIR = DATA_DIR / "onnx"
ENGINE_DIR = DATA_DIR / "engines"


# Ground truth for detection validation
class GroundTruth:
    """Expected detection results for test images."""

    HORSE = {
        "min_detections": 1,
        "expected_detections": 1,
        "max_detections": 3,
        "required_class_ids": [17],  # COCO horse
    }
    PEOPLE = {
        "min_detections": 2,
        "expected_detections": 4,
        "max_detections": 8,
        "required_class_ids": [0],  # COCO person
    }
```

### tests/models/test_models.py — Parametrized Model Class Tests

**Marker:** `@pytest.mark.cpu` for config validation, `@pytest.mark.gpu` for inference

#### All 19+ Model Classes

```python
from trtutils.models import (
    YOLO, YOLO3, YOLO5, YOLO7, YOLO8, YOLO9, YOLO10, YOLO11, YOLO12, YOLO13,
    YOLOX, DETR, RTDETRv1, RTDETRv2, RTDETRv3, DFINE, DEIM, DEIMv2, RFDETR,
)

# Parametrize across ALL model classes
ALL_MODEL_CLASSES = [
    pytest.param(YOLO, "yolo", id="YOLO"),
    pytest.param(YOLO3, "yolov3tu", id="YOLO3"),
    pytest.param(YOLO5, "yolov5nu", id="YOLO5"),
    pytest.param(YOLO7, "yolov7t", id="YOLO7"),
    pytest.param(YOLO8, "yolov8n", id="YOLO8"),
    pytest.param(YOLO9, "yolov9t", id="YOLO9"),
    pytest.param(YOLO10, "yolov10n", id="YOLO10"),
    pytest.param(YOLO11, "yolov11n", id="YOLO11"),
    pytest.param(YOLO12, "yolov12n", id="YOLO12"),
    pytest.param(YOLO13, "yolov13n", id="YOLO13"),
    pytest.param(YOLOX, "yoloxn", id="YOLOX"),
    pytest.param(RTDETRv1, "rtdetrv1_r18", id="RTDETRv1"),
    pytest.param(RTDETRv2, "rtdetrv2_r18", id="RTDETRv2"),
    pytest.param(RTDETRv3, "rtdetrv3_r18", id="RTDETRv3"),
    pytest.param(DFINE, "dfine_n", id="DFINE"),
    pytest.param(DEIM, "deim_dfine_n", id="DEIM"),
    pytest.param(DEIMv2, "deimv2_atto", id="DEIMv2"),
    pytest.param(RFDETR, "rfdetr_n", id="RFDETR"),
]

YOLO_CLASSES = [c for c in ALL_MODEL_CLASSES if "YOLO" in c.id or c.id == "YOLOX"]
DETR_CLASSES = [c for c in ALL_MODEL_CLASSES if c.id not in [x.id for x in YOLO_CLASSES] and c.id != "YOLO"]
```

#### Test Cases — Config Validation (CPU)

```
@pytest.mark.cpu
class TestModelClassAttributes:
    @parametrize("model_cls,model_name", ALL_MODEL_CLASSES)
    test_has_download_method               # hasattr(model_cls, "download")

    @parametrize("model_cls,model_name", ALL_MODEL_CLASSES)
    test_has_build_method                  # hasattr(model_cls, "build")

    @parametrize("model_cls,model_name", ALL_MODEL_CLASSES)
    test_download_is_static                # isinstance(model_cls.download, staticmethod)

@pytest.mark.cpu
class TestModelDefaults:
    @parametrize("model_cls,model_name", YOLO_CLASSES)
    test_yolo_defaults                     # input_range=(0,1), resize="letterbox", preprocessor="trt"

    @parametrize("model_cls,model_name", DETR_CLASSES)
    test_detr_defaults                     # resize="linear", preprocessor="trt"

@pytest.mark.cpu
class TestModelImgszValidation:
    test_rtdetrv1_enforces_640             # imgsz != 640 → error
    test_rtdetrv2_enforces_640             # imgsz != 640 → error
    test_rtdetrv3_enforces_640             # imgsz != 640 → error
    test_dfine_enforces_640                # imgsz != 640 → error
    test_deimv2_atto_enforces_320          # imgsz != 320 → error
    test_deimv2_femto_enforces_416         # imgsz != 416 → error
    test_rfdetr_divisible_by_32            # imgsz not divisible by 32 → error
    test_rfdetr_valid_imgsz                # 384, 512, 576 all valid
```

### tests/models/test_detector_correctness.py

**Marker:** `@pytest.mark.gpu`, `@pytest.mark.correctness`

Tests that detectors produce correct detections on known images.

```
@pytest.mark.gpu
@pytest.mark.correctness
class TestDetectorCorrectness:
    # NOTE: Each model needs a pre-built engine. Use build_test_engine fixture.
    # Test only models with available ONNX/engines in data/

    @parametrize("model_id", [...available models...])
    test_detects_horse                     # Horse image → class 17 detected

    @parametrize("model_id", [...available models...])
    test_detects_people                    # People image → class 0 detected

    @parametrize("model_id", [...available models...])
    test_detection_count_range             # Count within expected range

    @parametrize("preprocessor", ["cpu", "cuda", "trt"])
    test_preprocessor_parity               # Same detections across preprocessors

    @parametrize("inference_mode", ["end2end", "run"])
    test_inference_modes                   # Both modes produce valid detections
```

### tests/models/test_classifier_correctness.py

**Marker:** `@pytest.mark.gpu`, `@pytest.mark.correctness`

```
@pytest.mark.gpu
@pytest.mark.correctness
class TestClassifierCorrectness:
    test_classifies_horse                  # Horse → equine class
    test_classification_structure          # Returns [(class_id, confidence), ...]
    test_top_k_parameter                   # top_k limits results
    test_confidence_sum_near_one           # Probabilities sum ≈ 1.0
```

### tests/models/test_depth_estimator_correctness.py

**Marker:** `@pytest.mark.gpu`, `@pytest.mark.correctness`

```
@pytest.mark.gpu
@pytest.mark.correctness
class TestDepthEstimatorCorrectness:
    test_output_is_depth_map               # Output shape matches input spatial dims
    test_depth_values_positive             # All depth values > 0
    test_depth_map_dtype                   # Output is float32
    test_batch_processing                  # Batch input works
```

---

## Coverage Verification

```bash
# Download module
pytest tests/download/ --cov=src/trtutils/download --cov-branch --cov-report=term-missing -v

# Inspect module
pytest tests/inspect/ --cov=src/trtutils/inspect --cov-branch --cov-report=term-missing -v

# Models module
pytest tests/models/ --cov=src/trtutils/models --cov-branch --cov-report=term-missing -v

# Update ratchet
python ci/check_coverage.py --update
```

---

## Files Created Summary

| File | Lines (est.) | Tests (est.) | GPU Required |
|------|-------------|-------------|--------------|
| `tests/download/conftest.py` | 30 | — | No |
| `tests/download/test_config.py` | 150 | 25+ | No |
| `tests/download/test_download.py` | 200 | 20+ | Partial |
| `tests/inspect/conftest.py` | 25 | — | Yes |
| `tests/inspect/test_inspect.py` | 200 | 18 | Yes |
| `tests/models/conftest.py` | 50 | — | Partial |
| `tests/models/test_models.py` | 250 | 40+ | Partial |
| `tests/models/test_detector_correctness.py` | 150 | 15 | Yes |
| `tests/models/test_classifier_correctness.py` | 80 | 5 | Yes |
| `tests/models/test_depth_estimator_correctness.py` | 60 | 4 | Yes |
| **Total** | **~1195** | **~127+** | |
