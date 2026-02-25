# Phase 6: Image Module Port

**Branch:** `test-overhaul/phase-6-image-port`
**Goal:** Port all existing image submodule tests into the new structure, add missing coverage (SAHI, dynamic batch, schemas), achieve 100% branch coverage, remove `tests/legacy/`.

---

## Prerequisites
- Phase 0 (infrastructure)
- Phase 1-2 (core/engine fixtures available)
- Phase 3 (coverage ratchet operational)
- Phases 4-5 (all other modules tested)

## Deliverables
- [ ] `tests/image/conftest.py` — Ported and expanded image fixtures
- [ ] `tests/image/test_preproc.py` — Port preprocessor tests
- [ ] `tests/image/test_postproc.py` — Port postprocessor tests
- [ ] `tests/image/test_image_model.py` — ImageModel base class
- [ ] `tests/image/test_detector.py` — Detector class tests
- [ ] `tests/image/test_classifier.py` — Classifier class tests
- [ ] `tests/image/test_depth_estimator.py` — DepthEstimator class tests
- [ ] `tests/image/test_sahi.py` — NEW: SAHI tests
- [ ] `tests/image/kernels/conftest.py` — Ported kernel fixtures
- [ ] `tests/image/kernels/test_letterbox.py` — Port letterbox kernel tests
- [ ] `tests/image/kernels/test_linear.py` — Port linear resize kernel tests
- [ ] `tests/image/kernels/test_sst.py` — Port SST kernel tests (all 3 variants)
- [ ] `tests/image/kernels/test_performance.py` — Port performance benchmarks
- [ ] `tests/image/onnx/test_preproc_engine.py` — Port TRT preproc tests
- [ ] Coverage ratchet updated for `trtutils/image`
- [ ] 100% branch coverage verified
- [ ] `tests/legacy/` removed

---

## Porting Strategy

**Rule:** Preserve existing test assertions and expected values. Restructure into new parametrized patterns. Do NOT change what the tests are checking — only HOW they are organized.

### What to Preserve
- All numerical tolerance values (rtol, atol)
- All expected shapes, dtypes, ranges
- All parity assertions (CPU == CUDA == TRT)
- All performance comparisons (GPU faster than CPU)
- All edge case handling (empty detections, batch parity)

### What to Change
- Test organization: classes per feature, not per test type
- Fixtures: use new conftest hierarchy
- Parametrization: unify duplicate test patterns
- Markers: add `@pytest.mark.gpu`, `@pytest.mark.performance`

---

## Image Conftest (tests/image/conftest.py)

Port from `tests/legacy/image/conftest.py` with expansions:

```python
"""Image test fixtures — preprocessor factories, output mock generators."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants (ported from legacy)
# ---------------------------------------------------------------------------
PREPROC_SIZE = (640, 640)
PREPROC_RANGE = (0.0, 1.0)
PREPROC_DTYPE = np.float32
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Tolerance for CPU/GPU parity
CUDA_MAG_BOUNDS = 0.01  # Max per-element difference

# ---------------------------------------------------------------------------
# Parametrized fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(
    params=["cpu", "cuda", "trt"],
    ids=["preproc_cpu", "preproc_cuda", "preproc_trt"],
)
def preprocessor_type(request):
    """Parametrized preprocessor type."""
    return request.param


@pytest.fixture(
    params=["linear", "letterbox"],
    ids=["resize_linear", "resize_letterbox"],
)
def resize_method(request):
    """Parametrized resize method."""
    return request.param


# ---------------------------------------------------------------------------
# Preprocessor factory (ported from legacy)
# ---------------------------------------------------------------------------
@pytest.fixture
def make_preprocessor():
    """Factory fixture: create preprocessor by type."""
    def _make(
        ptype: str,
        size: tuple[int, int] = PREPROC_SIZE,
        input_range: tuple[float, float] = PREPROC_RANGE,
        dtype: np.dtype = PREPROC_DTYPE,
        resize_method: str = "letterbox",
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        batch_size: int = 1,
    ):
        if ptype == "cpu":
            from trtutils.image.preprocessors import CPUPreprocessor
            return CPUPreprocessor(
                size=size, input_range=input_range, dtype=dtype,
                resize=resize_method, mean=mean, std=std,
            )
        elif ptype == "cuda":
            from trtutils.image.preprocessors import CUDAPreprocessor
            return CUDAPreprocessor(
                size=size, input_range=input_range, dtype=dtype,
                resize=resize_method, mean=mean, std=std,
                batch_size=batch_size,
            )
        elif ptype == "trt":
            from trtutils.image.preprocessors import TRTPreprocessor
            return TRTPreprocessor(
                size=size, input_range=input_range, dtype=dtype,
                resize=resize_method, mean=mean, std=std,
                batch_size=batch_size,
            )
        raise ValueError(f"Unknown preprocessor type: {ptype}")
    return _make


# ---------------------------------------------------------------------------
# Output mock generators (ported from legacy)
# ---------------------------------------------------------------------------
@pytest.fixture
def make_yolov10_output():
    """Factory: generate mock YOLOv10 engine output."""
    rng = np.random.default_rng(42)

    def _make(batch_size: int = 1, num_dets: int = 10):
        # (batch, 300, 6) with (x1,y1,x2,y2,score,class_id)
        output = np.zeros((batch_size, 300, 6), dtype=np.float32)
        for b in range(batch_size):
            for i in range(num_dets):
                x1, y1 = rng.uniform(0, 400, 2)
                w, h = rng.uniform(20, 200, 2)
                output[b, i] = [x1, y1, x1 + w, y1 + h, rng.uniform(0.1, 1.0), rng.integers(0, 80)]
        return [output]
    return _make


@pytest.fixture
def make_efficient_nms_output():
    """Factory: generate mock EfficientNMS engine output."""
    rng = np.random.default_rng(42)

    def _make(batch_size: int = 1, num_dets: int = 10):
        num_dets_arr = np.full((batch_size, 1), num_dets, dtype=np.int32)
        bboxes = rng.uniform(0, 640, (batch_size, 100, 4)).astype(np.float32)
        scores = rng.uniform(0, 1, (batch_size, 100, 1)).astype(np.float32)
        class_ids = rng.integers(0, 80, (batch_size, 100, 1)).astype(np.int32)
        return [num_dets_arr, bboxes, scores, class_ids]
    return _make


@pytest.fixture
def make_rfdetr_output():
    """Factory: generate mock RF-DETR engine output."""
    rng = np.random.default_rng(42)

    def _make(
        batch_size: int = 1,
        num_queries: int = 300,
        num_classes: int = 80,
        num_dets: int = 10,
    ):
        dets = rng.uniform(0, 640, (batch_size, num_queries, 4)).astype(np.float32)
        labels = rng.uniform(0, 1, (batch_size, num_queries, num_classes)).astype(np.float32)
        return [dets, labels]
    return _make


@pytest.fixture
def make_detr_output():
    """Factory: generate mock DETR engine output."""
    rng = np.random.default_rng(42)

    def _make(
        batch_size: int = 1,
        num_queries: int = 300,
        num_dets: int = 10,
    ):
        scores = rng.uniform(0, 1, (batch_size, num_queries, 81)).astype(np.float32)
        labels = rng.integers(0, 80, (batch_size, num_queries)).astype(np.int64)
        boxes = rng.uniform(0, 640, (batch_size, num_queries, 4)).astype(np.float32)
        return [scores, labels, boxes]
    return _make


@pytest.fixture
def make_classification_output():
    """Factory: generate mock classification engine output."""
    rng = np.random.default_rng(42)

    def _make(batch_size: int = 1, num_classes: int = 1000):
        logits = rng.standard_normal((batch_size, num_classes)).astype(np.float32)
        return [logits]
    return _make


# ---------------------------------------------------------------------------
# Ratios/padding factory
# ---------------------------------------------------------------------------
@pytest.fixture
def make_ratios_padding():
    """Factory: generate dummy ratios and padding for postprocessing."""
    def _make(batch_size: int = 1):
        ratios = [(1.0, 1.0)] * batch_size
        padding = [(0.0, 0.0)] * batch_size
        return ratios, padding
    return _make
```

---

## Detailed Test Plans

### tests/image/test_preproc.py — Preprocessor Tests

**Marker:** `@pytest.mark.gpu`

#### Tests to Port (from legacy test_preproc.py + test_model_classes.py)

```
class TestPreprocessorLoads:
    # PORT: TestPreprocessorLoads from legacy
    @parametrize("preprocessor_type", ["cpu", "cuda", "trt"])
    test_load_without_normalization        # Load with mean=None, std=None
    test_load_with_imagenet_normalization  # Load with mean/std

class TestPreprocessorDeterminism:
    # PORT: TestPreprocessorDuplicate from legacy
    @parametrize("preprocessor_type", ["cpu", "cuda", "trt"])
    test_same_input_same_output            # Deterministic results
    test_same_input_same_output_imagenet   # With mean/std

class TestPreprocessorParity:
    # PORT: TestPreprocessorParity from legacy
    # KEY: CPU results == CUDA/TRT results within tolerance
    @parametrize("gpu_type", ["cuda", "trt"])
    @parametrize("resize_method", ["linear", "letterbox"])
    @parametrize("use_imagenet", [False, True], ids=["no_norm", "imagenet"])
    test_gpu_matches_cpu                   # np.allclose(cpu, gpu, atol=CUDA_MAG_BOUNDS)
    test_ratios_match                      # Ratios identical across backends
    test_padding_matches                   # Padding identical across backends

class TestPreprocessorAPI:
    # PORT: TestPreprocessorAPI from legacy test_model_classes.py
    @parametrize("preprocessor_type", ["cpu", "cuda", "trt"])
    test_accepts_list_input                # list[np.ndarray] works
    test_accepts_ndarray_input             # Single np.ndarray works
    test_output_shape_single               # (1, 3, H, W) for single image
    test_output_shape_batch                # (N, 3, H, W) for batch
    test_output_dtype                      # float32
    test_output_range                      # Values in expected range
    test_ratio_padding_types               # Ratios/padding are list of tuples

class TestBatchProcessing:
    # PORT: TestBatchProcessing from legacy
    @parametrize("preprocessor_type", ["cpu", "cuda", "trt"])
    @parametrize("batch_size", [1, 2, 4])
    test_batch_output_shape                # Shape matches (batch, 3, H, W)
    test_batch_parity_with_single          # Batch result == stacked single results
    test_cuda_dynamic_reallocation         # Changing batch sizes works (CUDA)

class TestPerformance:
    # PORT: TestPerformance from legacy
    @pytest.mark.performance
    @parametrize("gpu_type", ["cuda", "trt"])
    test_gpu_faster_than_cpu               # GPU preprocessing faster
    test_gpu_pagelocked_faster             # Pagelocked memory improves GPU speed
```

---

### tests/image/test_postproc.py — Postprocessor Tests

**Marker:** `@pytest.mark.gpu` (some can be `@pytest.mark.cpu`)

#### Tests to Port (from legacy test_postproc.py)

```
class TestYOLOv10Postproc:
    # PORT: TestYOLOv10 from legacy
    test_single_image                      # batch_size=1
    test_batch                             # batch_size=4
    test_batch_parity                      # Batch == stacked singles
    test_confidence_threshold              # High threshold filters detections
    test_empty_detections                  # No detections → empty lists

class TestEfficientNMSPostproc:
    # PORT: TestEfficientNMS from legacy
    test_single_image
    test_batch
    test_batch_parity
    test_zero_detections                   # num_dets=0 → empty

class TestRFDETRPostproc:
    # PORT: TestRFDETR from legacy
    test_single_image
    test_batch
    test_batch_parity
    test_with_input_size                   # input_size parameter

class TestDETRPostproc:
    # PORT: TestDETR from legacy
    test_single_image
    test_batch
    test_batch_parity
    test_confidence_threshold

class TestClassificationPostproc:
    # PORT: TestClassifications from legacy
    test_single_image
    test_batch
    test_batch_parity

class TestGetDetections:
    # PORT: TestGetDetections from legacy
    test_single_image                      # Returns list[list[tuple]]
    test_batch                             # Returns list per image
    test_confidence_filtering              # conf_thres filters
    test_structure                         # Each: ((x1,y1,x2,y2), score, class_id)

class TestGetClassifications:
    # PORT: TestGetClassifications from legacy
    test_single_image
    test_batch
    @parametrize("top_k", [1, 5, 10])
    test_top_k                             # Limits number of results

class TestDifferentRatiosPerImage:
    # PORT: TestDifferentRatiosPerImage from legacy
    test_varying_ratios_affect_output      # Different ratios → different bbox coords
```

---

### tests/image/test_image_model.py — ImageModel Base Class

**Marker:** `@pytest.mark.gpu`

#### NEW tests (not in legacy)

```
class TestImageModelInit:
    @parametrize("preprocessor", ["cpu", "cuda", "trt"])
    test_init_with_preprocessor_types      # All 3 preprocessor backends

    @parametrize("resize_method", ["linear", "letterbox"])
    test_init_with_resize_methods          # Both resize methods

    @parametrize("backend", ["auto", "async_v3", "async_v2"])
    test_init_with_backends                # All execution backends

class TestImageModelPreprocessing:
    test_preprocess_single_image           # np.ndarray input
    test_preprocess_batch                  # list[np.ndarray] input
    test_preprocess_output_shape           # Correct output dimensions

class TestImageModelUpdates:
    test_update_input_range                # Change range, preprocessors rebuilt
    test_update_mean_std                   # Change normalization, preprocessors rebuilt

class TestImageModelUtilities:
    test_get_random_input                  # Generates valid random image
    test_mock_run                          # Mock run without real inference

class TestImageModelCUDAGraph:
    @pytest.mark.cuda_graph
    test_cuda_graph_integration            # cuda_graph=True works with ImageModel
```

---

### tests/image/test_detector.py — Detector Class

**Marker:** `@pytest.mark.gpu`

#### NEW tests for schema handling and full pipeline

```
class TestDetectorSchemaDetection:
    # Test automatic schema detection from engine I/O
    test_auto_detect_yolo_schema           # YOLO engine → correct schemas
    test_auto_detect_efficient_nms_schema  # EfficientNMS output detected
    test_auto_detect_detr_schema           # DETR-style output detected

class TestDetectorInference:
    @parametrize("mode", ["end2end", "run"])
    test_inference_modes                   # Both modes produce valid output

    @parametrize("preprocessor", ["cpu", "cuda", "trt"])
    test_preprocessor_variants             # All preprocessors work

    test_preprocessed_input                # preprocessed=True skips preprocessing
    test_postprocess_false                 # postprocess=False returns raw output
    test_no_copy_flag                      # no_copy=True returns refs

class TestDetectorBatch:
    @parametrize("batch_size", [1, 2, 4])
    test_batch_processing                  # Batch sizes work correctly
    test_batch_parity                      # Batch == stacked singles

class TestDetectorEnd2End:
    test_end2end_single                    # Full pipeline, single image
    test_end2end_batch                     # Full pipeline, batch
    test_end2end_returns_detections        # Returns list[list[tuple]]

class TestDetectorMultiInput:
    # Test models with extra inputs (RT-DETR, RT-DETRv3)
    test_detr_extra_inputs                 # orig_target_sizes handled
    test_rtdetrv3_three_inputs             # im_shape, image, scale_factor
```

---

### tests/image/test_classifier.py — Classifier Class

**Marker:** `@pytest.mark.gpu`

```
class TestClassifierInference:
    test_run_single_image                  # Single image classification
    test_run_batch                         # Batch classification
    test_end2end                           # Full pipeline
    test_get_classifications               # Top-k results

class TestClassifierPostprocessing:
    test_postprocess_returns_probabilities # Probabilities sum ≈ 1
    test_top_k_limits                      # top_k parameter works
```

---

### tests/image/test_depth_estimator.py — DepthEstimator Class

**Marker:** `@pytest.mark.gpu`

```
class TestDepthEstimatorInference:
    test_run_single_image
    test_run_batch
    test_output_is_depth_map               # Spatial dimensions preserved
    test_depth_values_positive
```

---

### tests/image/test_sahi.py — SAHI Integration (NEW)

**Marker:** `@pytest.mark.gpu`

**NOTE:** Investigate what SAHI functionality exists in `src/trtutils/image/sahi.py` or `src/trtutils/compat/sahi/`. If the module exists, test:

```
class TestSAHIIntegration:
    test_sahi_available                    # Module can be imported
    test_sahi_slicing                      # Image slicing works
    test_sahi_merge                        # Detection merging works
    test_sahi_with_detector                # End-to-end with Detector class
```

If SAHI module doesn't exist yet, create placeholder tests that `pytest.skip("SAHI not implemented")`.

---

### Kernel Tests (tests/image/kernels/)

#### tests/image/kernels/conftest.py

```python
"""Kernel test fixtures — CUDA streams, compilation helpers."""
from __future__ import annotations

import pytest


@pytest.fixture
def cuda_stream():
    """Create and destroy CUDA stream for kernel tests."""
    from trtutils.core import create_stream, destroy_stream
    stream = create_stream()
    yield stream
    destroy_stream(stream)
```

#### tests/image/kernels/test_letterbox.py

**Marker:** `@pytest.mark.gpu`
**Port from:** `tests/legacy/image/kernels/test_letterbox_kernel.py`

```
class TestLetterboxKernel:
    test_compiles                          # Kernel compiles without error
    test_correctness_against_cv2ext        # GPU result matches cv2ext.letterbox()

    @parametrize("input_size", [(480, 640), (720, 1280), (100, 200)])
    test_various_input_sizes               # Different input dimensions

    @parametrize("target_size", [(640, 640), (416, 416), (320, 320)])
    test_various_target_sizes              # Different output dimensions
```

#### tests/image/kernels/test_linear.py

**Marker:** `@pytest.mark.gpu`
**Port from:** `tests/legacy/image/kernels/test_linear_kernel.py`

```
class TestLinearResizeKernel:
    test_compiles
    test_correctness_against_cv2            # GPU result matches cv2.resize(INTER_LINEAR)

    @parametrize("input_size", [(480, 640), (720, 1280)])
    test_various_input_sizes

    @parametrize("target_size", [(640, 640), (416, 416)])
    test_various_target_sizes
```

#### tests/image/kernels/test_sst.py

**Marker:** `@pytest.mark.gpu`
**Port from:** `tests/legacy/image/kernels/test_sst_kernel.py`, `test_sst_fast_kernel.py`, `test_sst_imagenet_kernel.py`

Unified test file for all SST variants:

```
# Parametrize across SST variants
SST_KERNELS = [
    pytest.param("sst", id="sst_standard"),
    pytest.param("sst_fast", id="sst_fast"),
    pytest.param("sst_imagenet", id="sst_imagenet"),
]

class TestSSTKernelCompilation:
    @parametrize("kernel_type", SST_KERNELS)
    test_compiles                          # Each variant compiles

    @parametrize("kernel_type", ["sst_fast"])
    @parametrize("precision", ["float32", "float16"])
    test_fast_precision_variants           # Fast kernel has f32 and f16

class TestSSTKernelCorrectness:
    @parametrize("kernel_type", ["sst", "sst_fast"])
    test_correctness_against_cpu_preprocess # GPU matches CPU preprocess()

    test_imagenet_normalization            # SST_IMAGENET applies mean/std correctly

class TestSSTKernelInputVariations:
    @parametrize("kernel_type", SST_KERNELS)
    @parametrize("input_size", [(480, 640), (720, 1280)])
    test_various_sizes                     # Different input dimensions
```

#### tests/image/kernels/test_performance.py

**Marker:** `@pytest.mark.gpu`, `@pytest.mark.performance`
**Port from:** `tests/legacy/image/kernels/test_sst_performance.py`

```
@pytest.mark.performance
class TestKernelPerformance:
    test_sst_faster_than_cpu               # SST kernel faster than CPU preprocess
    test_sst_fast_faster_than_sst          # Fast variant is faster
    test_letterbox_kernel_speed            # Benchmark letterbox kernel
    test_linear_kernel_speed               # Benchmark linear kernel
```

---

### ONNX Preprocessing Tests

#### tests/image/onnx/test_preproc_engine.py

**Marker:** `@pytest.mark.gpu`
**Port from:** `tests/legacy/image/onnx/test_image_preproc.py`

```
class TestTRTPreprocEngine:
    test_trt_preproc_matches_cpu           # TRT preprocessing matches CPU
    test_trt_preproc_imagenet_matches_cpu  # TRT ImageNet preprocessing matches CPU
    test_numerical_tolerance               # Within expected tolerance bounds
```

---

## Final Cleanup

After all image tests pass:

1. Verify all legacy tests have been ported:
   ```bash
   # Compare test function names
   pytest tests/legacy/image/ --collect-only -q | sort > /tmp/legacy.txt
   pytest tests/image/ --collect-only -q | sort > /tmp/new.txt
   diff /tmp/legacy.txt /tmp/new.txt  # Every legacy test should have a counterpart
   ```

2. Verify coverage:
   ```bash
   pytest tests/image/ --cov=src/trtutils/image --cov-branch --cov-report=term-missing -v
   ```

3. Update coverage ratchet:
   ```bash
   python ci/check_coverage.py --update
   ```

4. Remove legacy tests:
   ```bash
   git rm -r tests/legacy/
   ```

5. Update Makefile to remove `test-legacy` target.

6. Run full test suite to verify nothing broken:
   ```bash
   make test-new
   ```

---

## Files Created Summary

| File | Lines (est.) | Tests (est.) | GPU Required | Port Source |
|------|-------------|-------------|--------------|-------------|
| `tests/image/conftest.py` | 200 | — | — | legacy/image/conftest.py |
| `tests/image/test_preproc.py` | 350 | 35+ | Yes | legacy test_preproc + test_model_classes |
| `tests/image/test_postproc.py` | 400 | 35+ | Partial | legacy test_postproc |
| `tests/image/test_image_model.py` | 150 | 15 | Yes | NEW |
| `tests/image/test_detector.py` | 200 | 20 | Yes | NEW + legacy test_model_classes |
| `tests/image/test_classifier.py` | 80 | 6 | Yes | NEW |
| `tests/image/test_depth_estimator.py` | 60 | 4 | Yes | NEW |
| `tests/image/test_sahi.py` | 60 | 4 | Yes | NEW |
| `tests/image/kernels/conftest.py` | 20 | — | Yes | legacy kernels/conftest.py |
| `tests/image/kernels/test_letterbox.py` | 80 | 6 | Yes | legacy test_letterbox_kernel |
| `tests/image/kernels/test_linear.py` | 80 | 6 | Yes | legacy test_linear_kernel |
| `tests/image/kernels/test_sst.py` | 120 | 12 | Yes | legacy test_sst_* (3 files → 1) |
| `tests/image/kernels/test_performance.py` | 80 | 5 | Yes | legacy test_sst_performance |
| `tests/image/onnx/test_preproc_engine.py` | 60 | 3 | Yes | legacy onnx/test_image_preproc |
| **Total** | **~1940** | **~151+** | | |

---

## Post-Port Checklist

- [ ] All legacy test assertions preserved in new structure
- [ ] No legacy test file left unported
- [ ] 100% branch coverage on `src/trtutils/image/`
- [ ] Coverage ratchet updated
- [ ] `tests/legacy/` removed from repo
- [ ] `make test-new` passes
- [ ] Makefile `test-legacy` target removed
- [ ] Pre-commit coverage check passes
