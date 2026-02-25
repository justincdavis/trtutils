# Phase 4: Builder Tests (100% Branch Coverage)

**Branch:** `test-overhaul/phase-4-builder`
**Goal:** Achieve 100% branch coverage on `src/trtutils/builder/` with well-structured, parametrized tests. Add builder to coverage ratchet.

---

## Prerequisites
- Phase 0 (infrastructure)
- Phase 1 (core tests — provides fixtures for engine/stream/memory)
- Phase 3 (coverage ratchet mechanism operational)

## Deliverables
- [ ] `tests/builder/conftest.py` — Builder-specific fixtures
- [ ] `tests/builder/test_build.py` — build_engine() all parameter combos
- [ ] `tests/builder/test_onnx.py` — read_onnx() validation and errors
- [ ] `tests/builder/test_calibrator.py` — EngineCalibrator
- [ ] `tests/builder/test_image_batcher.py` — ImageBatcher with threading
- [ ] `tests/builder/test_synthetic_batcher.py` — SyntheticBatcher
- [ ] `tests/builder/test_dla.py` — DLA analysis and building
- [ ] `tests/builder/test_hooks.py` — YOLO NMS hook
- [ ] `tests/builder/test_onnx_utils.py` — ONNX graph operations
- [ ] `tests/builder/test_progress.py` — ProgressBar
- [ ] Coverage ratchet updated for `trtutils/builder`
- [ ] 100% branch coverage verified

---

## Builder Conftest (tests/builder/conftest.py)

```python
"""Builder test fixtures — ONNX files, temp dirs, test images."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# tests/builder/conftest.py -> tests/builder -> tests -> project_root -> data/
ONNX_PATH = Path(__file__).parent.parent.parent / "data" / "simple.onnx"


@pytest.fixture(scope="session")
def onnx_path() -> Path:
    """Path to the test ONNX model."""
    if not ONNX_PATH.exists():
        pytest.skip("Test ONNX model not found")
    return ONNX_PATH


@pytest.fixture
def output_engine_path(tmp_path) -> Path:
    """Temporary path for built engine output."""
    return tmp_path / "test_output.engine"


@pytest.fixture
def test_image_dir(tmp_path) -> Path:
    """Create a temp directory with synthetic test images."""
    import cv2
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(42)
    for i in range(8):
        img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"test_{i:03d}.jpg"), img)
    return img_dir


@pytest.fixture
def empty_dir(tmp_path) -> Path:
    """Empty directory (no images)."""
    d = tmp_path / "empty"
    d.mkdir()
    return d
```

---

## Detailed Test Plans

### tests/builder/test_build.py — build_engine()

**Marker:** `@pytest.mark.gpu`

#### Branch Coverage Map

| Branch | Condition | Test |
|--------|-----------|------|
| Cache hit | `cache=True` + engine exists in cache | `test_build_with_cache_hit` |
| Cache miss | `cache=True` + not cached | `test_build_with_cache_store` |
| Timing cache: global | `timing_cache=True` or `"global"` | `test_timing_cache_global` |
| Timing cache: local file | `timing_cache=Path(...)` | `test_timing_cache_local` |
| Timing cache: None | `timing_cache=None` | `test_timing_cache_none` |
| Timing cache: invalid | Invalid type | `test_timing_cache_invalid_raises` |
| Device: string "gpu" | `default_device="gpu"` | parametrized |
| Device: string "dla" | `default_device="dla"` | parametrized |
| Device: enum GPU | `default_device=trt.DeviceType.GPU` | parametrized |
| Device: invalid string | `default_device="invalid"` | `test_invalid_device_raises` |
| Hooks applied | `hooks=[hook_fn]` | `test_hooks_applied` |
| Shapes set | `shapes=[("input", (1,3,640,640))]` | `test_manual_shapes` |
| FP16 enabled | `fp16=True` | `test_fp16_build` |
| INT8 enabled | `int8=True` with batcher | `test_int8_build` |
| INT8 no calibrator | `int8=True` without batcher/cache | `test_int8_no_calibrator_warning` |
| Optimization level valid | `optimization_level=3` | default test |
| Optimization level invalid | `optimization_level=6` | `test_invalid_opt_level_raises` |
| Prefer precision constraints | `prefer_precision_constraints=True` | `test_precision_constraints` |
| Reject empty algos | `reject_empty_algorithms=True` | `test_reject_empty` |
| Direct IO auto-enable | tensor formats + `direct_io=False` → auto-enable | `test_direct_io_auto` |
| Input tensor formats | `input_tensor_formats=[...]` | `test_input_formats` |
| Input tensor not found | Format for nonexistent tensor | `test_input_format_not_found_warning` |
| Output tensor formats | `output_tensor_formats=[...]` | `test_output_formats` |
| Layer precision | `layer_precision=[(0, trt.DataType.HALF)]` | `test_layer_precision` |
| Layer precision None | `layer_precision=[(0, None)]` → skip | `test_layer_precision_none_skip` |
| Layer device GPU | `layer_device=[(0, trt.DeviceType.GPU)]` | `test_layer_device_gpu` |
| Layer device DLA invalid | Layer can't run on DLA + no fallback | `test_layer_device_dla_raises` |
| Layer device DLA with fallback | Layer can't + `gpu_fallback=True` | `test_layer_device_dla_fallback` |
| Build serialized | `FLAGS.BUILD_SERIALIZED` | depends on TRT version |
| Build failure | Invalid ONNX → None | `test_build_failure_raises` |
| Progress bar | `verbose=True` + `FLAGS.BUILD_PROGRESS` | `test_progress_bar` |
| Profiling verbosity | `profiling_verbosity=...` | `test_profiling_verbosity` |
| Tiling optimization | `tiling_optimization_level=...` | `test_tiling_opt` if supported |

#### Test Cases

```
class TestBuildEngineBasic:
    test_build_minimal                     # Minimal args: onnx, output
    test_build_creates_file                # Output file exists after build
    test_build_output_is_valid_engine      # Can load result with TRTEngine

class TestBuildEnginePrecision:
    test_build_default_precision           # No FP16/INT8
    test_build_fp16                        # fp16=True
    test_build_int8_with_synthetic_batcher # int8=True + SyntheticBatcher
    test_build_int8_no_calibrator_warning  # int8=True, no batcher → warning

class TestBuildEngineCache:
    test_build_with_cache_stores           # cache=True stores engine
    test_build_cache_hit_skips_build       # Second build uses cache
    test_build_without_cache               # cache=False/None

class TestBuildEngineTimingCache:
    @parametrize("timing_cache", [True, "global", None])
    test_timing_cache_modes                # Global, local, none
    test_timing_cache_local_file           # Path to local file

class TestBuildEngineShapes:
    test_manual_shapes                     # shapes parameter fixes input dims

class TestBuildEngineDevice:
    @parametrize("device", ["gpu", "GPU", "dla", "DLA"])
    test_device_string_variants            # String device types

class TestBuildEngineOptLevel:
    @parametrize("level", [0, 1, 2, 3, 4, 5])
    test_valid_optimization_levels         # All valid levels
    test_invalid_optimization_level        # level=6 → ValueError

class TestBuildEngineHooks:
    test_single_hook                       # One hook function applied
    test_multiple_hooks                    # Multiple hooks chained

class TestBuildEngineErrors:
    test_invalid_onnx_raises               # Bad ONNX file → RuntimeError
    test_build_failure_raises              # Engine build returns None → RuntimeError
```

---

### tests/builder/test_onnx.py — read_onnx()

**Marker:** `@pytest.mark.gpu`

#### Branch Coverage Map

| Branch | Condition | Test |
|--------|-----------|------|
| File not found | Path doesn't exist | `test_file_not_found` |
| Is directory | Path is a directory | `test_is_directory` |
| Wrong extension | `.txt` not `.onnx` | `test_wrong_extension` |
| Old workspace API | `hasattr(config, "max_workspace_size")` | version-dependent |
| New workspace API | `set_memory_pool_limit` | version-dependent |
| Parse success | Valid ONNX parses | `test_parse_success` |
| Parse failure | Invalid ONNX content | `test_parse_failure` |

#### Test Cases

```
class TestReadOnnx:
    test_valid_onnx                        # Returns (network, builder, config, parser)
    test_network_has_inputs_outputs        # Network has I/O tensors
    test_workspace_set                     # Workspace size applied

class TestReadOnnxErrors:
    test_file_not_found                    # FileNotFoundError
    test_is_directory                      # IsADirectoryError
    test_wrong_extension                   # ValueError
    test_invalid_onnx_content              # RuntimeError (parse fails)
```

---

### tests/builder/test_calibrator.py — EngineCalibrator

**Marker:** `@pytest.mark.gpu`

#### Branch Coverage Map

| Method | Branches |
|--------|----------|
| `__init__` | cache_path=None → default, cache_path=Path → custom |
| `set_batcher` | Sets batcher reference |
| `get_batch_size` | batcher exists → batch_size, None → 1 |
| `get_batch` | batcher=None → None, batch=None → None, batch exists → alloc+copy |
| `read_calibration_cache` | cache_path=None → None, file exists → bytes, not exists → None |
| `write_calibration_cache` | Writes to file |

#### Test Cases

```
class TestCalibratorInit:
    test_default_cache_path                # Defaults to calibration.cache
    test_custom_cache_path                 # Uses provided path

class TestCalibratorBatcher:
    test_set_batcher                       # Assigns batcher
    test_get_batch_size_with_batcher       # Returns batcher.batch_size
    test_get_batch_size_no_batcher         # Returns 1

class TestCalibratorBatch:
    test_get_batch_no_batcher              # Returns None
    test_get_batch_exhausted               # Returns None when batcher exhausted
    test_get_batch_returns_gpu_ptr         # Returns [int] (GPU pointer)

class TestCalibratorCache:
    test_read_cache_missing                # Returns None
    test_write_then_read_cache             # Roundtrip works
    test_read_cache_none_path              # cache_path=None → None
```

---

### tests/builder/test_image_batcher.py — ImageBatcher

**Markers:** `@pytest.mark.gpu` (for GPU allocation via calibrator), some tests `@pytest.mark.cpu`

#### Branch Coverage Map

| Branch | Condition | Test |
|--------|-----------|------|
| Invalid resize method | Not "letterbox"/"linear" | `test_invalid_resize_raises` |
| Invalid order | Not "NCHW"/"NHWC" | `test_invalid_order_raises` |
| Dir not found | Path doesn't exist | `test_dir_not_found_raises` |
| Dir is file | Path is a file | `test_dir_is_file_raises` |
| No images found | Empty directory | `test_no_images_raises` |
| No valid batches | Too few images for batch_size | `test_no_valid_batches_raises` |
| max_images < 1 | max_images=0 | `test_max_images_invalid_raises` |
| max_images truncation | max_images > 0 | `test_max_images_truncates` |
| NCHW order | order="NCHW" | parametrized |
| NHWC order | order="NHWC" | parametrized |
| letterbox resize | resize_method="letterbox" | parametrized |
| linear resize | resize_method="linear" | parametrized |
| Image load failure | cv2.imread returns None | `test_image_load_failure` |
| C_CONTIGUOUS check | Array not contiguous → ascontiguousarray | implicit |
| Queue full retry | Put blocks on full queue | `test_batch_prefetch` |
| Event set → shutdown | Shutdown signal stops thread | `test_cleanup_on_close` |

#### Parametrization

```python
@pytest.mark.parametrize("order", ["NCHW", "NHWC"], ids=["nchw", "nhwc"])
@pytest.mark.parametrize("resize_method", ["letterbox", "linear"], ids=["letterbox", "linear"])
def test_batch_shape(test_image_dir, order, resize_method):
    ...
```

#### Test Cases

```
class TestImageBatcherInit:
    test_valid_init                         # Creates batcher successfully
    test_invalid_resize_method             # ValueError
    test_invalid_order                     # ValueError
    test_dir_not_found                     # FileNotFoundError
    test_dir_is_file                       # NotADirectoryError
    test_no_images                         # ValueError
    test_no_valid_batches                  # ValueError (batch_size > num_images)
    test_max_images_invalid                # ValueError (max_images < 1)
    test_max_images_truncates              # Only max_images loaded

class TestImageBatcherOutput:
    @parametrize("order", ["NCHW", "NHWC"])
    @parametrize("resize_method", ["letterbox", "linear"])
    test_batch_shape                       # Output shape matches (N,C,H,W) or (N,H,W,C)
    test_batch_dtype                       # Output dtype matches requested
    test_batch_count                       # num_batches matches expected

class TestImageBatcherIteration:
    test_get_all_batches                   # Can iterate through all batches
    test_exhausted_returns_none            # Returns None after all batches
    test_num_batches_property              # Matches actual count
    test_batch_size_property               # Returns configured batch size

class TestImageBatcherThreading:
    test_prefetch_queue                    # Batches prefetched in background
    test_cleanup_on_close                  # Thread stops on close
    test_atexit_registered                 # Cleanup registered

class TestImageBatcherValidation:
    @parametrize("dtype", [np.float32, np.float16])
    test_dtype_handling                    # Different dtypes work
    test_c_contiguous_output               # Output is always C-contiguous
```

---

### tests/builder/test_synthetic_batcher.py — SyntheticBatcher

**Marker:** `@pytest.mark.cpu`

#### Test Cases

```
class TestSyntheticBatcherInit:
    test_valid_init                         # Creates batcher
    test_invalid_order                     # ValueError
    test_invalid_num_batches               # ValueError (num_batches < 1)

class TestSyntheticBatcherOutput:
    @parametrize("order", ["NCHW", "NHWC"])
    test_batch_shape                       # Shape matches config
    @parametrize("dtype", [np.float32, np.float16, np.int8])
    test_batch_dtype                       # Dtype matches config
    test_data_range                        # Values within configured range

class TestSyntheticBatcherIteration:
    @parametrize("num_batches", [1, 5, 10])
    test_correct_batch_count               # Exact num_batches batches
    test_exhausted_returns_none            # None after all batches
    test_num_batches_property              # Returns configured count
    test_batch_size_property               # Returns configured size
    test_c_contiguous_output               # Output is C-contiguous
```

---

### tests/builder/test_dla.py — DLA Utilities

**Marker:** `@pytest.mark.gpu`

#### Test Cases

```
class TestCanRunOnDla:
    test_returns_tuple                     # Returns (bool, list)
    test_chunks_structure                  # Each chunk is (layers, start, end, on_dla)
    test_with_onnx_path                    # Accepts Path to ONNX
    test_with_network                      # Accepts trt.INetworkDefinition + config
    test_without_config_raises             # ValueError if network + no config
    test_verbose_layers                    # verbose_layers=True works
    test_verbose_chunks                    # verbose_chunks=True works

class TestBuildDlaEngine:
    # NOTE: These tests only run on Jetson or can be tested with gpu_fallback=True
    test_build_basic                       # Builds engine with DLA
    test_full_dla_path                     # Model fully on DLA → single build
    test_no_dla_chunks                     # No DLA layers → GPU-only build
    test_mixed_dla_gpu                     # Partial DLA → layer assignments
    test_max_chunks_limit                  # max_chunks parameter respected
    test_min_layers_filter                 # min_layers parameter respected

class TestGetCheckDla:
    test_returns_callable                  # Returns a function
    test_function_accepts_layer            # Can call with ILayer
```

---

### tests/builder/test_hooks.py — YOLO NMS Hook

**Marker:** `@pytest.mark.gpu`

#### Test Cases

```
class TestYoloEfficientNmsHook:
    test_returns_callable                  # Returns a function
    test_hook_modifies_network             # Network has 4 outputs after hook
    test_output_names                      # num_dets, det_boxes, det_scores, det_classes
    test_yolov10_passthrough               # (1,300,6) output → no modification
    test_with_objectness                   # YOLOv7/X format (N+5 channels)
    test_without_objectness                # YOLOv8+ format (N+4 channels)

    @parametrize("box_coding", ["center_size", "corner"])
    test_box_coding_modes                  # Both box coding modes

    @parametrize("class_agnostic", [True, False, None])
    test_class_agnostic                    # Class-agnostic NMS option

class TestMakePluginField:
    test_float_value                       # float → FLOAT32
    test_int_value                         # int → INT32
    test_float_list                        # [float] → FLOAT32
    test_int_list                          # [int] → INT32
```

---

### tests/builder/test_onnx_utils.py — ONNX Graph Operations

**Marker:** `@pytest.mark.cpu` (pure graph manipulation, no GPU)

#### Test Cases

```
class TestExtractSubgraph:
    test_extract_middle_nodes              # Extract nodes from middle of graph
    test_extract_first_nodes               # Start from index 0
    test_extract_last_nodes                # End at last node
    test_invalid_start_idx                 # start_idx < 0 → ValueError
    test_invalid_end_idx                   # end_idx >= num_nodes → ValueError
    test_start_greater_than_end            # start > end → ValueError
    test_result_is_valid_onnx              # Output is valid ONNX model
    test_accepts_graph_surgeon_graph       # Works with gs.Graph input

class TestSplitModel:
    test_split_into_two                    # Single split point
    test_split_into_three                  # Two split points
    test_empty_indices_raises              # Empty list → ValueError
    test_invalid_index_raises              # Out of range → ValueError
    test_subgraphs_are_valid_onnx          # Each part is valid ONNX

class TestFileOperations:
    test_extract_subgraph_from_file        # File → subgraph → file
    test_split_model_from_file             # File → parts → files
```

**Note:** These tests need a small ONNX model to operate on. Create a simple one with `onnx.helper` or use `data/simple.onnx` from the project root.

---

### tests/builder/test_progress.py — ProgressBar

**Marker:** `@pytest.mark.cpu`

#### Test Cases

```
class TestProgressBar:
    test_init                              # Creates without error
    test_phase_start                       # Registers phase
    test_phase_finish                      # Closes phase cleanly
    test_step_complete                     # Updates progress
    test_step_complete_returns_true        # Returns True (continue)
    test_nested_phases                     # Child phases indent
    test_phase_finish_cleanup              # Removes all tracking

class TestProgressBarEdgeCases:
    test_step_zero_delta                   # step_diff=0 → no update
    test_unknown_phase_step                # step_complete on unknown phase
    test_delete_closes_bars                # __del__ closes tqdm bars
```

---

## Coverage Verification

```bash
pytest tests/builder/ \
    --cov=src/trtutils/builder \
    --cov-branch \
    --cov-report=term-missing \
    -v
```

Then update ratchet:
```bash
python ci/check_coverage.py --update
```

---

## Files Created Summary

| File | Lines (est.) | Tests (est.) | GPU Required |
|------|-------------|-------------|--------------|
| `tests/builder/conftest.py` | 50 | — | Partial |
| `tests/builder/test_build.py` | 400 | 30+ | Yes |
| `tests/builder/test_onnx.py` | 100 | 8 | Yes |
| `tests/builder/test_calibrator.py` | 120 | 10 | Yes |
| `tests/builder/test_image_batcher.py` | 250 | 25 | Partial |
| `tests/builder/test_synthetic_batcher.py` | 120 | 15 | No |
| `tests/builder/test_dla.py` | 150 | 12 | Yes |
| `tests/builder/test_hooks.py` | 150 | 12 | Yes |
| `tests/builder/test_onnx_utils.py` | 150 | 12 | No |
| `tests/builder/test_progress.py` | 100 | 9 | No |
| **Total** | **~1590** | **~133** | |
